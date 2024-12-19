from pathlib import Path
import os
import json
import torch
import faiss
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datetime import datetime
from rich import print
import logging
import pickle

class ArxivKnowledgeBase:
    """The arXiv Knowledge Base (KB) used to retrieve relevant papers."""
    name = 'arxiv_kb'
    CHUNK_SIZE = 100_000

    def __init__(
        self,
        data_base_dir: str    = "data/",
        embedding_model: str  = "Alibaba-NLP/gte-base-en-v1.5",
        device: Optional[str] = None
    ):
        # Setup paths and dirs
        self.kb_dir = Path(data_base_dir + "arxiv/knowledge_base/")
        self.src_data_path = data_base_dir + "arxiv/arxiv-metadata-oai-snapshot.json"
        os.makedirs(self.kb_dir, exist_ok=True)
        self.checkpoint_dir = self.kb_dir / "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.index_path = self.kb_dir / "arxiv.faiss"
        self.metadata_path = self.kb_dir / "metadata.pkl"
        
        # Setup model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            embedding_model, 
            trust_remote_code=True, 
            config_kwargs=dict(resume_download=None), 
            device=self.device
        )
        
        # Initialize index and metadata
        self.index = None
        self.metadata = self._load_or_init_metadata()
        
        if not self._is_built():
            self._build()
        else:
            self._load()
            
    def _load_or_init_metadata(self) -> Dict:
        if self.metadata_path.exists():
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                # Add last_chunk_id if it doesn't exist in older metadata
                if 'last_chunk_id' not in metadata:
                    metadata['last_chunk_id'] = -1
                return metadata
        return {
            'processed_papers': 0,
            'paper_ids': [],
            'total_papers': 0,
            'dimension': self.model.get_sentence_embedding_dimension(),
            'last_chunk_id': -1
        }

    def _save_metadata(self):
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def _save_checkpoint(self, chunk_id: int, embeddings: np.ndarray, paper_ids: List[str]):
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{chunk_id}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'paper_ids': paper_ids,
                'chunk_id': chunk_id
            }, f)

    def _load_checkpoint(self, chunk_id: int) -> Optional[Dict]:
        """Load processing checkpoint if exists."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{chunk_id}.pkl"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        return None

    def _is_built(self) -> bool:
        """Check if the knowledge base is fully built and contains all papers."""
        if not (self.index_path.exists() and self.metadata_path.exists()):
            return False
            
        # Count total papers in source file
        with open(self.src_data_path, 'r') as f:
            total_papers = sum(1 for _ in f)
            
        # Load index to check its size
        try:
            index = faiss.read_index(str(self.index_path))
            print("total_papers", total_papers, "index.ntotal", index.ntotal, "processed_papers", self.metadata['processed_papers'], "len(paper_ids)", len(self.metadata['paper_ids']))
            return index.ntotal == total_papers
        except Exception as e:
            logging.error(f"Error reading index: {e}")
            return False

    def _embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.model.encode(text, normalize_embeddings=True)

    def _embed_many(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts efficiently."""
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    def _build(self):
        """Creates the FAISS index for the arXiv abstracts."""
        print("Building the arXiv knowledge base...")
        
        # Count total papers if not already counted
        with open(self.src_data_path, 'r') as f:
            self.metadata['total_papers'] = sum(1 for _ in f)
            self._save_metadata()

        # Initialize or load existing FAISS index
        if self.index_path.exists():
            print("Loading existing index...")
            self.index = faiss.read_index(str(self.index_path))
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        elif self.index is None:
            self.index = faiss.IndexFlatL2(self.metadata['dimension'])
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        # Outer Chunking Loop
        total_chunks = (self.metadata['total_papers'] + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
        with tqdm(total=total_chunks, desc="Overall Progress") as pbar:
            # Update progress bar for already processed chunks
            start_chunk = self.metadata['last_chunk_id'] + 1
            pbar.update(start_chunk)

            # Individual Chunk Loop
            for chunk_id, start_idx in enumerate(range(start_chunk * self.CHUNK_SIZE, self.metadata['total_papers'], self.CHUNK_SIZE), start=start_chunk):
                # Check for existing checkpoint
                checkpoint = self._load_checkpoint(chunk_id)
                if checkpoint is not None:
                    print(f"Resuming from checkpoint {chunk_id}")
                    embeddings = checkpoint['embeddings']
                    paper_ids = checkpoint['paper_ids']
                else:
                    end_idx = min(start_idx + self.CHUNK_SIZE, self.metadata['total_papers'])
                    papers = []
                    paper_ids = []
                    
                    with open(self.src_data_path, 'r') as f:
                        for i, line in enumerate(f):
                            if i >= start_idx and i < end_idx:
                                paper = json.loads(line)
                                if 'abstract' in paper and paper['abstract']:
                                    papers.append(paper['abstract'])
                                    paper_ids.append(paper['id'])

                    embeddings = self._embed_many(papers)
                    self._save_checkpoint(chunk_id, embeddings, paper_ids)

                # Add to index
                self.index.add(embeddings)
                self.metadata['paper_ids'].extend(paper_ids)
                self.metadata['processed_papers'] += len(paper_ids)
                self._save_metadata()
                pbar.update(1)

                # Clean up checkpoint after successful processing
                checkpoint_path = self.checkpoint_dir / f"checkpoint_{chunk_id}.pkl"
                if checkpoint_path.exists():
                    checkpoint_path.unlink()

                # Save index after each chunk
                if torch.cuda.is_available():
                    index_cpu = faiss.index_gpu_to_cpu(self.index)
                else:
                    index_cpu = self.index
                faiss.write_index(index_cpu, str(self.index_path))

                # Update metadata with last processed chunk
                self.metadata['last_chunk_id'] = chunk_id
                self._save_metadata()
                pbar.update(1)

        print("[green]Successfully built the arXiv knowledge base![/green]")

    def _load(self):
        """Load the FAISS index."""
        self.index = faiss.read_index(str(self.index_path))
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for papers using the query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing paper id, abstract, and search score
        """
        query_embedding = self._embed(query).reshape(1, -1)
        D, I = self.index.search(query_embedding, limit)
        
        results = []
        for idx, (distance, paper_idx) in enumerate(zip(D[0], I[0])):
            if paper_idx < len(self.metadata['paper_ids']):
                paper_id = self.metadata['paper_ids'][paper_idx]
                _, abstract, _ = self.retrieve(paper_id)
                if abstract:
                    results.append({
                        'id': paper_id,
                        'abstract': abstract,
                        'score': float(distance)
                    })

        return results

    def retrieve(self, paper_id: str) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
        """
        Retrieve a specific paper by ID.
        
        Returns:
            Tuple of (paper_id, abstract, datetime) or (None, None, None) if not found
        """
        with open(self.src_data_path, 'r') as f:
            for line in f:
                paper = json.loads(line)
                if paper['id'] == paper_id:
                    return paper['id'], paper['abstract'], None
        return None, None, None