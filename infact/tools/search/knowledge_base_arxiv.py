"""
This file contains the ArxivKnowledgeBase class, which is used to build and search the arXiv Knowledge Base.
Download the arXiv dataset from https://www.kaggle.com/datasets/Cornell-University/arxiv and place in data/arxiv/arxiv-metadata-oai-snapshot.json
"""

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
from infact.tools.search.knowledge_base import KnowledgeBase, SearchResult

class ArxivKnowledgeBase:
    """The arXiv Knowledge Base (KB) used to retrieve relevant papers."""
    name = 'arxiv_kb'
    CHUNK_SIZE = 100_000

    def __init__(
        self,
        variant: str = "arxiv",
        logger = None,
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
        
        # Now load or build after all attributes are initialized
        if self._is_built():
            self._load()
        else:
            self._build()
            
    def _load_or_init_metadata(self) -> Dict:
        """Load existing metadata or initialize new metadata including paper ID scanning."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'rb') as f:
                return pickle.load(f)
        
        metadata = {
            'processed_paper_ids': set(),
            'embedding_to_id': [],  # List maintaining exact order of embeddings in FAISS index
            'dimension': self.model.get_sentence_embedding_dimension(),
            'all_paper_ids': set()  # Store all valid paper IDs
        }
        
        print("Scanning source file for paper IDs...")
        with open(self.src_data_path, 'r') as f:
            for line in f:
                paper = json.loads(line)
                if 'abstract' in paper and paper['abstract']:
                    metadata['all_paper_ids'].add(paper['id'])
        
        return metadata

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
        
        try:
            index = faiss.read_index(str(self.index_path))
            # Verify index size matches our embedding_to_id list
            if index.ntotal != len(self.metadata['embedding_to_id']):
                return False
            # Verify we've processed all papers
            return len(self.metadata['processed_paper_ids']) == len(self.metadata['all_paper_ids'])
        except Exception as e:
            logging.error(f"Error reading index: {e}")
            return False

    def _embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.model.encode(text, normalize_embeddings=True)

    def _embed_many(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts efficiently."""
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32)

    def _build(self):
        """Creates the FAISS index for the arXiv abstracts."""
        print("Building the arXiv knowledge base...")
        
        # Calculate papers to process using set difference
        papers_to_process_ids = self.metadata['all_paper_ids'] - self.metadata['processed_paper_ids']
        
        if not papers_to_process_ids:
            print("No new papers to process.")
            return
        
        print(f"Processing {len(papers_to_process_ids)} new papers...")
        
        # Create lookup set for faster membership testing
        papers_to_process_set = set(papers_to_process_ids)
        papers_chunk = []
        
        # Initialize or load existing FAISS index
        if self.index_path.exists():
            print("Loading existing index...")
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatL2(self.metadata['dimension'])
        
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        total_papers = sum(1 for line in open(self.src_data_path, 'r'))
        with tqdm(total=len(papers_to_process_ids), desc="Processing papers") as pbar:
            with open(self.src_data_path, 'r') as f:
                for line in tqdm(f, total=total_papers, desc="Scanning file", leave=False):
                    paper = json.loads(line)
                    if paper['id'] in papers_to_process_set:
                        papers_chunk.append(paper)
                        
                        # Process when chunk is full
                        if len(papers_chunk) >= self.CHUNK_SIZE:
                            self._process_chunk(papers_chunk)
                            pbar.update(len(papers_chunk))
                            papers_chunk = []
                
                # Process remaining papers
                if papers_chunk:
                    self._process_chunk(papers_chunk)
                    pbar.update(len(papers_chunk))

        print("[green]Successfully built the arXiv knowledge base![/green]")

    def _process_chunk(self, papers: List[Dict]):
        """Process a chunk of papers and add to index."""
        # Move index to CPU before embedding to free GPU memory
        if torch.cuda.is_available():
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            self.index = index_cpu

        # Do embeddings with freed GPU memory
        abstracts  = [paper['abstract'] for paper in papers]
        paper_ids  = [paper['id'] for paper in papers]
        embeddings = self._embed_many(abstracts)
        
        # Add embeddings on CPU
        self.index.add(embeddings)
        
        # Save metadata and index while on CPU
        self.metadata['embedding_to_id'].extend(paper_ids)
        self.metadata['processed_paper_ids'].update(paper_ids)
        self._save_metadata()
        faiss.write_index(self.index, str(self.index_path))
        
        # Move back to GPU for next operations if needed
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def _load(self):
        """Load the FAISS index."""
        self.index = faiss.read_index(str(self.index_path))
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def _call_api(self, query: str, limit: int) -> List[SearchResult]:
        """Match parent class search interface"""
        query_embedding = self._embed(query).reshape(1, -1)
        D, I = self.index.search(query_embedding, limit)
        
        results = []
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            # Use direct lookup from embedding index to paper_id
            paper_id = self.metadata['embedding_to_id'][idx]
            url, text, date = self.retrieve(paper_id)
            if text:
                results.append(SearchResult(
                    source=url,
                    text=text,
                    query=query,
                    rank=i,
                    date=date
                ))
        return results

    def retrieve(self, paper_id: str) -> Tuple[str, str, Optional[datetime]]:
        """Match parent class retrieve interface"""
        with open(self.src_data_path, 'r') as f:
            for line in f:
                paper = json.loads(line)
                if paper['id'] == paper_id:
                    return paper['id'], paper['abstract'], None
        return None, None, None