"""Builds the arXiv Knowledge Base (KB) for paper search."""

from infact.tools.search.knowledge_base_arxiv import ArxivKnowledgeBase


if __name__ == '__main__':  # KB building uses multiprocessing
    print("Starting to build the arXiv Knowledge Base...")
    kb = ArxivKnowledgeBase()
    
    # Run sanity check
    kb.current_claim_id = 0
    result = kb.search("quantum computing", limit=10)
    print(result)
    assert len(result) == 10

    print("arXiv KB is ready for usage!")