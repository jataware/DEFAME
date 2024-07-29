from src.tools.search.knowledge_base import KnowledgeBase


if __name__ == '__main__':  # KB building uses multiprocessing
    kb = KnowledgeBase("train")

    # Run simple test
    kb.current_claim_id = 0
    print(kb.search("Apple", limit=10))