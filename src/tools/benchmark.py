import time
import numpy as np
import os
import sys

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_db import SimpleVectorDB
from config import DB_DEFAULT_PATH

def benchmark_search():
    """Performs a performance analysis of the search engine.
    
    Measures the speed of hybrid search across the current document base.
    """

    # 1. Load the database
    db = SimpleVectorDB()
    if not os.path.exists(DB_DEFAULT_PATH):
        print(f"ERROR: DB not found at {DB_DEFAULT_PATH}. Please run indexer first.")
        return
    
    db.load(DB_DEFAULT_PATH)
    
    if not db.documents:
        print("ERROR: DB is empty. Nothing to search.")
        return

    query = "지식 정보 프로젝트의 목적" # Example query
    print(f"\n{'-'*50}")
    print(f"🚀 Thought-Search Performance Benchmark")
    print(f"{'-'*50}")
    print(f"[*] Target DB: {len(db.documents)} documents")
    print(f"[*] Query: '{query}'")
    print(f"[*] Model: {db.model_name}")

    # 2. Benchmark Dense (Vector) Search
    start_time = time.time()
    for _ in range(100):
        db.search(query, top_k=5)
    dense_time = (time.time() - start_time) / 100
    print(f"\n[Dense Search]  Avg Time: {dense_time*1000:.4f} ms")

    # 3. Benchmark Sparse (BM25) Search
    start_time = time.time()
    for _ in range(100):
        db.search_bm25(query, top_k=5)
    sparse_time = (time.time() - start_time) / 100
    print(f"[Sparse Search] Avg Time: {sparse_time*1000:.4f} ms")

    # 4. Benchmark Hybrid (RRF) Search
    start_time = time.time()
    for _ in range(100):
        db.search_hybrid(query, top_k=5)
    hybrid_time = (time.time() - start_time) / 100
    print(f"[Hybrid Search] Avg Time: {hybrid_time*1000:.4f} ms")
    
    print(f"{'-'*50}")
    print(f"✔ Benchmark complete.\n")

if __name__ == "__main__":
    benchmark_search()
