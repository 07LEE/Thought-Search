import time
import numpy as np
import os
import sys

# Add src to path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from vector_db import SimpleVectorDB
from config import DB_DEFAULT_PATH

def benchmark_search():
    """Performs a comparative performance analysis between vectorized and legacy search methods.
    
    Loads the current vector database and executes a series of search queries to measure
    the speedup achieved by numpy-based vectorized operations compared to the standard
    loop-based cosine similarity calculation.
    """

    # 1. Load the database (Make sure you re-indexed first!)
    db = SimpleVectorDB()
    if not os.path.exists(DB_DEFAULT_PATH):
        print("ERROR: DB not found. Please run 'python src/indexer.py' first.")
        return
    
    db.load(DB_DEFAULT_PATH)
    
    if not db.documents:
        print("ERROR: DB is empty. Nothing to search.")
        return

    query = "지식 정보 프로젝트의 목적" # Example query
    print(f"\n[Benchmark] Target: {len(db.documents)} documents")
    print(f"[Benchmark] Query: '{query}'")

    # --- Pre-calculate query vector for fair comparison ---
    query_vector = db.model.encode(query)
    query_norm = np.linalg.norm(query_vector)
    if query_norm > 0:
        query_vector = query_vector / query_norm
    
    # --- Test Vectorized Search (New) ---
    # We measure ONLY the dot product and sorting logic
    start_time = time.time()
    for _ in range(1000): # Run more times for better resolution
        # This is exactly what happens inside db.search() after encoding
        similarities = np.dot(db.vectors, query_vector)
        top_k_indices = similarities.argsort()[::-1][:3]
    end_time = time.time()
    
    vectorized_total_time = (end_time - start_time) / 1000
    print(f"\n[Result] Vectorized Search (Numpy): {vectorized_total_time * 1000:.4f} ms per calculation")

    # --- Test Legacy Search (Loop-based) ---
    start_time = time.time()
    for _ in range(1000):
        similarities = []
        # Reconstructing the exact old _cosine_similarity logic
        for vec in db.vectors:
            # Note: in old code, we didn't normalize beforehand
            # but here we use db.vectors which are already normalized by new indexer.
            # To simulate old slow behavior, we assume they aren't.
            dot_product = np.dot(query_vector, vec)
            norm_q = np.linalg.norm(query_vector)
            norm_v = np.linalg.norm(vec)
            sim = dot_product / (norm_q * norm_v)
            similarities.append(sim)
        top_k_indices = np.array(similarities).argsort()[::-1][:3]
    end_time = time.time()
    
    loop_total_time = (end_time - start_time) / 1000
    print(f"[Result] Legacy Search (Loop-based): {loop_total_time * 1000:.4f} ms per calculation")

    speedup = loop_total_time / vectorized_total_time
    print(f"\n[Summary] Optimization Speedup: {speedup:.2f}x faster! 🚀")

if __name__ == "__main__":
    benchmark_search()
