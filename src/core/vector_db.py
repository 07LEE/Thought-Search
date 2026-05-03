import json
import os
import warnings
import logging
import numpy as np
from collections import Counter
from sentence_transformers import CrossEncoder

# Hide internal warning messages
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from .config import EMBEDDING_MODEL, RERANK_MODEL
from .engines import SparseIndex, DenseIndex

class SimpleVectorDB:
    def __init__(self, model_name=None, rerank_model_name=None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.rerank_model_name = rerank_model_name or RERANK_MODEL
        
        # Initialize specialized engines
        self.sparse_engine = SparseIndex()
        self.dense_engine = DenseIndex(self.model_name)
        
        # Lazy loading for reranker to save memory if not used
        self.reranker = None

        self.documents = []
        self.metadata = []
        self.file_hashes = {}

    def pre_load_models(self):
        """Pre-loads the embedding and re-ranking models into memory."""
        print(f"LOGE: [VectorDB] Pre-loading models...")
        # Dense engine already initializes its model in its constructor
        if self.reranker is None:
            self.reranker = CrossEncoder(self.rerank_model_name)
        print(f"LOGE: [VectorDB] All models loaded.")

    def add_texts(self, texts, metadatas=None):
        """Calculates embeddings for a list of texts, normalizes them, and adds them to the database.

        Args:
            texts (list[str]): A list of string texts to embed and add.
            metadatas (list[dict], optional): A list of metadata dictionaries corresponding to the texts.
        """
        print(f"LOGE: [VectorDB] Processing {len(texts)} texts...")
        
        # 1. Update Dense (Vector) Engine
        new_vectors = self.dense_engine.embed(texts)
        self.dense_engine.add_vectors(new_vectors)

        # 2. Update Common Data
        self.documents.extend(texts)
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in range(len(texts))])
        
        # 3. Update Sparse (BM25) Engine
        self.sparse_engine.rebuild(self.documents)
        print("LOGE: [VectorDB] Done adding texts.")

    def remove_by_filename(self, filename):
        """Removes all entries associated with a given source filename.

        Args:
            filename: The basename of the source file to remove.

        Returns:
            The number of entries removed.
        """
        indices_to_keep = [
            i for i, m in enumerate(self.metadata) 
            if m.get("rel_path", m.get("filename")) != filename
        ]
        removed_count = len(self.documents) - len(indices_to_keep)

        if removed_count == 0:
            return 0

        # Update common data
        self.documents = [self.documents[i] for i in indices_to_keep]
        self.metadata = [self.metadata[i] for i in indices_to_keep]
        
        # Update Dense Engine vectors
        all_vectors = self.dense_engine.get_vectors()
        if all_vectors is not None and len(indices_to_keep) > 0:
            self.dense_engine.set_vectors(all_vectors[indices_to_keep])
        else:
            self.dense_engine.set_vectors(None)

        # Update Sparse Engine index
        self.sparse_engine.rebuild(self.documents)
        return removed_count

    def search(self, query, top_k=3):
        """Searches the database for the most similar documents to the given query using vectorized operations.

        Args:
            query (str): The search query text.
            top_k (int, optional): The number of top results to return. Defaults to 3.

        Returns:
            list[dict]: A list of dictionaries containing the score, text, and metadata for each result.
        """
        return self.dense_engine.search(query, self.documents, self.metadata, top_k=top_k)

    def search_bm25(self, query, top_k=3):
        """Searches the database using the BM25 algorithm.

        Args:
            query (str): The search query text.
            top_k (int, optional): The number of top results to return. Defaults to 3.

        Returns:
            list[dict]: A list of results with scores, text, and metadata.
        """
        return self.sparse_engine.search(query, self.documents, self.metadata, top_k=top_k)

    def search_hybrid(self, query, top_k=3, k_factor=20):
        """Combines Vector and BM25 search results using Reciprocal Rank Fusion (RRF).

        Args:
            query (str): The search query text.
            top_k (int, optional): The number of top results to return. Defaults to 3.
            k_factor (int, optional): Smoothing factor for RRF calculation. Defaults to 20.

        Returns:
            list[dict]: A list of combined and ranked results.
        """
        # 1. Get results from both methods
        internal_k = max(top_k * 2, 20)
        vector_results = self.search(query, top_k=internal_k)
        bm25_results = self.search_bm25(query, top_k=internal_k)
        
        # 2. Reciprocal Rank Fusion (RRF)
        rrf_scores = Counter()
        for rank, res in enumerate(vector_results, 1):
            rrf_scores[res["index"]] += 1.0 / (k_factor + rank)
        for rank, res in enumerate(bm25_results, 1):
            rrf_scores[res["index"]] += 1.0 / (k_factor + rank)
            
        # 3. Combine and sort
        combined_results = []
        all_candidates = vector_results + bm25_results
        id_to_obj = {res["index"]: res for res in all_candidates}
        
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        for idx, rrf_score in sorted_indices[:top_k]:
            res = id_to_obj[idx]
            res["hybrid_score"] = rrf_score
            combined_results.append(res)
            
        return combined_results

    def rerank(self, query, results):
        """Refines search results using a Cross-Encoder for better accuracy.

        Args:
            query (str): The search query text.
            results (list[dict]): Initial search results to rerank.

        Returns:
            list[dict]: Reranked results with new scores.
        """
        if not results:
            return []
            
        if self.reranker is None:
            self.reranker = CrossEncoder(self.rerank_model_name)
            
        pairs = [[query, res["text"]] for res in results]
        rerank_scores = self.reranker.predict(pairs)
        
        for i, res in enumerate(results):
            res["rerank_score"] = float(rerank_scores[i])
            
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results

    def save(self, filepath):
        """Saves the database state to a JSON file and vectors to a binary .npy file.

        Args:
            filepath (str): The destination file path for the JSON metadata.
        """
        vector_path = filepath.rsplit('.', 1)[0] + ".vectors.npy"
        
        data = {
            "model_name": self.model_name,
            "documents": self.documents,
            "metadata": self.metadata,
            "file_hashes": self.file_hashes,
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        vectors = self.dense_engine.get_vectors()
        if vectors is not None:
            np.save(vector_path, vectors)
            
        print(f"LOGE: [VectorDB] Saved DB to {filepath}")

    def load(self, filepath):
        """Loads the database state from JSON and binary files.

        Args:
            filepath (str): The source JSON file path.
        """
        if not os.path.exists(filepath):
            print("LOGE: [VectorDB] No existing DB found. Starting fresh.")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        saved_model = data.get("model_name", self.model_name)
        if saved_model and saved_model != self.model_name:
            import sys
            print(f"LOGE: [VectorDB] ERROR: Model mismatch!")
            sys.exit(1)

        self.documents = data.get("documents", [])
        self.metadata = data.get("metadata", [])
        self.file_hashes = data.get("file_hashes", {})
        
        # Load vectors
        vector_path = filepath.rsplit('.', 1)[0] + ".vectors.npy"
        if os.path.exists(vector_path):
            self.dense_engine.set_vectors(np.load(vector_path))
        
        # Rebuild Sparse Index
        self.sparse_engine.rebuild(self.documents)
