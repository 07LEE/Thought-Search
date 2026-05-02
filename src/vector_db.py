import json
import os
import warnings
import logging
import numpy as np
import re
import math
from collections import Counter
from sentence_transformers import SentenceTransformer, CrossEncoder

# Hide internal warning messages
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


from config import EMBEDDING_MODEL, RERANK_MODEL

class SimpleVectorDB:
    def __init__(self, model_name=None, rerank_model_name=None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.rerank_model_name = rerank_model_name or RERANK_MODEL
        # print(f"LOGE: [VectorDB] Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        
        # Lazy loading for reranker to save memory if not used
        self.reranker = None

        self.documents = []
        self.metadata = []
        self.vectors = None
        self.file_hashes = {}
        
        # BM25 related data
        self.bm25_data = {
            "tf": [],          # Term frequencies per document
            "idf": {},         # Inverse document frequency
            "avgdl": 0,        # Average document length
            "doc_lengths": []  # Length of each document
        }

    def _cosine_similarity(self, vec_a, vec_b):
        """Calculates the cosine similarity between two vectors.

        Args:
            vec_a (np.ndarray): The first vector.
            vec_b (np.ndarray): The second vector.

        Returns:
            float: The cosine similarity score.
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot_product / (norm_a * norm_b)

    def _tokenize(self, text):
        """Simple tokenizer for BM25.

        Args:
            text (str): The raw text to tokenize.

        Returns:
            list[str]: A list of alphanumeric tokens in lowercase.
        """
        # Lowercase and split by non-alphanumeric characters
        return re.findall(r'\w+', text.lower())

    def _rebuild_bm25_index(self):
        """Rebuilds the BM25 index from current documents.
        
        Calculates Term Frequencies (TF), Inverse Document Frequencies (IDF),
        and average document length (avgdl) for the BM25 algorithm.
        """
        if not self.documents:
            self.bm25_data = {"tf": [], "idf": {}, "avgdl": 0, "doc_lengths": []}
            return

        print("LOGE: [VectorDB] Rebuilding BM25 index...")
        doc_tokens = [self._tokenize(doc) for doc in self.documents]
        self.bm25_data["doc_lengths"] = [len(tokens) for tokens in doc_tokens]
        self.bm25_data["avgdl"] = sum(self.bm25_data["doc_lengths"]) / len(self.documents)
        
        # Calculate TF
        self.bm25_data["tf"] = [Counter(tokens) for tokens in doc_tokens]
        
        # Calculate IDF
        num_docs = len(self.documents)
        word_doc_counts = Counter()
        for tokens in doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                word_doc_counts[token] += 1
        
        self.bm25_data["idf"] = {}
        for word, count in word_doc_counts.items():
            # Standard BM25 IDF: log(1 + (N - n + 0.5) / (n + 0.5))
            self.bm25_data["idf"][word] = math.log(1 + (num_docs - count + 0.5) / (count + 0.5))
        
        print(f"LOGE: [VectorDB] BM25 index rebuilt for {num_docs} documents.")

    def add_texts(self, texts, metadatas=None):
        """Calculates embeddings for a list of texts, normalizes them, and adds them to the database.

        Args:
            texts (list[str]): A list of string texts to embed and add.
            metadatas (list[dict], optional): A list of metadata dictionaries corresponding to the texts.
        """
        print(f"LOGE: [VectorDB] Embedding {len(texts)} texts...")
        new_vectors = self.model.encode(texts)
        
        # --- L2 Normalization ---
        # We normalize vectors to unit length (norm=1). 
        # This allows us to calculate Cosine Similarity using simple Dot Product later.
        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        new_vectors = new_vectors / (norms + 1e-10) # Avoid division by zero

        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack((self.vectors, new_vectors))

        self.documents.extend(texts)
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in range(len(texts))])
        
        # Rebuild BM25 index when documents are added
        self._rebuild_bm25_index()
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

        self.documents = [self.documents[i] for i in indices_to_keep]
        self.metadata = [self.metadata[i] for i in indices_to_keep]
        if self.vectors is not None and len(indices_to_keep) > 0:
            self.vectors = self.vectors[indices_to_keep]
        else:
            self.vectors = None

        # Rebuild BM25 index when documents are removed
        self._rebuild_bm25_index()
        return removed_count

    def search(self, query, top_k=3):
        """Searches the database for the most similar documents to the given query using vectorized operations.

        Args:
            query (str): The search query text.
            top_k (int, optional): The number of top results to return. Defaults to 3.

        Returns:
            list[dict]: A list of dictionaries containing the score, text, and metadata for each result.
        """
        if self.vectors is None or len(self.documents) == 0:
            return []

        # 1. Encode query and normalize it to match the stored vectors
        query_vector = self.model.encode(query)
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm

        # 2. Vectorized calculation
        similarities = np.dot(self.vectors, query_vector)

        # 3. Get top-k indices efficiently
        top_k_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_k_indices:
            results.append({
                "score": float(similarities[idx]),
                "text": self.documents[idx],
                "metadata": self.metadata[idx],
                "index": int(idx),
                "type": "vector"
            })

        return results

    def search_bm25(self, query, top_k=3):
        """Searches the database using the BM25 algorithm.

        Args:
            query (str): The search query text.
            top_k (int, optional): The number of top results to return. Defaults to 3.

        Returns:
            list[dict]: A list of results with scores, text, and metadata.
        """
        if not self.documents:
            return []
            
        query_tokens = self._tokenize(query)
        scores = np.zeros(len(self.documents))
        
        k1 = 1.5
        b = 0.75
        avgdl = self.bm25_data["avgdl"]
        
        for token in query_tokens:
            if token not in self.bm25_data["idf"]:
                continue
            
            idf = self.bm25_data["idf"][token]
            for i in range(len(self.documents)):
                tf = self.bm25_data["tf"][i].get(token, 0)
                dl = self.bm25_data["doc_lengths"][i]
                
                # BM25 Formula
                score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))
                scores[i] += score
                
        # Normalize BM25 scores to 0-1 range for easier comparison (optional but helpful)
        if scores.max() > 0:
            scores = scores / scores.max()
            
        top_k_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_k_indices:
            if scores[idx] <= 0: continue
            results.append({
                "score": float(scores[idx]),
                "text": self.documents[idx],
                "metadata": self.metadata[idx],
                "index": int(idx),
                "type": "bm25"
            })
        return results

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
        # Use a larger internal k to ensure good fusion
        internal_k = max(top_k * 2, 20)
        vector_results = self.search(query, top_k=internal_k)
        bm25_results = self.search_bm25(query, top_k=internal_k)
        
        # 2. Reciprocal Rank Fusion (RRF)
        # Score = sum( 1 / (k + rank) )
        rrf_scores = Counter()
        
        for rank, res in enumerate(vector_results, 1):
            rrf_scores[res["index"]] += 1.0 / (k_factor + rank)
            
        for rank, res in enumerate(bm25_results, 1):
            rrf_scores[res["index"]] += 1.0 / (k_factor + rank)
            
        # 3. Combine and sort
        combined_results = []
        
        # Merge vector and bm25 results to get the full objects
        all_candidates = vector_results + bm25_results
        id_to_obj = {res["index"]: res for res in all_candidates}
        
        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        for idx, rrf_score in sorted_indices[:top_k]:
            res = id_to_obj[idx]
            # Add hybrid-specific metadata
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
            # print(f"LOGE: [VectorDB] Loading reranker model: {self.rerank_model_name}...")
            self.reranker = CrossEncoder(self.rerank_model_name)
            
        # Prepare pairs for cross-encoder: (query, document_text)
        pairs = [[query, res["text"]] for res in results]
        
        # Predict scores (Cross-Encoder gives raw relevance scores)
        rerank_scores = self.reranker.predict(pairs)
        
        # Update scores in the result dictionaries
        for i, res in enumerate(results):
            res["rerank_score"] = float(rerank_scores[i])
            
        # Sort by the new rerank_score
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return results

    def save(self, filepath):
        """Saves the database state to a JSON file and vectors to a binary .npy file.

        Args:
            filepath (str): The destination file path for the JSON metadata.
        """
        # Save vectors separately as binary to maximize performance and minimize disk usage
        vector_path = filepath.rsplit('.', 1)[0] + ".vectors.npy"
        
        data = {
            "model_name": self.model_name,
            "documents": self.documents,
            "metadata": self.metadata,
            "file_hashes": self.file_hashes,
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # 1. Save metadata as JSON
        with open(filepath, "w", encoding="utf-8") as f:
            # Don't save large TF/IDF data in JSON, it can be rebuilt or saved separately
            # Actually, for simplicity, let's not save it and just rebuild on load
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        # 2. Save vectors as Binary
        if self.vectors is not None:
            np.save(vector_path, self.vectors)
            
        print(f"LOGE: [VectorDB] Saved DB to {filepath} (Vectors: {vector_path})")

    def load(self, filepath):
        """Loads the database state from JSON and binary files.

        Args:
            filepath (str): The source JSON file path.
        """
        if not os.path.exists(filepath):
            print("LOGE: [VectorDB] No existing DB found. Starting fresh.")
            return

        # 1. Load metadata
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        saved_model = data.get("model_name", self.model_name)
        if saved_model and saved_model != self.model_name:
            import sys
            print(f"LOGE: [VectorDB] ERROR: Model mismatch! DB uses '{saved_model}' but you requested '{self.model_name}'.")
            print(f"LOGE: [VectorDB] Hint: Delete the DB file ({filepath}) or specify a different DB path.")
            sys.exit(1)

        self.documents = data.get("documents", [])
        self.metadata = data.get("metadata", [])
        self.file_hashes = data.get("file_hashes", {})
        
        # 2. Load vectors from binary file
        vector_path = filepath.rsplit('.', 1)[0] + ".vectors.npy"
        if os.path.exists(vector_path):
            self.vectors = np.load(vector_path)
            # Compatibility check: ensure vectors match document count
            if len(self.vectors) != len(self.documents):
                print(f"LOGE: [VectorDB] WARNING: Vector count ({len(self.vectors)}) mismatch with documents ({len(self.documents)})!")
        else:
            # Fallback for old JSON-based DBs
            vectors = data.get("vectors", [])
            if vectors:
                print("LOGE: [VectorDB] Migrating old JSON vectors to binary format...")
                self.vectors = np.array(vectors)
            else:
                self.vectors = None
                
        # Rebuild BM25 index after loading
        self._rebuild_bm25_index()
                
        # print(f"LOGE: [VectorDB] Loaded DB from {filepath} ({len(self.documents)} documents)")
