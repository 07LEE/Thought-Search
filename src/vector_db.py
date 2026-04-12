import json
import os
import warnings
import logging
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# Hide internal warning messages
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


from config import EMBEDDING_MODEL, RERANK_MODEL

class SimpleVectorDB:
    def __init__(self, model_name=None, rerank_model_name=None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.rerank_model_name = rerank_model_name or RERANK_MODEL
        print(f"LOGE: [VectorDB] Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        
        # Lazy loading for reranker to save memory if not used
        self.reranker = None

        self.documents = []
        self.metadata = []
        self.vectors = None
        self.file_hashes = {}

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

        # 2. Vectorized calculation!
        # Since both query and stored vectors are unit vectors (norm=1),
        # Cosine Similarity is simply the Dot Product.
        # np.dot(matrix, vector) calculates similarity with ALL documents at once in C/ASM.
        similarities = np.dot(self.vectors, query_vector)

        # 3. Get top-k indices efficiently
        top_k_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_k_indices:
            results.append(
                {
                    "score": float(similarities[idx]),
                    "text": self.documents[idx],
                    "metadata": self.metadata[idx],
                }
            )

        return results

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
            print(f"LOGE: [VectorDB] Loading reranker model: {self.rerank_model_name}...")
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
                
        print(f"LOGE: [VectorDB] Loaded DB from {filepath} ({len(self.documents)} documents)")
