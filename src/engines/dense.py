import numpy as np
from sentence_transformers import SentenceTransformer

class DenseIndex:
    """Handles Vector-based dense search logic using sentence-transformers."""
    
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.vectors = None

    def embed(self, texts):
        """Calculates normalized embeddings for a list of texts.

        Args:
            texts (list[str]): List of texts to embed.

        Returns:
            np.ndarray: Normalized embedding vectors.
        """
        embeddings = self.model.encode(texts)
        # L2 Normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)

    def add_vectors(self, new_vectors):
        """Adds pre-calculated vectors to the index.

        Args:
            new_vectors (np.ndarray): Vectors to add.
        """
        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack((self.vectors, new_vectors))

    def search(self, query, documents, metadata, top_k=3):
        """Searches the index using cosine similarity.

        Args:
            query (str): The search query text.
            documents (list[str]): The document pool to search in.
            metadata (list[dict]): Metadata for each document.
            top_k (int, optional): Number of results to return. Defaults to 3.

        Returns:
            list[dict]: Ranked search results.
        """
        if self.vectors is None or len(documents) == 0:
            return []

        # Encode and normalize query
        query_vector = self.model.encode(query)
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm

        # Vectorized similarity calculation (Dot product on normalized vectors)
        similarities = np.dot(self.vectors, query_vector)
        top_k_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_k_indices:
            results.append({
                "score": float(similarities[idx]),
                "text": documents[idx],
                "metadata": metadata[idx],
                "index": int(idx),
                "type": "vector"
            })
        return results

    def set_vectors(self, vectors):
        """Sets the internal vectors (e.g., after loading from disk)."""
        self.vectors = vectors

    def get_vectors(self):
        """Returns the internal vectors for persistence."""
        return self.vectors
