import re
import math
import numpy as np
from collections import Counter

class SparseIndex:
    """Handles BM25-based sparse search logic."""
    
    def __init__(self):
        self.tf = []          # Term frequencies per document
        self.idf = {}         # Inverse document frequency
        self.avgdl = 0        # Average document length
        self.doc_lengths = []  # Length of each document
        self.k1 = 1.5
        self.b = 0.75

    def _tokenize(self, text):
        """Simple tokenizer for BM25.

        Args:
            text (str): The raw text to tokenize.

        Returns:
            list[str]: A list of alphanumeric tokens in lowercase.
        """
        return re.findall(r'\w+', text.lower())

    def rebuild(self, documents):
        """Rebuilds the BM25 index from the provided documents.

        Args:
            documents (list[str]): List of document texts.
        """
        if not documents:
            self.tf = []
            self.idf = {}
            self.avgdl = 0
            self.doc_lengths = []
            return

        doc_tokens = [self._tokenize(doc) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in doc_tokens]
        self.avgdl = sum(self.doc_lengths) / len(documents)
        
        # Calculate TF
        self.tf = [Counter(tokens) for tokens in doc_tokens]
        
        # Calculate IDF
        num_docs = len(documents)
        word_doc_counts = Counter()
        for tokens in doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                word_doc_counts[token] += 1
        
        self.idf = {}
        for word, count in word_doc_counts.items():
            self.idf[word] = math.log(1 + (num_docs - count + 0.5) / (count + 0.5))

    def search(self, query, documents, metadata, top_k=3):
        """Searches the index using the BM25 algorithm.

        Args:
            query (str): The search query text.
            documents (list[str]): The document pool to search in.
            metadata (list[dict]): Metadata for each document.
            top_k (int, optional): Number of results to return. Defaults to 3.

        Returns:
            list[dict]: Ranked search results.
        """
        if not documents:
            return []
            
        query_tokens = self._tokenize(query)
        scores = np.zeros(len(documents))
        
        for token in query_tokens:
            if token not in self.idf:
                continue
            
            idf = self.idf[token]
            for i in range(len(documents)):
                tf = self.tf[i].get(token, 0)
                dl = self.doc_lengths[i]
                
                # BM25 Formula
                score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl)))
                scores[i] += score
                
        if scores.max() > 0:
            scores = scores / scores.max()
            
        top_k_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_k_indices:
            if scores[idx] <= 0: continue
            results.append({
                "score": float(scores[idx]),
                "text": documents[idx],
                "metadata": metadata[idx],
                "index": int(idx),
                "type": "bm25"
            })
        return results

    def get_state(self):
        """Returns the internal state for persistence."""
        return {
            "tf": self.tf,
            "idf": self.idf,
            "avgdl": self.avgdl,
            "doc_lengths": self.doc_lengths
        }

    def set_state(self, state):
        """Restores the internal state."""
        if not state: return
        self.tf = state.get("tf", [])
        self.idf = state.get("idf", {})
        self.avgdl = state.get("avgdl", 0)
        self.doc_lengths = state.get("doc_lengths", [])
