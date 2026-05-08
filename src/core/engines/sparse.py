import re
import math
import numpy as np
from collections import Counter

class SparseIndex:
    """Handles BM25-based sparse search logic for keyword-based retrieval.

    This class implements the BM25 (Best Matching 25) ranking function, 
    which is widely used for estimating the relevance of documents to a 
    given search query.

    Attributes:
        tf (list[Counter]): Term frequencies for each document in the index.
        idf (dict[str, float]): Inverse document frequency scores for each term.
        avgdl (float): Average length of all documents in the index.
        doc_lengths (list[int]): Length (word count) of each document.
        k1 (float): BM25 parameter for term frequency scaling.
        b (float): BM25 parameter for document length normalization.
    """
    
    def __init__(self):
        """Initializes the SparseIndex with default BM25 parameters."""
        self.tf = []          # Term frequencies per document
        self.idf = {}         # Inverse document frequency
        self.avgdl = 0        # Average document length
        self.doc_lengths = []  # Length of each document
        self.k1 = 1.5
        self.b = 0.75

    def _tokenize(self, text):
        """Simple alphanumeric tokenizer for BM25.

        Args:
            text (str): The raw text to tokenize.

        Returns:
            list[str]: A list of alphanumeric tokens in lowercase.
        """
        return re.findall(r'\w+', text.lower())

    def rebuild(self, documents):
        """Rebuilds the BM25 index from the provided document collection.

        Args:
            documents (list[str]): List of raw document texts.
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
        all_terms = set()
        for counter in self.tf:
            all_terms.update(counter.keys())
            
        self.idf = {}
        for term in all_terms:
            # Number of documents containing the term
            doc_freq = sum(1 for counter in self.tf if term in counter)
            # Standard BM25 IDF formula
            self.idf[term] = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def search(self, query, documents, metadata, top_k=3):
        """Calculates BM25 scores for the query and returns ranked results.

        Args:
            query (str): The search query text.
            documents (list[str]): The document pool to search in.
            metadata (list[dict]): Metadata associated with each document.
            top_k (int, optional): Number of results to return. Defaults to 3.

        Returns:
            list[dict]: Ranked results with scores and document info.
        """
        if not self.tf or not documents:
            return []

        query_tokens = self._tokenize(query)
        scores = np.zeros(len(self.tf))
        
        for term in query_tokens:
            if term not in self.idf:
                continue
            
            idf_val = self.idf[term]
            for i, counter in enumerate(self.tf):
                tf_val = counter[term]
                # BM25 scoring formula
                numerator = tf_val * (self.k1 + 1)
                denominator = tf_val + self.k1 * (1 - self.b + self.b * self.doc_lengths[i] / self.avgdl)
                scores[i] += idf_val * (numerator / denominator)

        # Handle cases where all scores might be 0
        if np.all(scores == 0):
            return []

        # Rank and filter
        top_k_indices = scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_k_indices:
            if scores[idx] <= 0: continue
            results.append({
                "score": float(scores[idx]),
                "text": documents[idx],
                "metadata": metadata[idx],
                "index": int(idx),
                "type": "keyword"
            })
        return results
