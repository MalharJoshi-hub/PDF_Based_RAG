"""
TF-IDF baseline retriever using scikit-learn with sublinear term frequency and n-grams.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.retrieval.base import BaseRetriever


class TFIDFRetriever(BaseRetriever):
    """
    TF-IDF retriever utilizing unigrams, bigrams, and cosine similarity.
    """

    def __init__(self, ngram_range: Tuple[int, int] = (1, 2), sublinear_tf: bool = True):
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.chunk_ids: List[int] = []

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """Fits vectorizer and transforms all chunk texts."""
        self.chunk_ids = [c["id"] for c in chunks]
        if not chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        texts = [c["text"] for c in chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            stop_words="english",
            min_df=1
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Transforms query and calculates cosine similarity with chunk matrix.
        """
        if self.vectorizer is None or self.tfidf_matrix is None or not self.chunk_ids:
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(sims)[::-1][:top_k]
        results: List[Tuple[int, float]] = []

        for idx in top_indices:
            score = float(sims[idx])
            if score > 0.0:
                results.append((self.chunk_ids[idx], score))

        return results
