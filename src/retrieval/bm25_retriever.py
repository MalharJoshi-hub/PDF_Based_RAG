"""
Sparse BM25 (Okapi) retriever using rank_bm25 and alphanumeric tokenization.
"""

import re
from typing import List, Tuple, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """
    BM25 Okapi retriever for keyword and lexical matching across document chunks.
    """

    TOKEN_REGEX = re.compile(r"\b\w+\b")

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[int] = []

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Lowercases and extracts alphanumeric words."""
        if not text:
            return []
        return cls.TOKEN_REGEX.findall(text.lower())

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """Constructs the BM25 inverted index from tokenized chunks."""
        self.chunk_ids = [c["id"] for c in chunks]
        if not chunks:
            self.bm25 = None
            return

        corpus = [self.tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Calculates BM25 relevance scores for the query.

        Returns:
            List of (chunk_id, score) tuples.
        """
        if self.bm25 is None or not self.chunk_ids:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[Tuple[int, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                results.append((self.chunk_ids[idx], score))

        return results
