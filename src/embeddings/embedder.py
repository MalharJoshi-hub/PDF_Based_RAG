"""
High-level Embedder coordinator providing caching, batching, and vector formatting.
"""

import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings.sentence_transformer_provider import SentenceTransformerProvider
from src.embeddings.cache import EmbeddingCache


class Embedder:
    """Coordinates embedding generation with optional LRU caching and performance profiling."""

    def __init__(
        self,
        provider: Optional[BaseEmbeddingProvider] = None,
        cache: Optional[EmbeddingCache] = None,
        enable_cache: bool = True
    ):
        self.provider = provider or SentenceTransformerProvider()
        self.cache = cache or (EmbeddingCache() if enable_cache else None)

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    def embed_query(self, query: str) -> Tuple[np.ndarray, float]:
        """
        Embeds a single query, checking cache first.

        Returns:
            Tuple of (vector: np.ndarray, latency_ms: float)
        """
        start_time = time.perf_counter()

        if self.cache is not None:
            cached_vec = self.cache.get(query)
            if cached_vec is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                return cached_vec, latency_ms

        vec = self.provider.embed_query(query)

        if self.cache is not None:
            self.cache.put(query, vec)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return vec, latency_ms

    def embed_documents(self, texts: List[str]) -> Tuple[np.ndarray, float]:
        """
        Embeds a collection of document texts, leveraging cache for known texts.

        Returns:
            Tuple of (vectors: np.ndarray, latency_ms: float)
        """
        start_time = time.perf_counter()

        if not texts:
            empty_arr = np.empty((0, self.dimension), dtype=np.float32)
            return empty_arr, 0.0

        if self.cache is None:
            vecs = self.provider.embed_documents(texts)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return vecs, latency_ms

        results: List[Optional[np.ndarray]] = [None] * len(texts)
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        for idx, text in enumerate(texts):
            cached_vec = self.cache.get(text)
            if cached_vec is not None:
                results[idx] = cached_vec
            else:
                missing_indices.append(idx)
                missing_texts.append(text)

        if missing_texts:
            computed_vecs = self.provider.embed_documents(missing_texts)
            for idx, text, vec in zip(missing_indices, missing_texts, computed_vecs):
                results[idx] = vec
                self.cache.put(text, vec)

        stacked = np.vstack(results).astype(np.float32)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return stacked, latency_ms

    def get_stats(self) -> Dict[str, Any]:
        """Returns embedding cache telemetry."""
        return {
            "dimension": self.dimension,
            "cache": self.cache.stats if self.cache else None
        }
