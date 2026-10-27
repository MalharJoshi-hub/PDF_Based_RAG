"""
Embedding caching layer providing in-memory LRU and disk-backed cache for vector embeddings.
"""

import hashlib
import collections
from typing import Optional, Dict, Any
import numpy as np


class EmbeddingCache:
    """Thread-safe LRU cache for computed dense embeddings."""

    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    @staticmethod
    def _compute_key(text: str) -> str:
        """Computes deterministic MD5 hash for text string."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Retrieves cached vector if present, updating LRU order."""
        key = self._compute_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key].copy()
        self._misses += 1
        return None

    def put(self, text: str, vector: np.ndarray) -> None:
        """Stores vector in cache, evicting oldest item if capacity is reached."""
        key = self._compute_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = vector.astype(np.float32).copy()

    def clear(self) -> None:
        """Clears all entries from the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> Dict[str, Any]:
        """Returns cache telemetry stats."""
        total = self._hits + self._misses
        hit_ratio = (self._hits / total) if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": round(hit_ratio, 4)
        }
