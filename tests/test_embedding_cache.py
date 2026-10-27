"""
Unit tests for EmbeddingCache module.
"""

import numpy as np
from src.embeddings.cache import EmbeddingCache


def test_embedding_cache_put_get():
    cache = EmbeddingCache(max_size=10)
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    cache.put("test query", vec)
    cached = cache.get("test query")

    assert cached is not None
    assert np.allclose(cached, vec)
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 0


def test_embedding_cache_miss():
    cache = EmbeddingCache(max_size=10)
    cached = cache.get("non-existent text")
    assert cached is None
    assert cache.stats["misses"] == 1


def test_embedding_cache_lru_eviction():
    cache = EmbeddingCache(max_size=2)
    vec1 = np.array([1.0], dtype=np.float32)
    vec2 = np.array([2.0], dtype=np.float32)
    vec3 = np.array([3.0], dtype=np.float32)

    cache.put("t1", vec1)
    cache.put("t2", vec2)
    assert cache.stats["size"] == 2

    # Put third item, t1 should be evicted
    cache.put("t3", vec3)
    assert cache.get("t1") is None
    assert cache.get("t2") is not None
    assert cache.get("t3") is not None


def test_embedding_cache_clear():
    cache = EmbeddingCache(max_size=5)
    cache.put("t1", np.array([1.0], dtype=np.float32))
    assert cache.stats["size"] == 1

    cache.clear()
    assert cache.stats["size"] == 0
    assert cache.stats["hits"] == 0
    assert cache.stats["misses"] == 0
