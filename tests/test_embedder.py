"""
Unit tests for Embedder coordinator module.
"""

import numpy as np
from src.embeddings.embedder import Embedder
from src.embeddings.cache import EmbeddingCache


def test_embedder_query_with_mock(mock_embedding_provider):
    embedder = Embedder(provider=mock_embedding_provider, enable_cache=True)
    vec, latency = embedder.embed_query("Sample query")

    assert vec.shape == (mock_embedding_provider.dimension,)
    assert latency >= 0.0

    # Second call should hit cache
    vec2, latency2 = embedder.embed_query("Sample query")
    assert np.allclose(vec, vec2)
    assert embedder.cache.stats["hits"] == 1


def test_embedder_documents_batch(mock_embedding_provider):
    embedder = Embedder(provider=mock_embedding_provider, enable_cache=True)
    texts = ["Text one", "Text two", "Text three"]
    vecs, latency = embedder.embed_documents(texts)

    assert vecs.shape == (3, mock_embedding_provider.dimension)
    assert latency >= 0.0

    # Test empty documents
    empty_vecs, empty_lat = embedder.embed_documents([])
    assert empty_vecs.shape == (0, mock_embedding_provider.dimension)


def test_embedder_stats(mock_embedding_provider):
    embedder = Embedder(provider=mock_embedding_provider, enable_cache=True)
    stats = embedder.get_stats()
    assert stats["dimension"] == mock_embedding_provider.dimension
    assert stats["cache"] is not None
