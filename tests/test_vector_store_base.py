"""
Unit tests for VectorSearchResult and BaseVectorStore interfaces.
"""

from src.vector_store.base import VectorSearchResult


def test_vector_search_result_to_dict():
    res = VectorSearchResult(id=42, score=0.87654, rank=1)
    d = res.to_dict()
    assert d["id"] == 42
    assert d["score"] == 0.8765
    assert d["rank"] == 1
