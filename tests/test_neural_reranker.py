"""
Unit tests for NeuralReranker module.
"""

from src.retrieval.reranker import NeuralReranker


def test_neural_reranker_fallback_scoring():
    # Test fallback without loading heavy PyTorch weights
    reranker = NeuralReranker(lazy_load=True)
    candidates = [
        {"id": 1, "text": "Unrelated passage about cooking pasta recipes.", "score": 0.5},
        {"id": 2, "text": "Detailed explanation of Hibernate ORM mapping architecture.", "score": 0.6}
    ]

    top_chunks, latency = reranker.rerank(
        query="What is Hibernate ORM mapping?",
        candidates=candidates,
        top_k=2
    )

    assert len(top_chunks) == 2
    assert latency >= 0.0
    assert "rerank_score" in top_chunks[0]
    # Chunk 2 has high overlap with query
    assert top_chunks[0]["id"] == 2


def test_neural_reranker_empty():
    reranker = NeuralReranker(lazy_load=True)
    results, lat = reranker.rerank("Query", [], top_k=3)
    assert results == []
    assert lat == 0.0


def test_neural_reranker_single_candidate():
    reranker = NeuralReranker(lazy_load=True)
    candidates = [{"id": 1, "text": "Only one chunk present."}]
    results, _ = reranker.rerank("query", candidates, top_k=1)
    assert len(results) == 1
    assert results[0]["rerank_score"] == 1.0
