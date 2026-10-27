"""
Unit tests for RetrievalMetrics module.
"""

from src.evaluation.metrics import RetrievalMetrics


def test_precision_at_k():
    retrieved = [1, 2, 3, 4, 5]
    relevant = {2, 4, 6}

    p1 = RetrievalMetrics.precision_at_k(retrieved, relevant, k=1)
    assert p1 == 0.0  # 1 is not in relevant

    p2 = RetrievalMetrics.precision_at_k(retrieved, relevant, k=2)
    assert p2 == 0.5  # 1 hit (2) out of 2

    p4 = RetrievalMetrics.precision_at_k(retrieved, relevant, k=4)
    assert p4 == 0.5  # 2 hits (2, 4) out of 4


def test_recall_at_k():
    retrieved = [1, 2, 3, 4]
    relevant = {2, 4}  # total 2 relevant

    r2 = RetrievalMetrics.recall_at_k(retrieved, relevant, k=2)
    assert r2 == 0.5  # found 1 of 2

    r4 = RetrievalMetrics.recall_at_k(retrieved, relevant, k=4)
    assert r4 == 1.0  # found 2 of 2


def test_mean_reciprocal_rank():
    retrieved = [5, 9, 2, 8]
    relevant = {2, 10}
    # First relevant doc is at rank 3
    mrr = RetrievalMetrics.mean_reciprocal_rank(retrieved, relevant)
    assert round(mrr, 4) == round(1.0 / 3, 4)

    # When no hit
    mrr_none = RetrievalMetrics.mean_reciprocal_rank(retrieved, {99})
    assert mrr_none == 0.0


def test_keyword_relevance_score():
    text = "Hibernate is an ORM solution mapping tables to classes."
    keywords = ["hibernate", "orm", "mapping"]
    score = RetrievalMetrics.keyword_relevance_score(text, keywords)
    assert score > 0.5
