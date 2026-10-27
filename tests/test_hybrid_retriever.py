"""
Unit tests for HybridRetriever and Reciprocal Rank Fusion (RRF).
"""

from src.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_rrf_fusion():
    hybrid = HybridRetriever(rrf_k=60)
    ranked_lists = {
        "dense": [(10, 0.95), (20, 0.85), (30, 0.75)],
        "bm25": [(20, 4.2), (10, 3.8), (40, 2.1)]
    }

    fused = hybrid.fuse(ranked_lists, top_k=3)
    assert len(fused) == 3
    # Document 10 and 20 appear in both, so should have highest RRF scores
    fused_ids = [doc_id for doc_id, score in fused]
    assert 10 in fused_ids
    assert 20 in fused_ids
    assert fused[0][1] > fused[2][1]


def test_hybrid_empty():
    hybrid = HybridRetriever()
    fused = hybrid.fuse({}, top_k=5)
    assert fused == []
