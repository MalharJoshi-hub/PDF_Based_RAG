"""
Unit tests for RetrieverManager module.
"""

from src.retrieval.retriever_manager import RetrieverManager
from src.embeddings.embedder import Embedder
from src.vector_store.index_manager import IndexManager


def test_retriever_manager_lifecycle(mock_embedding_provider, sample_chunks):
    embedder = Embedder(provider=mock_embedding_provider)
    idx_mgr = IndexManager(embedder=embedder)
    mgr = RetrieverManager(embedder=embedder, index_manager=idx_mgr)

    # Rebuild all
    metrics = mgr.rebuild_all(sample_chunks)
    assert metrics["total_chunks"] == 3
    assert metrics["faiss_count"] == 3

    lookup = {c["id"]: c for c in sample_chunks}

    # Search dense
    dense_res = mgr.search("Hibernate", method="dense", top_k=2, chunks_lookup=lookup)
    assert dense_res["method"] == "dense"
    assert len(dense_res["results"]) <= 2

    # Search bm25
    bm25_res = mgr.search("SessionFactory", method="bm25", top_k=2, chunks_lookup=lookup)
    assert bm25_res["method"] == "bm25"
    assert len(bm25_res["results"]) >= 1

    # Search tfidf
    tfidf_res = mgr.search("SQL database", method="tfidf", top_k=2, chunks_lookup=lookup)
    assert tfidf_res["method"] == "tfidf"

    # Search hybrid
    hybrid_res = mgr.search("Hibernate Session", method="hybrid", top_k=2, chunks_lookup=lookup)
    assert hybrid_res["method"] == "hybrid"

    # Compare
    comp = mgr.compare_all("Hibernate ORM", top_k=2, chunks_lookup=lookup)
    assert "dense" in comp
    assert "bm25" in comp
    assert "tfidf" in comp
    assert "overlap_stats" in comp
