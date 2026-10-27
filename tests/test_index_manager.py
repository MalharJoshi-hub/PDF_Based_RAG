"""
Unit tests for IndexManager module.
"""

from src.vector_store.index_manager import IndexManager
from src.embeddings.embedder import Embedder


def test_index_manager_build_and_search(mock_embedding_provider, sample_chunks, tmp_path):
    embedder = Embedder(provider=mock_embedding_provider)
    idx_file = tmp_path / "faiss.index"
    map_file = tmp_path / "id_map.pkl"

    mgr = IndexManager(
        embedder=embedder,
        index_file=idx_file,
        id_map_file=map_file
    )

    count, latency = mgr.build_from_chunks(sample_chunks)
    assert count == 3
    assert mgr.total_indexed == 3
    assert idx_file.exists()
    assert map_file.exists()

    hits, search_lat = mgr.search("Hibernate SessionFactory", top_k=2)
    assert len(hits) == 2
    assert search_lat >= 0.0


def test_index_manager_empty_build(mock_embedding_provider, tmp_path):
    embedder = Embedder(provider=mock_embedding_provider)
    mgr = IndexManager(embedder=embedder)
    count, _ = mgr.build_from_chunks([])
    assert count == 0
    assert mgr.total_indexed == 0
