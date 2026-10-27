"""
Unit tests for FAISSVectorStore module.
"""

import numpy as np
import pytest
from src.vector_store.faiss_store import FAISSVectorStore


def test_faiss_store_initialization():
    store = FAISSVectorStore(dimension=8)
    assert store.count == 0
    assert store.dimension == 8
    assert store.id_map == []


def test_faiss_store_add_and_search():
    store = FAISSVectorStore(dimension=4)
    # Unit-normalized vectors
    vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float32)
    ids = [101, 102, 103]

    store.add(vecs, ids)
    assert store.count == 3

    # Query closest to vec 0
    query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, top_k=2)

    assert len(results) == 2
    assert results[0].id == 101
    assert results[0].rank == 1
    assert results[0].score > results[1].score


def test_faiss_store_dimension_mismatch():
    store = FAISSVectorStore(dimension=4)
    vecs = np.ones((1, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        store.add(vecs, [1])


def test_faiss_store_save_load(tmp_path):
    store = FAISSVectorStore(dimension=4)
    vecs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    store.add(vecs, [55])

    idx_file = tmp_path / "test.index"
    map_file = tmp_path / "test.pkl"

    store.save(str(idx_file), str(map_file))
    assert idx_file.exists()
    assert map_file.exists()

    new_store = FAISSVectorStore(dimension=4)
    loaded = new_store.load(str(idx_file), str(map_file))
    assert loaded is True
    assert new_store.count == 1
    assert new_store.id_map == [55]


def test_faiss_store_clear():
    store = FAISSVectorStore(dimension=4)
    store.add(np.ones((2, 4), dtype=np.float32), [1, 2])
    assert store.count == 2

    store.clear()
    assert store.count == 0
    assert store.id_map == []
