"""
Unit tests for BM25Retriever module.
"""

from src.retrieval.bm25_retriever import BM25Retriever


def test_bm25_tokenize():
    tokens = BM25Retriever.tokenize("Hibernate: An Object/Relational mapping tool!")
    assert "hibernate" in tokens
    assert "object" in tokens
    assert "relational" in tokens


def test_bm25_search_success(sample_chunks):
    retriever = BM25Retriever()
    retriever.build(sample_chunks)

    hits = retriever.search("SessionFactory", top_k=2)
    assert len(hits) >= 1
    # Chunk 2 explicitly discusses SessionFactory
    top_hit_id = hits[0][0]
    assert top_hit_id == 2


def test_bm25_empty_query(sample_chunks):
    retriever = BM25Retriever()
    retriever.build(sample_chunks)
    assert retriever.search("", top_k=5) == []


def test_bm25_unindexed():
    retriever = BM25Retriever()
    assert retriever.search("anything", top_k=5) == []
