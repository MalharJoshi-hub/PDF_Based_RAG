"""
Unit tests for TFIDFRetriever module.
"""

from src.retrieval.tfidf_retriever import TFIDFRetriever


def test_tfidf_search_success(sample_chunks):
    retriever = TFIDFRetriever()
    retriever.build(sample_chunks)

    hits = retriever.search("underlying SQL database", top_k=2)
    assert len(hits) >= 1
    # Chunk 3 has database
    top_hit_id = hits[0][0]
    assert top_hit_id == 3


def test_tfidf_empty_query(sample_chunks):
    retriever = TFIDFRetriever()
    retriever.build(sample_chunks)
    assert retriever.search("", top_k=5) == []


def test_tfidf_unindexed():
    retriever = TFIDFRetriever()
    assert retriever.search("query", top_k=3) == []
