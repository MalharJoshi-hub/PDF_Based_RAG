"""
Unit tests for API request validation schemas.
"""

import pytest
from src.api.schemas import QueryRequestSchema, CompareRequestSchema


def test_query_request_schema_valid():
    data = {
        "query": "What is Hibernate SessionFactory?",
        "retriever_method": "bm25",
        "top_k_retrieved": 5,
        "top_k_rerank": 3,
        "use_reranker": False
    }
    schema = QueryRequestSchema.from_dict(data)
    assert schema.query == "What is Hibernate SessionFactory?"
    assert schema.retriever_method == "bm25"
    assert schema.top_k_retrieved == 5
    assert schema.top_k_rerank == 3
    assert schema.use_reranker is False


def test_query_request_schema_empty_query():
    with pytest.raises(ValueError):
        QueryRequestSchema.from_dict({"query": "   "})


def test_query_request_schema_invalid_method():
    with pytest.raises(ValueError):
        QueryRequestSchema.from_dict({"query": "Valid question", "retriever_method": "unsupported"})


def test_compare_request_schema_valid():
    data = {"query": "Explain ORM", "top_k": 4}
    schema = CompareRequestSchema.from_dict(data)
    assert schema.query == "Explain ORM"
    assert schema.top_k == 4


def test_compare_request_schema_empty():
    with pytest.raises(ValueError):
        CompareRequestSchema.from_dict({})
