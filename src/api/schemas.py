"""
API request and response schemas and validation helpers.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class QueryRequestSchema:
    """Validation schema for incoming /api/query requests."""
    query: str
    retriever_method: str = "dense"
    top_k_retrieved: int = 4
    top_k_rerank: int = 2
    use_reranker: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryRequestSchema":
        if not data or "query" not in data or not str(data["query"]).strip():
            raise ValueError("Query string is required and cannot be empty.")

        method = str(data.get("retriever_method", "dense")).lower()
        if method not in ["dense", "bm25", "tfidf", "hybrid"]:
            raise ValueError(f"Invalid retriever_method '{method}'. Must be dense, bm25, tfidf, or hybrid.")

        return cls(
            query=str(data["query"]).strip(),
            retriever_method=method,
            top_k_retrieved=int(data.get("top_k_retrieved", 4)),
            top_k_rerank=int(data.get("top_k_rerank", 2)),
            use_reranker=bool(data.get("use_reranker", True))
        )


@dataclass
class CompareRequestSchema:
    """Validation schema for incoming /api/compare requests."""
    query: str
    top_k: int = 5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompareRequestSchema":
        if not data or "query" not in data or not str(data["query"]).strip():
            raise ValueError("Query string is required and cannot be empty.")

        return cls(
            query=str(data["query"]).strip(),
            top_k=int(data.get("top_k", 5))
        )
