"""
Vector store module providing FAISS index abstractions and index management.
"""

from src.vector_store.base import BaseVectorStore, VectorSearchResult
from src.vector_store.faiss_store import FAISSVectorStore
from src.vector_store.index_manager import IndexManager

__all__ = [
    "BaseVectorStore",
    "VectorSearchResult",
    "FAISSVectorStore",
    "IndexManager"
]
