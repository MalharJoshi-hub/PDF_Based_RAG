"""
Pytest shared fixtures for in-memory databases, sample documents, and mock retrievers.
"""

import pytest
import sqlite3
import numpy as np
from pathlib import Path
from src.persistence.database import DatabaseManager
from src.persistence.migrations import MigrationManager
from src.persistence.repository import DocumentRepository, ChunkRepository
from src.embeddings.base import BaseEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic mock embedding provider for fast test execution."""

    def __init__(self, dimension: int = 16):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_query(self, query: str) -> np.ndarray:
        # Deterministic vector based on query length and hash
        vec = np.ones(self._dim, dtype=np.float32) * (len(query) % 10 + 1)
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32)

    def embed_documents(self, texts):
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        arr = np.array([self.embed_query(t) for t in texts], dtype=np.float32)
        return arr


@pytest.fixture
def mock_embedding_provider():
    return MockEmbeddingProvider(dimension=16)


@pytest.fixture
def temp_db(tmp_path):
    """Provides an isolated, migrated SQLite database in a temporary directory."""
    db_file = tmp_path / "test_pdf_rag.db"
    db_mgr = DatabaseManager(db_path=db_file)
    migration_mgr = MigrationManager(db_mgr)
    migration_mgr.initialize_schema()
    return db_mgr


@pytest.fixture
def sample_pdf_bytes():
    """Provides valid dummy PDF file bytes with proper header."""
    return b"%PDF-1.4\n1 0 obj\n<< /Title (Test Document) >>\nendobj\ntrailer\n<< >>\n%%EOF\n"


@pytest.fixture
def sample_chunks():
    """Sample chunk dictionaries for testing retrieval and persistence."""
    return [
        {
            "id": 1,
            "chunk_index": 0,
            "page_number": 1,
            "text": "Hibernate is an Object-Relational Mapping (ORM) solution for Java environments.",
            "token_count": 12,
            "filename": "hibernate_guide.pdf",
            "pdf_id": 1
        },
        {
            "id": 2,
            "chunk_index": 1,
            "page_number": 1,
            "text": "The SessionFactory is a thread-safe factory for Session objects in Hibernate.",
            "token_count": 13,
            "filename": "hibernate_guide.pdf",
            "pdf_id": 1
        },
        {
            "id": 3,
            "chunk_index": 2,
            "page_number": 2,
            "text": "A Session represents a physical connection with the underlying SQL database.",
            "token_count": 12,
            "filename": "hibernate_guide.pdf",
            "pdf_id": 1
        }
    ]
