"""
Data models representing persistent entities: Documents, Chunks, and Query logs.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class PDFDocumentRecord:
    """Represents a PDF document record stored in the database."""
    filename: str
    total_pages: int = 1
    total_chunks: int = 0
    id: Optional[int] = None
    created_at: Optional[str] = None
    active_chunks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "filename": self.filename,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "created_at": self.created_at,
            "active_chunks": self.active_chunks
        }
        return data


@dataclass
class ChunkRecord:
    """Represents an extracted chunk record stored in SQLite."""
    pdf_id: int
    chunk_index: int
    text: str
    page_number: int = 1
    token_count: int = 0
    id: Optional[int] = None
    filename: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pdf_id": self.pdf_id,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "text": self.text,
            "token_count": self.token_count,
            "filename": self.filename
        }


@dataclass
class QueryAuditRecord:
    """Represents an audit entry for user queries."""
    query: str
    retriever_method: str
    latency_ms: float
    top_k: int
    answer: str
    id: Optional[int] = None
    created_at: Optional[str] = None
