"""
Base interfaces and data structures for text chunking strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DocumentChunk:
    """Represents a discrete segmented text passage from a document."""
    chunk_index: int
    page_number: int
    text: str
    token_count: int
    id: Optional[int] = None
    pdf_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to standard dictionary format."""
        data = {
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "text": self.text,
            "token_count": self.token_count
        }
        if self.id is not None:
            data["id"] = self.id
        if self.pdf_id is not None:
            data["pdf_id"] = self.pdf_id
        if self.metadata:
            data["metadata"] = self.metadata
        return data


class BaseChunker(ABC):
    """Abstract base class for all text chunking implementations."""

    @abstractmethod
    def split_text(self, text: str, page_number: int = 1) -> List[DocumentChunk]:
        """
        Splits a text block into a list of DocumentChunks.

        Args:
            text: Raw input text.
            page_number: Source page number for attribution.

        Returns:
            List of DocumentChunk instances.
        """
        pass

    @abstractmethod
    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a list of extracted pages and produces chunk dictionaries.

        Args:
            pages: List of dictionaries with 'page_number' and 'text'.

        Returns:
            List of chunk dictionaries compatible with persistence layer.
        """
        pass
