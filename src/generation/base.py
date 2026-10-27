"""
Base interfaces and data structures for generative synthesis and response formatting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Citation:
    """Represents a cited context source in the generated answer."""
    citation_id: int
    pdf_id: int
    filename: str
    page_number: int
    chunk_index: int
    snippet: str
    score: float
    rerank_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "citation_id": self.citation_id,
            "pdf_id": self.pdf_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "snippet": self.snippet,
            "score": round(float(self.score), 4)
        }
        if self.rerank_score is not None:
            data["rerank_score"] = round(float(self.rerank_score), 4)
        return data


@dataclass
class GenerationResponse:
    """Comprehensive RAG pipeline response payload."""
    query: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    retriever_method: str = "dense"
    timings: Dict[str, float] = field(default_factory=dict)
    model: str = "google/flan-t5-base"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "retriever_method": self.retriever_method,
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
            "model": self.model
        }


class BaseGenerator(ABC):
    """Abstract interface for generative models."""

    @abstractmethod
    def generate_answer(self, prompt: str) -> str:
        """Generates textual answer from prompt context."""
        pass
