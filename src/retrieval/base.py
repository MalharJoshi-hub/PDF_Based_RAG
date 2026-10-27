"""
Abstract interfaces and data structures for retrieval algorithms and candidates.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional


@dataclass
class RetrievalCandidate:
    """Represents a candidate chunk retrieved by a retrieval method."""
    id: int
    score: float
    method: str
    text: str = ""
    page_number: int = 1
    filename: str = ""
    rank: int = 1
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "score": round(float(self.score), 4),
            "method": self.method,
            "text": self.text,
            "page_number": self.page_number,
            "filename": self.filename,
            "rank": self.rank
        }
        if self.rerank_score is not None:
            data["rerank_score"] = round(float(self.rerank_score), 4)
        return data


class BaseRetriever(ABC):
    """Abstract interface for sparse or dense chunk retrievers."""

    @abstractmethod
    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """Indexes chunks for retrieval."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Executes search over indexed corpus.

        Returns:
            List of (chunk_id: int, score: float) tuples ordered descending by score.
        """
        pass
