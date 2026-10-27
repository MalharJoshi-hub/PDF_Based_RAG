"""
Abstract interfaces and data models for vector indexing and similarity search.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class VectorSearchResult:
    """Represents a single nearest neighbor hit from the vector index."""
    id: int
    score: float
    rank: int
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": round(float(self.score), 4),
            "rank": self.rank
        }


class BaseVectorStore(ABC):
    """Abstract interface for dense vector indices."""

    @property
    @abstractmethod
    def count(self) -> int:
        """Returns the total number of indexed vectors."""
        pass

    @abstractmethod
    def add(self, vectors: np.ndarray, ids: List[int]) -> None:
        """Adds dense vectors associated with external entity IDs."""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[VectorSearchResult]:
        """Performs nearest-neighbor search for a query vector."""
        pass

    @abstractmethod
    def save(self, index_path: str, id_map_path: str) -> None:
        """Serializes vector index and ID mapping to disk."""
        pass

    @abstractmethod
    def load(self, index_path: str, id_map_path: str) -> bool:
        """Loads serialized index and ID mapping from disk."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears all vectors and indices."""
        pass
