"""
Abstract base class and interfaces for embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbeddingProvider(ABC):
    """Abstract interface for dense embedding generation."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimensionality of the generated vectors."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string into a 1D float32 normalized vector.

        Args:
            query: User search query or question.

        Returns:
            np.ndarray of shape (dimension,) and dtype float32.
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a batch of document texts into a 2D float32 normalized array.

        Args:
            texts: List of document text strings.

        Returns:
            np.ndarray of shape (len(texts), dimension) and dtype float32.
        """
        pass
