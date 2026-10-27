"""
Embeddings package for dense vector representations, caching, and sentence transformers.
"""

from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings.sentence_transformer_provider import SentenceTransformerProvider
from src.embeddings.cache import EmbeddingCache
from src.embeddings.embedder import Embedder

__all__ = [
    "BaseEmbeddingProvider",
    "SentenceTransformerProvider",
    "EmbeddingCache",
    "Embedder"
]
