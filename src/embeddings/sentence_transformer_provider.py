"""
Sentence-Transformers embedding provider using all-MiniLM-L6-v2 with lazy loading and normalization.
"""

from typing import List, Optional
import numpy as np
from src.embeddings.base import BaseEmbeddingProvider


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """
    Embedding provider powered by Hugging Face sentence-transformers.
    Defaults to 'all-MiniLM-L6-v2' (384-dimensional dense vectors).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None
        self._dimension = 384

    def _ensure_model_loaded(self):
        """Lazy loads the SentenceTransformer model on first usage."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load SentenceTransformer model '{self.model_name}': {e}"
                )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds single query string into normalized 1D float32 array."""
        self._ensure_model_loaded()
        vec = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return vec.astype(np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embeds multiple document strings in batches."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._ensure_model_loaded()
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return vecs.astype(np.float32)
