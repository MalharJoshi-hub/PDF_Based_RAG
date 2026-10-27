"""
FAISS Vector Store implementation with IndexFlatIP (Cosine Similarity) and ID mapping.
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from src.vector_store.base import BaseVectorStore, VectorSearchResult


class FAISSVectorStore(BaseVectorStore):
    """
    Vector store backed by FAISS IndexFlatIP (Inner Product on unit-normalized vectors).
    Maintains a deterministic 0..N-1 to Chunk ID mapping table.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.id_map: List[int] = []
        self._init_empty_index()

    def _init_empty_index(self):
        """Initializes empty FAISS IndexFlatIP."""
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.dimension)
            self.id_map = []
        except ImportError:
            raise ImportError("faiss-cpu package is required for FAISSVectorStore.")

    @property
    def count(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def add(self, vectors: np.ndarray, ids: List[int]) -> None:
        """
        Adds vectors to FAISS index with corresponding external IDs.
        Vectors are validated for shape and float32 dtype.
        """
        if len(ids) == 0:
            return

        if vectors.shape[0] != len(ids):
            raise ValueError(
                f"Mismatch: {vectors.shape[0]} vectors provided for {len(ids)} IDs."
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}."
            )

        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.add(vecs)
        self.id_map.extend(ids)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[VectorSearchResult]:
        """
        Performs nearest-neighbor search for a query vector.

        Args:
            query_vector: 1D array of shape (dimension,) or 2D of shape (1, dimension).
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of VectorSearchResult ordered by descending cosine similarity.
        """
        if self.count == 0 or top_k <= 0:
            return []

        if query_vector.ndim == 1:
            q_vec = query_vector.reshape(1, -1)
        else:
            q_vec = query_vector

        q_vec = np.ascontiguousarray(q_vec, dtype=np.float32)
        effective_k = min(top_k, self.count)

        distances, indices = self.index.search(q_vec, effective_k)

        results: List[VectorSearchResult] = []
        for rank, (score, internal_idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if internal_idx == -1 or internal_idx >= len(self.id_map):
                continue
            entity_id = self.id_map[internal_idx]
            results.append(
                VectorSearchResult(
                    id=entity_id,
                    score=float(score),
                    rank=rank
                )
            )

        return results

    def save(self, index_path: str, id_map_path: str) -> None:
        """Saves FAISS binary index and id_map pickle to disk."""
        import faiss
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(id_map_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(id_map_path, "wb") as f:
            pickle.dump(self.id_map, f)

    def load(self, index_path: str, id_map_path: str) -> bool:
        """Loads FAISS index and id_map from disk if both files exist."""
        import faiss
        if not os.path.exists(index_path) or not os.path.exists(id_map_path):
            return False

        try:
            self.index = faiss.read_index(str(index_path))
            with open(id_map_path, "rb") as f:
                self.id_map = pickle.load(f)
            self.dimension = self.index.d
            return True
        except Exception:
            self._init_empty_index()
            return False

    def clear(self) -> None:
        """Resets the FAISS index and id_map."""
        self._init_empty_index()
