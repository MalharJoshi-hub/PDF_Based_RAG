"""
Index manager coordinating vector building, disk synchronization, and incremental updates.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from src.vector_store.base import BaseVectorStore, VectorSearchResult
from src.vector_store.faiss_store import FAISSVectorStore
from src.embeddings.embedder import Embedder


class IndexManager:
    """Orchestrates indexing lifecycle, batch embedding, and vector persistence."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: Optional[BaseVectorStore] = None,
        index_file: Optional[Path] = None,
        id_map_file: Optional[Path] = None
    ):
        self.embedder = embedder
        self.store = vector_store or FAISSVectorStore(dimension=embedder.dimension)
        self.index_file = index_file
        self.id_map_file = id_map_file

    @property
    def total_indexed(self) -> int:
        return self.store.count

    def build_from_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[int, float]:
        """
        Builds or rebuilds index from database chunks list.

        Args:
            chunks: List of dictionaries containing 'id' and 'text'.

        Returns:
            Tuple of (count: int, latency_ms: float)
        """
        self.store.clear()
        if not chunks:
            if self.index_file and self.id_map_file:
                self.save_to_disk()
            return 0, 0.0

        texts = [c["text"] for c in chunks]
        ids = [c["id"] for c in chunks]

        vectors, latency_ms = self.embedder.embed_documents(texts)
        self.store.add(vectors, ids)

        if self.index_file and self.id_map_file:
            self.save_to_disk()

        return len(ids), latency_ms

    def search(self, query: str, top_k: int = 5) -> Tuple[List[VectorSearchResult], float]:
        """
        Encodes query and retrieves nearest neighbor chunk matches.

        Returns:
            Tuple of (results: List[VectorSearchResult], latency_ms: float)
        """
        if self.store.count == 0:
            return [], 0.0

        query_vec, embed_latency = self.embedder.embed_query(query)
        results = self.store.search(query_vec, top_k=top_k)
        return results, embed_latency

    def save_to_disk(self) -> None:
        """Flushes in-memory vector index to filesystem."""
        if self.index_file and self.id_map_file:
            self.store.save(str(self.index_file), str(self.id_map_file))

    def load_from_disk(self) -> bool:
        """Loads index from disk if files exist."""
        if self.index_file and self.id_map_file:
            return self.store.load(str(self.index_file), str(self.id_map_file))
        return False
