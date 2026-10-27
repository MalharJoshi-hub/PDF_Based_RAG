"""
RetrieverManager coordinating Dense FAISS, BM25, TF-IDF, and Hybrid search engines.
"""

import time
from typing import List, Dict, Any, Optional
from src.embeddings.embedder import Embedder
from src.vector_store.index_manager import IndexManager
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.tfidf_retriever import TFIDFRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.config import config


class RetrieverManager:
    """
    Central orchestrator for Dense, BM25, TF-IDF, and Hybrid retrieval models.
    Supports individual querying as well as comparative multi-retriever benchmarking.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        index_manager: Optional[IndexManager] = None
    ):
        self.embedder = embedder or Embedder()
        self.index_manager = index_manager or IndexManager(
            embedder=self.embedder,
            index_file=config.faiss.index_file,
            id_map_file=config.faiss.id_map_file
        )
        self.bm25 = BM25Retriever()
        self.tfidf = TFIDFRetriever()
        self.hybrid = HybridRetriever()

    def rebuild_all(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Rebuilds FAISS, BM25, and TF-IDF indices simultaneously from chunk records.
        """
        start = time.perf_counter()

        faiss_count, faiss_latency = self.index_manager.build_from_chunks(chunks)

        bm25_start = time.perf_counter()
        self.bm25.build(chunks)
        bm25_latency = (time.perf_counter() - bm25_start) * 1000.0

        tfidf_start = time.perf_counter()
        self.tfidf.build(chunks)
        tfidf_latency = (time.perf_counter() - tfidf_start) * 1000.0

        total_latency = (time.perf_counter() - start) * 1000.0

        return {
            "total_chunks": len(chunks),
            "faiss_count": faiss_count,
            "timings_ms": {
                "faiss": round(faiss_latency, 2),
                "bm25": round(bm25_latency, 2),
                "tfidf": round(tfidf_latency, 2),
                "total": round(total_latency, 2)
            }
        }

    def search(
        self,
        query: str,
        method: str = "dense",
        top_k: int = 5,
        chunks_lookup: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Searches the corpus using the requested retrieval strategy.

        Args:
            query: Question or keywords.
            method: 'dense', 'bm25', 'tfidf', or 'hybrid'.
            top_k: Number of hits to return.
            chunks_lookup: Optional lookup dictionary from Chunk ID to chunk metadata.

        Returns:
            Dictionary containing method, results list, and latency.
        """
        start_time = time.perf_counter()
        method_lower = method.lower()
        raw_hits = []

        if method_lower == "dense":
            faiss_results, _ = self.index_manager.search(query, top_k=top_k)
            raw_hits = [(r.id, r.score) for r in faiss_results]
        elif method_lower == "bm25":
            raw_hits = self.bm25.search(query, top_k=top_k)
        elif method_lower == "tfidf":
            raw_hits = self.tfidf.search(query, top_k=top_k)
        elif method_lower == "hybrid":
            dense_hits, _ = self.index_manager.search(query, top_k=top_k * 2)
            bm25_hits = self.bm25.search(query, top_k=top_k * 2)
            raw_hits = self.hybrid.fuse(
                {
                    "dense": [(r.id, r.score) for r in dense_hits],
                    "bm25": bm25_hits
                },
                top_k=top_k
            )
        else:
            raise ValueError(f"Unknown retrieval method: '{method}'. Choose from dense, bm25, tfidf, hybrid.")

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        results = []
        for rank, (chunk_id, score) in enumerate(raw_hits, start=1):
            chunk_data = (chunks_lookup or {}).get(chunk_id, {})
            results.append({
                "id": chunk_id,
                "score": round(float(score), 4),
                "rank": rank,
                "text": chunk_data.get("text", ""),
                "page_number": chunk_data.get("page_number", 1),
                "filename": chunk_data.get("filename", "")
            })

        return {
            "method": method_lower,
            "latency_ms": round(latency_ms, 2),
            "results": results
        }

    def compare(
        self,
        query: str,
        top_k: int = 5,
        chunks_lookup: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Executes query against dense, bm25, and tfidf concurrently and returns side-by-side metrics.
        """
        dense_res = self.search(query, method="dense", top_k=top_k, chunks_lookup=chunks_lookup)
        bm25_res = self.search(query, method="bm25", top_k=top_k, chunks_lookup=chunks_lookup)
        tfidf_res = self.search(query, method="tfidf", top_k=top_k, chunks_lookup=chunks_lookup)

        dense_ids = set(r["id"] for r in dense_res["results"])
        bm25_ids = set(r["id"] for r in bm25_res["results"])
        tfidf_ids = set(r["id"] for r in tfidf_res["results"])

        all_ids = dense_ids | bm25_ids | tfidf_ids
        common_all = dense_ids & bm25_ids & tfidf_ids
        dense_bm25_overlap = len(dense_ids & bm25_ids)

        return {
            "query": query,
            "dense": dense_res,
            "bm25": bm25_res,
            "tfidf": tfidf_res,
            "overlap_stats": {
                "total_unique_chunks": len(all_ids),
                "common_to_all_three": len(common_all),
                "dense_bm25_overlap": dense_bm25_overlap
            }
        }

    def compare_all(
        self,
        query: str,
        top_k: int = 5,
        chunks_lookup: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Runs all three retrievers for comparative evaluation."""
        return self.compare(query, top_k=top_k, chunks_lookup=chunks_lookup)

