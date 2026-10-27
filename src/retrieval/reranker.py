"""
Neural Cross-Encoder reranker using cross-encoder/ms-marco-MiniLM-L-6-v2.
"""

from typing import List, Tuple, Dict, Any, Optional
import time


class NeuralReranker:
    """
    Reranks candidate chunks using a Transformer Cross-Encoder model.
    Passes (query, passage) pairs directly to score deep semantic relevance.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        lazy_load: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        if not lazy_load:
            self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        """Lazy loads CrossEncoder model on demand."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to load CrossEncoder model '{self.model_name}': {e}")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Reranks candidates for a given query string.

        Args:
            query: User's question or search query.
            candidates: List of chunk dictionaries containing 'text'.
            top_k: Number of highest-ranked chunks to return.

        Returns:
            Tuple of (reranked_chunks: List[Dict[str, Any]], latency_ms: float)
        """
        start_time = time.perf_counter()

        if not candidates or top_k <= 0:
            return [], 0.0

        if len(candidates) == 1:
            c = dict(candidates[0])
            c["rerank_score"] = 1.0
            latency = (time.perf_counter() - start_time) * 1000.0
            return [c], latency

        try:
            self._ensure_model_loaded()
            pairs = [[query, c.get("text", "")] for c in candidates]
            scores = self._model.predict(pairs)
        except Exception:
            # Fallback heuristic: token overlap scoring
            q_tokens = set(query.lower().split())
            scores = []
            for c in candidates:
                c_tokens = set(c.get("text", "").lower().split())
                overlap = len(q_tokens.intersection(c_tokens))
                scores.append(float(overlap))

        ranked = []
        for c, score in zip(candidates, scores):
            item = dict(c)
            item["rerank_score"] = round(float(score), 4)
            ranked.append(item)

        ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_results = ranked[:top_k]

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return top_results, latency_ms
