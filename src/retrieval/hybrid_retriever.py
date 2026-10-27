"""
Hybrid search retriever combining Dense and Sparse results via Reciprocal Rank Fusion (RRF).
"""

from typing import List, Tuple, Dict, Any
from collections import defaultdict


class HybridRetriever:
    """
    Combines ranked candidate lists from multiple retrievers using Reciprocal Rank Fusion:
    RRF(d) = sum_{m in models} 1 / (k + rank_m(d))
    """

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def fuse(
        self,
        ranked_lists: Dict[str, List[Tuple[int, float]]],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Merges dictionary of named ranked lists: {'dense': [...], 'bm25': [...]}
        into a unified ranked list based on RRF scores.

        Returns:
            List of (chunk_id: int, rrf_score: float) ordered descending.
        """
        rrf_scores: Dict[int, float] = defaultdict(float)

        for method, candidate_list in ranked_lists.items():
            for rank, (chunk_id, _) in enumerate(candidate_list, start=1):
                rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank)

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_items
