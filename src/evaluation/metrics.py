"""
Evaluation metrics for information retrieval: Precision@K, Recall@K, MRR, and Latency profiling.
"""

from typing import List, Set, Union


class RetrievalMetrics:
    """Calculates standardized quantitative IR ranking metrics."""

    @staticmethod
    def precision_at_k(retrieved_ids: List[int], relevant_ids: Union[Set[int], List[int]], k: int) -> float:
        """Calculates Precision@K = |Retrieved[:k] ∩ Relevant| / k."""
        if k <= 0 or not retrieved_ids:
            return 0.0
        k_retrieved = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        hits = sum(1 for doc_id in k_retrieved if doc_id in relevant_set)
        return round(hits / k, 4)

    @staticmethod
    def recall_at_k(retrieved_ids: List[int], relevant_ids: Union[Set[int], List[int]], k: int) -> float:
        """Calculates Recall@K = |Retrieved[:k] ∩ Relevant| / |Relevant|."""
        relevant_set = set(relevant_ids)
        if not relevant_set or k <= 0 or not retrieved_ids:
            return 0.0
        k_retrieved = retrieved_ids[:k]
        hits = sum(1 for doc_id in k_retrieved if doc_id in relevant_set)
        return round(hits / len(relevant_set), 4)

    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[int], relevant_ids: Union[Set[int], List[int]]) -> float:
        """Calculates MRR: 1 / rank of first relevant retrieved document."""
        relevant_set = set(relevant_ids)
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_set:
                return round(1.0 / rank, 4)
        return 0.0

    @staticmethod
    def keyword_relevance_score(text: str, keywords: List[str]) -> float:
        """Computes keyword density relevance proxy for zero-annotation benchmarking."""
        if not text or not keywords:
            return 0.0
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        target = max(1, len(keywords) * 0.5)
        return min(1.0, round(matches / target, 4))
