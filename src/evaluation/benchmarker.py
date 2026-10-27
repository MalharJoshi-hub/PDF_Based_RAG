"""
Automated benchmark runner evaluating Precision@K, Recall, and Latency across retrieval models.
"""

import time
from typing import List, Dict, Any, Optional
import numpy as np
from src.retrieval.retriever_manager import RetrieverManager
from src.evaluation.metrics import RetrievalMetrics
from src.evaluation.dataset import DEFAULT_BENCHMARK_SUITE, BenchmarkQuery


class Benchmarker:
    """Executes automated benchmark suites and compiles comparative evaluation metrics."""

    def __init__(self, retriever_manager: RetrieverManager):
        self.retriever_manager = retriever_manager

    def evaluate_methods(
        self,
        queries: Optional[List[BenchmarkQuery]] = None,
        chunks_lookup: Optional[Dict[int, Dict[str, Any]]] = None,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Runs benchmark queries across methods and computes precision and latency.

        Returns:
            Dictionary with aggregated metrics for each retriever method.
        """
        test_queries = queries or DEFAULT_BENCHMARK_SUITE
        eval_methods = methods or ["dense", "bm25", "tfidf"]

        results = {
            m: {
                "precision_at_1": [],
                "precision_at_3": [],
                "precision_at_5": [],
                "latencies_ms": []
            }
            for m in eval_methods
        }

        for q_obj in test_queries:
            query = q_obj.query
            keywords = q_obj.intent_keywords

            for m in eval_methods:
                start = time.perf_counter()
                res = self.retriever_manager.search(
                    query=query,
                    method=m,
                    top_k=5,
                    chunks_lookup=chunks_lookup
                )
                latency = (time.perf_counter() - start) * 1000.0
                results[m]["latencies_ms"].append(latency)

                hits = res.get("results", [])
                relevance_scores = [
                    RetrievalMetrics.keyword_relevance_score(h.get("text", ""), keywords)
                    for h in hits
                ]

                p1 = relevance_scores[0] if len(relevance_scores) >= 1 else 0.0
                p3 = float(np.mean(relevance_scores[:3])) if len(relevance_scores) >= 1 else 0.0
                p5 = float(np.mean(relevance_scores[:5])) if len(relevance_scores) >= 1 else 0.0

                results[m]["precision_at_1"].append(round(p1, 4))
                results[m]["precision_at_3"].append(round(p3, 4))
                results[m]["precision_at_5"].append(round(p5, 4))

        summary = {}
        for m, data in results.items():
            summary[m] = {
                "mean_p1": round(float(np.mean(data["precision_at_1"])), 4) if data["precision_at_1"] else 0.0,
                "mean_p3": round(float(np.mean(data["precision_at_3"])), 4) if data["precision_at_3"] else 0.0,
                "mean_p5": round(float(np.mean(data["precision_at_5"])), 4) if data["precision_at_5"] else 0.0,
                "mean_latency_ms": round(float(np.mean(data["latencies_ms"])), 2) if data["latencies_ms"] else 0.0
            }

        return summary
