"""
Evaluation package providing metrics, benchmark datasets, and retrieval evaluators.
"""

from src.evaluation.metrics import RetrievalMetrics
from src.evaluation.dataset import BenchmarkQuery, DEFAULT_BENCHMARK_SUITE
from src.evaluation.benchmarker import Benchmarker

__all__ = [
    "RetrievalMetrics",
    "BenchmarkQuery",
    "DEFAULT_BENCHMARK_SUITE",
    "Benchmarker"
]
