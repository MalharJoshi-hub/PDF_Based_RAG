"""
Analytics package providing statistical text analysis and headless chart visualizers.
"""

from src.analytics.corpus_analyzer import CorpusAnalyzer, STOP_WORDS
from src.analytics.visualizer import AnalyticsVisualizer

__all__ = [
    "CorpusAnalyzer",
    "STOP_WORDS",
    "AnalyticsVisualizer"
]
