"""
Unit tests for CorpusAnalyzer module.
"""

from src.analytics.corpus_analyzer import CorpusAnalyzer


def test_corpus_analyzer_metrics(sample_chunks):
    analyzer = CorpusAnalyzer()
    stats = analyzer.analyze_chunks(sample_chunks)

    assert stats["total_chunks"] == 3
    assert stats["total_words"] > 0
    assert stats["mean_chunk_words"] > 0.0
    assert stats["median_chunk_words"] > 0.0
    assert "top_keywords" in stats


def test_corpus_analyzer_empty():
    analyzer = CorpusAnalyzer()
    stats = analyzer.analyze_chunks([])
    assert stats["total_chunks"] == 0
    assert stats["total_words"] == 0
    assert stats["mean_chunk_words"] == 0.0
