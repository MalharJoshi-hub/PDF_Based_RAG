"""
Corpus statistical analytics: word counts, distributions, vocabulary, and keyword extraction.
"""

import re
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import Counter
import numpy as np

STOP_WORDS = {
    "hibernate", "java", "application", "database", "data", "using", "use",
    "will", "can", "also", "one", "two", "provides", "used", "table", "class",
    "the", "and", "is", "in", "to", "of", "it", "with", "as", "for", "on", "that",
    "this", "by", "an", "be", "are", "from", "at", "which", "or", "an"
}


class CorpusAnalyzer:
    """Computes comprehensive corpus-level text metrics and keyword frequency distributions."""

    def __init__(self, stop_words: Optional[set] = None):
        self.stop_words = stop_words or STOP_WORDS

    def analyze_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates descriptive statistics over all corpus text chunks.

        Args:
            chunks: List of chunk dictionaries containing 'text'.

        Returns:
            Dictionary with statistical metrics and top keywords.
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_words": 0,
                "total_unique_words": 0,
                "mean_chunk_words": 0.0,
                "median_chunk_words": 0.0,
                "std_chunk_words": 0.0,
                "min_chunk_words": 0,
                "max_chunk_words": 0,
                "top_keywords": {}
            }

        chunk_lengths = [len(c.get("text", "").split()) for c in chunks]
        all_text = " ".join(c.get("text", "") for c in chunks)

        words = re.findall(r"\b[a-zA-Z]{3,}\b", all_text.lower())
        filtered_words = [w for w in words if w not in self.stop_words]
        word_counts = Counter(filtered_words)
        top_20 = word_counts.most_common(20)

        return {
            "total_chunks": len(chunks),
            "total_words": sum(chunk_lengths),
            "total_unique_words": len(set(words)),
            "mean_chunk_words": round(float(np.mean(chunk_lengths)), 2),
            "median_chunk_words": round(float(np.median(chunk_lengths)), 2),
            "std_chunk_words": round(float(np.std(chunk_lengths)), 2),
            "min_chunk_words": int(np.min(chunk_lengths)),
            "max_chunk_words": int(np.max(chunk_lengths)),
            "top_keywords": dict(top_20)
        }
