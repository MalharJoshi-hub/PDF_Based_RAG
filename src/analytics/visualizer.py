"""
Matplotlib chart generation for corpus distribution, word frequency, and word cloud.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class AnalyticsVisualizer:
    """Generates styled charts and word clouds for exploratory data analysis."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("static/eda")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_length_distribution(self, chunk_lengths: List[int]) -> Path:
        """Renders histogram of chunk word count distribution."""
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

        mean_val = float(np.mean(chunk_lengths)) if chunk_lengths else 0.0
        med_val = float(np.median(chunk_lengths)) if chunk_lengths else 0.0

        ax.hist(
            chunk_lengths,
            bins=max(5, len(chunk_lengths) // 2 + 2),
            color="#6366f1",
            edgecolor="#818cf8",
            alpha=0.85
        )
        ax.axvline(mean_val, color="#f43f5e", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.1f} words")
        ax.axvline(med_val, color="#10b981", linestyle=":", linewidth=2, label=f"Median: {med_val:.1f} words")

        ax.set_title("Chunk Length Distribution (Word Count)", fontsize=13, fontweight="bold", pad=12, color="#f8fafc")
        ax.set_xlabel("Words per Chunk", fontsize=10, labelpad=8, color="#cbd5e1")
        ax.set_ylabel("Chunk Frequency", fontsize=10, labelpad=8, color="#cbd5e1")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155")

        fig.tight_layout()
        out_path = self.output_dir / "chunk_length_distribution.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def render_word_frequency(self, top_keywords: Dict[str, int]) -> Optional[Path]:
        """Renders horizontal bar chart of top domain keywords."""
        if not top_keywords:
            return None

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        labels = list(top_keywords.keys())
        counts = list(top_keywords.values())
        y_pos = np.arange(len(labels))

        ax.barh(y_pos, counts, color="#38bdf8", edgecolor="#0284c7", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10, color="#e2e8f0")
        ax.invert_yaxis()

        ax.set_title("Top Domain Keywords in Corpus", fontsize=13, fontweight="bold", pad=12, color="#f8fafc")
        ax.set_xlabel("Frequency Count", fontsize=10, labelpad=8, color="#cbd5e1")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        fig.tight_layout()
        out_path = self.output_dir / "word_frequency.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
