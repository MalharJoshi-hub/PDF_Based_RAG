import json
import re
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless server
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud, STOPWORDS

import db

EDA_DIR = Path("static/eda")
EDA_DIR.mkdir(parents=True, exist_ok=True)

STOP_WORDS = set(STOPWORDS).union({
    "hibernate", "java", "application", "database", "data", "using", "use",
    "will", "can", "also", "one", "two", "provides", "used", "table", "class"
})

def run_eda():
    print("[EDA] Loading chunks from DB...")
    chunks = db.get_all_chunks()
    if not chunks:
        print("[EDA] No chunks found in DB. Upload documents first.")
        return

    print(f"[EDA] Found {len(chunks)} chunks across documents.")

    # 1. Statistical Calculations
    chunk_lengths = [len(c["text"].split()) for c in chunks]
    char_lengths = [len(c["text"]) for c in chunks]
    all_text = " ".join([c["text"] for c in chunks])
    
    # Tokenize words (lowercase, alphanumeric)
    words = re.findall(r"\b[a-zA-Z]{3,}\b", all_text.lower())
    filtered_words = [w for w in words if w not in STOP_WORDS]
    word_counts = Counter(filtered_words)
    top_20 = word_counts.most_common(20)

    stats = {
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

    # Save summary stats JSON
    stats_path = EDA_DIR / "eda_summary.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[EDA] Saved statistics to {stats_path}")

    # Set dark aesthetic for charts to match dark mode theme
    plt.style.use("dark_background")

    # 2. Chunk Length Distribution Chart
    plt.figure(figsize=(9, 5), dpi=200)
    plt.hist(chunk_lengths, bins=max(5, len(chunks)//2 + 2), color="#6366f1", edgecolor="#818cf8", alpha=0.85)
    plt.axvline(np.mean(chunk_lengths), color="#f43f5e", linestyle="--", linewidth=2, label=f"Mean: {stats['mean_chunk_words']} words")
    plt.axvline(np.median(chunk_lengths), color="#10b981", linestyle=":", linewidth=2, label=f"Median: {stats['median_chunk_words']} words")
    plt.title("Chunk Length Distribution (Word Count)", fontsize=14, fontweight="bold", pad=15, color="#f8fafc")
    plt.xlabel("Words per Chunk", fontsize=11, labelpad=10, color="#cbd5e1")
    plt.ylabel("Chunk Frequency", fontsize=11, labelpad=10, color="#cbd5e1")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155")
    plt.tight_layout()
    chart1_path = EDA_DIR / "chunk_length_distribution.png"
    plt.savefig(chart1_path)
    plt.close()
    print(f"[EDA] Saved chart to {chart1_path}")

    # 3. Word Frequency Bar Chart
    if top_20:
        labels, counts = zip(*top_20)
        plt.figure(figsize=(10, 6), dpi=200)
        y_pos = np.arange(len(labels))
        plt.barh(y_pos, counts, color="#38bdf8", edgecolor="#0284c7", alpha=0.85)
        plt.yticks(y_pos, labels, fontsize=10, color="#e2e8f0")
        plt.gca().invert_yaxis()
        plt.title("Top 20 Domain Keywords in Corpus", fontsize=14, fontweight="bold", pad=15, color="#f8fafc")
        plt.xlabel("Frequency Count", fontsize=11, labelpad=10, color="#cbd5e1")
        plt.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()
        chart2_path = EDA_DIR / "word_frequency.png"
        plt.savefig(chart2_path)
        plt.close()
        print(f"[EDA] Saved chart to {chart2_path}")

    # 4. Word Cloud Generation
    if filtered_words:
        wc = WordCloud(
            width=1000,
            height=500,
            background_color="#0f172a",
            colormap="viridis",
            stopwords=STOP_WORDS,
            max_words=120,
            collocations=False
        ).generate(" ".join(filtered_words))

        plt.figure(figsize=(12, 6), dpi=200)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Corpus Domain Word Cloud", fontsize=15, fontweight="bold", pad=15, color="#f8fafc")
        plt.tight_layout(pad=0)
        chart3_path = EDA_DIR / "wordcloud.png"
        plt.savefig(chart3_path)
        plt.close()
        print(f"[EDA] Saved wordcloud to {chart3_path}")

    print("[EDA] Completed successfully!")
    return stats

if __name__ == "__main__":
    run_eda()
