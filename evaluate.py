import json
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import db
import retrieval
import generation

EVAL_DIR = Path("static/eda")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Evaluation Benchmark Queries & Relevance Keywords for our Corpus
EVAL_BENCHMARK = [
    {
        "query": "What is Hibernate and what does it provide?",
        "intent_keywords": ["hibernate", "orm", "mapping", "java", "relational", "database"]
    },
    {
        "query": "What is SessionFactory in Hibernate?",
        "intent_keywords": ["sessionfactory", "session", "thread-safe", "configuration", "mapping"]
    },
    {
        "query": "How does Object-Relational Mapping (ORM) work?",
        "intent_keywords": ["mapping", "object", "relational", "tables", "domain", "persistence"]
    },
    {
        "query": "What is the role of the Session interface in Hibernate?",
        "intent_keywords": ["session", "transaction", "save", "crud", "persistence", "connection"]
    },
    {
        "query": "What is JDBC and how does Hibernate interact with it?",
        "intent_keywords": ["jdbc", "driver", "sql", "connection", "query", "underlying"]
    }
]

def calculate_relevance(chunk_text: str, keywords: List[str]) -> float:
    """Calculates keyword overlap precision as a ground-truth proxy for relevance."""
    text_lower = chunk_text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return min(1.0, matches / max(1, len(keywords) * 0.5))

def run_evaluation():
    print("="*60)
    print("STARTING RETRIEVAL & GENERATION BENCHMARK EVALUATION (Day 4)")
    print("="*60)

    chunks = db.get_all_chunks()
    lookup = {c["id"]: c for c in chunks}
    
    retriever_mgr = retrieval.RetrieverManager()
    retriever_mgr.rebuild_all(chunks)
    pipeline = generation.RAGPipeline(retriever_mgr)

    methods = ["dense", "bm25", "tfidf"]
    metrics = {
        m: {
            "latencies": [],
            "p_at_1": [],
            "p_at_3": [],
            "p_at_5": []
        }
        for m in methods
    }

    spot_checks = []

    for item in EVAL_BENCHMARK:
        q = item["query"]
        kws = item["intent_keywords"]
        print(f"\n[Evaluating Query] {q}")

        query_spot_check = {"query": q, "methods": {}}

        for m in methods:
            # Measure retrieval latency & accuracy
            t0 = time.perf_counter()
            res = retriever_mgr.search(q, method=m, top_k=5, chunks_lookup=lookup)
            lat = (time.perf_counter() - t0) * 1000
            metrics[m]["latencies"].append(lat)

            hits = res["results"]
            # Precision @ K calculation
            relevances = [calculate_relevance(h.get("text", ""), kws) for h in hits]
            
            p1 = relevances[0] if len(relevances) > 0 else 0.0
            p3 = np.mean(relevances[:3]) if len(relevances) >= 3 else (np.mean(relevances) if relevances else 0.0)
            p5 = np.mean(relevances[:5]) if len(relevances) >= 5 else (np.mean(relevances) if relevances else 0.0)

            metrics[m]["p_at_1"].append(p1)
            metrics[m]["p_at_3"].append(p3)
            metrics[m]["p_at_5"].append(p5)

            # Generate spot check answer
            if hits:
                gen_res = pipeline.generator.generate_answer(q, hits[:2])
                answer = gen_res["answer"]
                gen_lat = gen_res["latency_ms"]
            else:
                answer = "No chunks found"
                gen_lat = 0.0

            query_spot_check["methods"][m] = {
                "top_hit_score": hits[0]["score"] if hits else 0,
                "latency_ms": round(lat, 2),
                "generated_answer": answer,
                "generation_ms": round(gen_lat, 2)
            }

        spot_checks.append(query_spot_check)

    # Compute Aggregate Stats
    summary = {}
    for m in methods:
        summary[m] = {
            "mean_latency_ms": round(float(np.mean(metrics[m]["latencies"])), 2),
            "std_latency_ms": round(float(np.std(metrics[m]["latencies"])), 2),
            "precision_at_1": round(float(np.mean(metrics[m]["p_at_1"])), 3),
            "precision_at_3": round(float(np.mean(metrics[m]["p_at_3"])), 3),
            "precision_at_5": round(float(np.mean(metrics[m]["p_at_5"])), 3)
        }

    results_data = {
        "benchmark_summary": summary,
        "spot_checks": spot_checks
    }

    # Save Evaluation JSON
    eval_json_path = EVAL_DIR / "evaluation_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n[Saved] Evaluation data to {eval_json_path}")

    # Generate Evaluation Charts
    plt.style.use("dark_background")

    # 1. Precision@K Comparison Chart
    plt.figure(figsize=(9, 5), dpi=200)
    bar_width = 0.25
    x = np.arange(3)  # P@1, P@3, P@5
    
    dense_scores = [summary["dense"]["precision_at_1"], summary["dense"]["precision_at_3"], summary["dense"]["precision_at_5"]]
    bm25_scores = [summary["bm25"]["precision_at_1"], summary["bm25"]["precision_at_3"], summary["bm25"]["precision_at_5"]]
    tfidf_scores = [summary["tfidf"]["precision_at_1"], summary["tfidf"]["precision_at_3"], summary["tfidf"]["precision_at_5"]]

    plt.bar(x - bar_width, dense_scores, width=bar_width, label="Dense (FAISS)", color="#6366f1", alpha=0.9)
    plt.bar(x, bm25_scores, width=bar_width, label="BM25 (Okapi)", color="#ec4899", alpha=0.9)
    plt.bar(x + bar_width, tfidf_scores, width=bar_width, label="TF-IDF (Cosine)", color="#10b981", alpha=0.9)

    plt.title("Retriever Precision Benchmark (P@1, P@3, P@5)", fontsize=13, fontweight="bold", pad=15, color="#f8fafc")
    plt.xlabel("Metric", fontsize=11, labelpad=10, color="#cbd5e1")
    plt.ylabel("Relevance Precision Score", fontsize=11, labelpad=10, color="#cbd5e1")
    plt.xticks(x, ["Precision@1", "Precision@3", "Precision@5"], color="#f1f5f9")
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155")
    plt.tight_layout()
    chart1_path = EVAL_DIR / "retriever_precision_comparison.png"
    plt.savefig(chart1_path)
    plt.close()
    print(f"[Saved] Precision comparison chart to {chart1_path}")

    # 2. Latency Comparison Chart
    plt.figure(figsize=(8, 4.5), dpi=200)
    methods_labels = ["Dense (FAISS)", "BM25 (Okapi)", "TF-IDF (Cosine)"]
    latencies = [summary["dense"]["mean_latency_ms"], summary["bm25"]["mean_latency_ms"], summary["tfidf"]["mean_latency_ms"]]
    colors = ["#6366f1", "#ec4899", "#10b981"]

    bars = plt.bar(methods_labels, latencies, color=colors, width=0.45, alpha=0.9)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f} ms", ha="center", va="bottom", fontsize=10, color="#f8fafc", fontweight="bold")

    plt.title("Retriever Latency Comparison (Mean Milliseconds)", fontsize=13, fontweight="bold", pad=15, color="#f8fafc")
    plt.ylabel("Latency (ms)", fontsize=11, labelpad=10, color="#cbd5e1")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    chart2_path = EVAL_DIR / "retriever_latency_comparison.png"
    plt.savefig(chart2_path)
    plt.close()
    print(f"[Saved] Latency comparison chart to {chart2_path}")

    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY TABLE:")
    print("="*60)
    print(f"{'Retriever':<18} | {'Mean Latency':<12} | {'P@1':<8} | {'P@3':<8} | {'P@5':<8}")
    print("-" * 65)
    for m in methods:
        s = summary[m]
        print(f"{m.upper():<18} | {s['mean_latency_ms']:>8.2f} ms | {s['precision_at_1']:>6.3f} | {s['precision_at_3']:>6.3f} | {s['precision_at_5']:>6.3f}")
    print("="*60)

    return results_data

if __name__ == "__main__":
    run_evaluation()
