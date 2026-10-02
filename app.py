import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import db
import retrieval
import generation

# ----------------- App Setup -----------------
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Initialize database schema and migrate existing data
db.init_db()
db.migrate_existing_data(chunk_func=retrieval.chunk_raw_text)

# Initialize retriever manager & RAG pipeline
retriever_manager = retrieval.RetrieverManager()
rag_pipeline = generation.RAGPipeline(retriever_manager)

def refresh_retrievers():
    """Reloads all chunks from DB and updates FAISS, TF-IDF, and BM25 indices."""
    chunks = db.get_all_chunks()
    retriever_manager.rebuild_all(chunks)
    return chunks

# Initial index load
_initial_chunks = refresh_retrievers()
print(f"[RAG Server] Initialized with {len(_initial_chunks)} chunks across active documents.")

# ----------------- Routes -----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/pdfs", methods=["GET"])
def list_pdfs():
    """Returns all uploaded PDFs with page and chunk metrics."""
    pdfs = db.get_all_pdfs()
    return jsonify(pdfs)

@app.route("/api/pdfs", methods=["POST"])
def upload_pdf():
    """Uploads a PDF, extracts page-level text, splits into sliding-window chunks, and updates indices."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected or empty filename"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    try:
        pages = retrieval.extract_pages_from_pdf(file.stream)
    except Exception as e:
        return jsonify({"error": f"Failed to parse PDF: {str(e)}"}), 400

    if not pages:
        return jsonify({"error": "Could not extract any readable text from the PDF"}), 400

    # Section-based chunking with optimal overlap for presentation slides (~150 words per chunk)
    chunks = retrieval.chunk_document(pages, chunk_size=150, overlap=30)
    if not chunks:
        return jsonify({"error": "No chunks could be generated from extracted text"}), 400

    # Save to SQLite
    total_pages = len(pages)
    pdf_id = db.insert_pdf(file.filename, total_pages=total_pages, total_chunks=len(chunks))
    chunk_ids = db.insert_chunks(pdf_id, chunks)

    # Rebuild FAISS, TF-IDF, and BM25 indices
    all_chunks = refresh_retrievers()

    return jsonify({
        "status": "success",
        "id": pdf_id,
        "filename": file.filename,
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "total_corpus_chunks": len(all_chunks)
    }), 201

@app.route("/api/pdfs/<int:pdf_id>", methods=["DELETE"])
def delete_pdf(pdf_id: int):
    """Deletes a PDF and its associated chunks, updating all retrievers."""
    success = db.delete_pdf(pdf_id)
    if not success:
        return jsonify({"error": "PDF not found"}), 404

    all_chunks = refresh_retrievers()
    return jsonify({
        "status": "deleted",
        "pdf_id": pdf_id,
        "remaining_chunks": len(all_chunks)
    })

@app.route("/api/chunks", methods=["GET"])
def list_chunks():
    """Lists chunks with optional filtering by pdf_id and limit for EDA / inspection."""
    pdf_id = request.args.get("pdf_id", type=int)
    limit = request.args.get("limit", default=100, type=int)
    
    all_chunks = db.get_all_chunks()
    if pdf_id:
        all_chunks = [c for c in all_chunks if c["pdf_id"] == pdf_id]
        
    return jsonify({
        "total": len(all_chunks),
        "chunks": all_chunks[:limit]
    })

@app.route("/api/rebuild", methods=["POST"])
def rebuild_indices():
    """Manually forces a rebuild of all retriever indices (FAISS, TF-IDF, BM25)."""
    all_chunks = refresh_retrievers()
    return jsonify({
        "status": "rebuilt",
        "total_chunks": len(all_chunks)
    })

@app.route("/api/query", methods=["POST"])
def query():
    """
    End-to-End Question Answering:
    Retrieval (Dense / TF-IDF / BM25) -> Neural Reranker (CrossEncoder) -> Generation (Flan-T5-base).
    """
    payload = request.get_json(silent=True) or {}
    q = (payload.get("query") or "").strip()
    method = (payload.get("retriever") or "dense").lower()
    top_k_retrieved = int(payload.get("top_k_retrieved", payload.get("top_k", 5)))
    top_k_rerank = int(payload.get("top_k_rerank", 3))
    use_reranker = bool(payload.get("use_reranker", True))

    if not q:
        return jsonify({"error": "Question/query cannot be empty"}), 400

    all_chunks = db.get_all_chunks()
    chunks_lookup = {c["id"]: c for c in all_chunks}

    result = rag_pipeline.process_query(
        query=q,
        retriever_method=method,
        top_k_retrieved=top_k_retrieved,
        top_k_rerank=top_k_rerank,
        use_reranker=use_reranker,
        chunks_lookup=chunks_lookup
    )

    return jsonify(result)

@app.route("/api/compare", methods=["POST"])
def compare_retrievers():
    """
    Executes Dense (FAISS), TF-IDF, and BM25 side-by-side on the query.
    Returns latencies, top chunks, cross-retriever overlap statistics,
    and optional generation comparisons.
    """
    payload = request.get_json(silent=True) or {}
    q = (payload.get("query") or "").strip()
    top_k = int(payload.get("top_k", 3))
    include_generation = bool(payload.get("include_generation", False))

    if not q:
        return jsonify({"error": "Question/query cannot be empty"}), 400

    all_chunks = db.get_all_chunks()
    chunks_lookup = {c["id"]: c for c in all_chunks}

    comparison = retriever_manager.compare_all(
        query=q,
        top_k=top_k,
        chunks_lookup=chunks_lookup
    )

    if include_generation:
        answers = {}
        for m in ["dense", "tfidf", "bm25"]:
            hits = comparison[m]["results"]
            if hits:
                gen_res = rag_pipeline.generator.generate_answer(q, hits)
                answers[m] = {
                    "answer": gen_res["answer"],
                    "generation_ms": gen_res["latency_ms"]
                }
            else:
                answers[m] = {"answer": "No chunks found", "generation_ms": 0}
        comparison["generated_answers"] = answers

    return jsonify(comparison)

@app.route("/api/eda", methods=["GET"])
def get_eda():
    """Returns the latest corpus EDA analytics."""
    summary_path = Path("static/eda/eda_summary.json")
    if summary_path.exists():
        import json
        with open(summary_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No EDA data found. Run EDA first."}), 404

@app.route("/api/eda/run", methods=["POST"])
def trigger_eda():
    """Runs EDA analysis on the active database corpus and generates charts."""
    import eda
    stats = eda.run_eda()
    return jsonify({
        "status": "success",
        "stats": stats
    })

# ----------------- Main -----------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
