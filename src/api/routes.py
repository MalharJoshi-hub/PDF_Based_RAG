"""
Flask Blueprint defining RESTful API endpoints for documents, retrieval, generation, and analytics.
"""

from pathlib import Path
from flask import Blueprint, request, jsonify, current_app
from src.ingestion.pdf_parser import PDFParser
from src.chunking.structural_chunker import StructuralChunker
from src.persistence.database import DatabaseManager
from src.persistence.repository import DocumentRepository, ChunkRepository
from src.api.schemas import QueryRequestSchema, CompareRequestSchema

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Lazy service instances
_db = None
_doc_repo = None
_chunk_repo = None


def get_repos():
    global _db, _doc_repo, _chunk_repo
    if _db is None:
        _db = DatabaseManager()
        _doc_repo = DocumentRepository(_db)
        _chunk_repo = ChunkRepository(_db)
    return _doc_repo, _chunk_repo


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for Docker / orchestration liveness probe."""
    return jsonify({
        "status": "healthy",
        "service": "Enterprise PDF RAG API",
        "version": "2.4.0"
    }), 200


@api_bp.route("/pdfs", methods=["GET"])
def list_pdfs():
    """Returns list of uploaded PDFs."""
    doc_repo, _ = get_repos()
    pdfs = doc_repo.get_all()
    return jsonify(pdfs), 200


@api_bp.route("/chunks", methods=["GET"])
def list_chunks():
    """Lists chunks with optional filtering by pdf_id and limit."""
    pdf_id = request.args.get("pdf_id", type=int)
    limit = request.args.get("limit", default=100, type=int)
    _, chunk_repo = get_repos()

    if pdf_id:
        chunks = chunk_repo.get_by_pdf_id(pdf_id)
    else:
        chunks = chunk_repo.get_all()

    return jsonify({
        "total": len(chunks),
        "chunks": chunks[:limit]
    }), 200


@api_bp.route("/eda", methods=["GET"])
def get_eda():
    """Returns the latest corpus EDA analytics."""
    summary_path = Path("static/eda/eda_summary.json")
    if summary_path.exists():
        import json
        with open(summary_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f)), 200
    return jsonify({"error": "No EDA data found. Run EDA first."}), 404
