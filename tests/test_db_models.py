"""
Unit tests for persistent entity dataclass models.
"""

from src.persistence.models import PDFDocumentRecord, ChunkRecord, QueryAuditRecord


def test_pdf_document_record_to_dict():
    record = PDFDocumentRecord(
        id=1,
        filename="guide.pdf",
        total_pages=10,
        total_chunks=35,
        created_at="2026-10-27 12:00:00",
        active_chunks=35
    )
    d = record.to_dict()
    assert d["id"] == 1
    assert d["filename"] == "guide.pdf"
    assert d["total_pages"] == 10
    assert d["total_chunks"] == 35


def test_chunk_record_to_dict():
    record = ChunkRecord(
        id=10,
        pdf_id=1,
        chunk_index=3,
        text="Sample chunk text",
        page_number=2,
        token_count=15,
        filename="guide.pdf"
    )
    d = record.to_dict()
    assert d["id"] == 10
    assert d["pdf_id"] == 1
    assert d["chunk_index"] == 3
    assert d["filename"] == "guide.pdf"


def test_query_audit_record():
    rec = QueryAuditRecord(
        id=5,
        query="What is Hibernate?",
        retriever_method="dense",
        latency_ms=12.5,
        top_k=4,
        answer="Hibernate is an ORM solution."
    )
    assert rec.query == "What is Hibernate?"
    assert rec.latency_ms == 12.5
