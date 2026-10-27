"""
Unit tests for DocumentRepository and ChunkRepository.
"""

from src.persistence.repository import DocumentRepository, ChunkRepository


def test_document_repository_crud(temp_db):
    doc_repo = DocumentRepository(temp_db)

    # Insert
    pdf_id = doc_repo.insert(filename="manual.pdf", total_pages=4, total_chunks=8)
    assert pdf_id >= 1

    # Get by ID
    doc = doc_repo.get_by_id(pdf_id)
    assert doc is not None
    assert doc["filename"] == "manual.pdf"
    assert doc["total_pages"] == 4

    # Get all
    all_docs = doc_repo.get_all()
    assert len(all_docs) >= 1
    assert any(d["id"] == pdf_id for d in all_docs)

    # Delete
    deleted = doc_repo.delete(pdf_id)
    assert deleted is True
    assert doc_repo.get_by_id(pdf_id) is None


def test_chunk_repository_crud(temp_db):
    doc_repo = DocumentRepository(temp_db)
    chunk_repo = ChunkRepository(temp_db)

    pdf_id = doc_repo.insert(filename="test_doc.pdf", total_pages=2, total_chunks=2)
    chunks_data = [
        {"chunk_index": 0, "page_number": 1, "text": "First chunk text", "token_count": 3},
        {"chunk_index": 1, "page_number": 2, "text": "Second chunk text", "token_count": 3}
    ]

    chunk_ids = chunk_repo.insert_batch(pdf_id, chunks_data)
    assert len(chunk_ids) == 2

    # Get all chunks
    all_chunks = chunk_repo.get_all()
    assert len(all_chunks) == 2
    assert all_chunks[0]["filename"] == "test_doc.pdf"

    # Get by IDs
    lookup = chunk_repo.get_by_ids([chunk_ids[0]])
    assert chunk_ids[0] in lookup
    assert lookup[chunk_ids[0]]["text"] == "First chunk text"

    # Get by PDF ID
    pdf_chunks = chunk_repo.get_by_pdf_id(pdf_id)
    assert len(pdf_chunks) == 2
