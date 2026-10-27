"""
Unit tests for StructuralChunker module.
"""

from src.chunking.structural_chunker import StructuralChunker, chunk_document


def test_structural_chunker_empty():
    chunker = StructuralChunker()
    chunks = chunker.split_text("")
    assert chunks == []


def test_structural_chunker_fallback_sliding_window():
    chunker = StructuralChunker(chunk_size=10, overlap=2, min_chunk_len=5)
    text = "word " * 30  # 30 words with no header
    chunks = chunker.split_text(text, page_number=1)
    assert len(chunks) > 1
    assert all(c.page_number == 1 for c in chunks)


def test_structural_chunker_with_headers():
    chunker = StructuralChunker(chunk_size=50, overlap=10, min_chunk_len=10)
    text = (
        "Introduction\n"
        "This is the introductory section explaining the architecture of Hibernate ORM.\n\n"
        "Configuration Details\n"
        "This section covers database connection settings and XML mappings for tables.\n"
    )
    chunks = chunker.split_text(text, page_number=2)
    assert len(chunks) >= 2
    assert any("introductory section" in c.text.lower() for c in chunks)
    assert any("configuration details" in c.text.lower() or "database connection" in c.text.lower() for c in chunks)


def test_chunk_document_wrapper():
    pages = [
        {"page_number": 1, "text": "Overview\nThis is page one text covering basic concepts."},
        {"page_number": 2, "text": "Deep Dive\nThis is page two text covering advanced concepts."}
    ]
    chunks = chunk_document(pages, chunk_size=50, overlap=10)
    assert len(chunks) >= 2
    assert chunks[0]["page_number"] == 1
    assert chunks[1]["page_number"] == 2
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1
