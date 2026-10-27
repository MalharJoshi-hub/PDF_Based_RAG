"""
Unit tests for RecursiveCharacterChunker module.
"""

from src.chunking.recursive_chunker import RecursiveCharacterChunker


def test_recursive_chunker_empty():
    chunker = RecursiveCharacterChunker()
    assert chunker.split_text("") == []


def test_recursive_chunker_paragraphs():
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    text = (
        "First paragraph with some text.\n\n"
        "Second paragraph with more detailed explanation.\n\n"
        "Third paragraph concluding the section."
    )
    chunks = chunker.split_text(text, page_number=1)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 150 for c in chunks)


def test_recursive_chunker_pages():
    chunker = RecursiveCharacterChunker(chunk_size=80, chunk_overlap=10)
    pages = [
        {"page_number": 1, "text": "Page one text content that is relatively short."},
        {"page_number": 2, "text": "Page two text content that is also relatively short."}
    ]
    results = chunker.chunk_pages(pages)
    assert len(results) >= 2
    assert results[0]["chunk_index"] == 0
    assert results[1]["chunk_index"] == 1
