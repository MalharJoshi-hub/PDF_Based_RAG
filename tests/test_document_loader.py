"""
Unit tests for DocumentLoader module and LoadedDocument properties.
"""

from unittest.mock import MagicMock
from src.ingestion.document_loader import DocumentLoader, LoadedDocument
from src.ingestion.pdf_parser import ExtractedPage


def test_loaded_document_properties():
    pages = [
        ExtractedPage(page_number=1, text="Hello world", char_count=11, word_count=2, raw_text="Hello world"),
        ExtractedPage(page_number=2, text="Second page text", char_count=16, word_count=3, raw_text="Second page text")
    ]
    doc = LoadedDocument(
        filename="test.pdf",
        sha256_hash="abc12345",
        file_size_bytes=1024,
        pages=pages
    )
    assert doc.total_pages == 2
    assert doc.total_words == 5
    assert doc.total_chars == 27
    assert "Hello world\n\nSecond page text" == doc.full_text()


def test_document_loader_load_bytes_success(sample_pdf_bytes):
    mock_parser = MagicMock()
    mock_parser.parse_stream.return_value = [
        ExtractedPage(page_number=1, text="Sample text", char_count=11, word_count=2, raw_text="Sample text")
    ]
    loader = DocumentLoader(parser=mock_parser)
    loaded = loader.load_bytes(sample_pdf_bytes, "test.pdf")

    assert loaded.filename == "test.pdf"
    assert loaded.total_pages == 1
    assert len(loaded.sha256_hash) == 64
    assert loaded.file_size_bytes == len(sample_pdf_bytes)


def test_document_loader_invalid_bytes():
    loader = DocumentLoader()
    try:
        loader.load_bytes(b"invalid data", "bad.pdf")
        assert False, "Expected ValueError on invalid bytes"
    except ValueError as e:
        assert "Document validation failed" in str(e)


def test_document_loader_file_not_found(tmp_path):
    loader = DocumentLoader()
    non_existent = tmp_path / "does_not_exist.pdf"
    try:
        loader.load_file(non_existent)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass
