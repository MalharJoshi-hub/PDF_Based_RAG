"""
Unit tests for DocumentValidator module.
"""

import io
from src.ingestion.validator import DocumentValidator


def test_validate_pdf_bytes_success(sample_pdf_bytes):
    is_valid, error = DocumentValidator.validate_pdf_bytes(sample_pdf_bytes)
    assert is_valid is True
    assert error is None


def test_validate_pdf_bytes_empty():
    is_valid, error = DocumentValidator.validate_pdf_bytes(b"")
    assert is_valid is False
    assert "empty" in error.lower()


def test_validate_pdf_bytes_invalid_header():
    is_valid, error = DocumentValidator.validate_pdf_bytes(b"INVALID_HEADER_DATA_12345")
    assert is_valid is False
    assert "invalid pdf header" in error.lower()


def test_validate_pdf_bytes_exceeds_size():
    # Artificially set max size to test boundary
    orig_max = DocumentValidator.MAX_FILE_SIZE_BYTES
    DocumentValidator.MAX_FILE_SIZE_BYTES = 100
    try:
        large_bytes = b"%PDF-" + b"0" * 200
        is_valid, error = DocumentValidator.validate_pdf_bytes(large_bytes)
        assert is_valid is False
        assert "exceeds" in error.lower()
    finally:
        DocumentValidator.MAX_FILE_SIZE_BYTES = orig_max


def test_validate_stream_success(sample_pdf_bytes):
    stream = io.BytesIO(sample_pdf_bytes)
    is_valid, error = DocumentValidator.validate_stream(stream)
    assert is_valid is True
    assert error is None
    # Stream position should be restored
    assert stream.tell() == 0


def test_validate_stream_none():
    is_valid, error = DocumentValidator.validate_stream(None)
    assert is_valid is False
    assert "none" in error.lower()


def test_validate_stream_empty():
    empty_stream = io.BytesIO(b"")
    is_valid, error = DocumentValidator.validate_stream(empty_stream)
    assert is_valid is False
    assert "empty" in error.lower()
