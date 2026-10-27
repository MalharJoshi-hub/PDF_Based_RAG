"""
Unit tests for PDFParser and text cleaning functionality.
"""

import io
from src.ingestion.pdf_parser import PDFParser, ExtractedPage, extract_pages_from_pdf


def test_clean_text_basic():
    raw = "Hello   world!\r\nThis is a\xa0test.\n\n\n\nNew paragraph."
    cleaned = PDFParser.clean_text(raw)
    assert "Hello world!" in cleaned
    assert "\xa0" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "\r" not in cleaned


def test_clean_text_empty():
    assert PDFParser.clean_text("") == ""
    assert PDFParser.clean_text(None) == ""


def test_clean_text_removes_control_characters():
    text_with_control = "Text\x00with\x07nulls\x1fand bells."
    cleaned = PDFParser.clean_text(text_with_control)
    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    assert "\x1f" not in cleaned
    assert "Textwithnullsand bells." in cleaned or "Text with nulls" in cleaned


def test_extracted_page_to_dict():
    page = ExtractedPage(
        page_number=1,
        text="Sample page text content.",
        char_count=26,
        word_count=4,
        raw_text="Sample page text content."
    )
    data = page.to_dict()
    assert data["page_number"] == 1
    assert data["word_count"] == 4
    assert data["char_count"] == 26
    assert data["text"] == "Sample page text content."


def test_parse_invalid_stream():
    parser = PDFParser()
    bad_stream = io.BytesIO(b"Not a valid PDF file")
    try:
        parser.parse_stream(bad_stream)
        assert False, "Expected ValueError on bad PDF stream"
    except ValueError as e:
        assert "Failed to initialize PDF reader" in str(e)
