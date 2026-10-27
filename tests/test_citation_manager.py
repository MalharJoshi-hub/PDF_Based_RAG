"""
Unit tests for CitationManager module.
"""

from src.generation.citation_manager import CitationManager


def test_extract_citations():
    chunks = [
        {
            "id": 1,
            "pdf_id": 10,
            "filename": "guide.pdf",
            "page_number": 3,
            "chunk_index": 2,
            "text": "Extracted text segment from page 3.",
            "score": 0.92,
            "rerank_score": 2.45
        }
    ]
    citations = CitationManager.extract_citations(chunks)
    assert len(citations) == 1
    c = citations[0]
    assert c.citation_id == 1
    assert c.filename == "guide.pdf"
    assert c.page_number == 3
    assert c.rerank_score == 2.45
    assert "Extracted text segment" in c.snippet


def test_format_markdown():
    chunks = [
        {
            "id": 1,
            "pdf_id": 10,
            "filename": "guide.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "Context text here.",
            "score": 0.95
        }
    ]
    citations = CitationManager.extract_citations(chunks)
    md = CitationManager.format_markdown("AI generated answer text.", citations)

    assert "### AI Generated Answer" in md
    assert "AI generated answer text." in md
    assert "### Grounded Citations & Sources" in md
    assert "`guide.pdf` — Page 1" in md
