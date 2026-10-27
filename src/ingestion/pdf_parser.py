"""
PDF text extraction engine using PyPDF2 with layout cleanup and page-level metadata.
"""

import io
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Union, BinaryIO
from PyPDF2 import PdfReader


@dataclass
class ExtractedPage:
    """Represents text extracted from a single PDF page."""
    page_number: int
    text: str
    char_count: int
    word_count: int
    raw_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert page metadata and content to dictionary representation."""
        return {
            "page_number": self.page_number,
            "text": self.text,
            "char_count": self.char_count,
            "word_count": self.word_count
        }


class PDFParser:
    """Enterprise parser for extracting and sanitizing text from PDF files."""

    def __init__(self, min_char_threshold: int = 5):
        self.min_char_threshold = min_char_threshold

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans extracted PDF text:
        - Replaces non-breaking and multi-horizontal spaces with a single space.
        - Preserves vertical structure (newlines) which are crucial for headings and bullets.
        - Removes control characters and excess blank lines.
        """
        if not text:
            return ""

        # Remove control characters except standard newlines and tabs
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Replace non-breaking spaces
        text = text.replace("\xa0", " ")

        # Collapse horizontal whitespace
        text = re.sub(r"[ \t]+", " ", text)

        # Clean up line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse 3 or more consecutive newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def parse_stream(self, stream: Union[BinaryIO, io.BytesIO]) -> List[ExtractedPage]:
        """
        Extracts structured text page-by-page from a binary stream.

        Args:
            stream: Binary stream of the PDF file.

        Returns:
            List of ExtractedPage objects.
        """
        try:
            reader = PdfReader(stream)
        except Exception as e:
            raise ValueError(f"Failed to initialize PDF reader from stream: {e}")

        pages: List[ExtractedPage] = []

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            try:
                raw_text = page.extract_text() or ""
            except Exception as extract_err:
                raw_text = ""

            cleaned = self.clean_text(raw_text)
            if len(cleaned) >= self.min_char_threshold:
                words = re.findall(r"\b\w+\b", cleaned)
                pages.append(
                    ExtractedPage(
                        page_number=page_num,
                        text=cleaned,
                        char_count=len(cleaned),
                        word_count=len(words),
                        raw_text=raw_text
                    )
                )

        return pages

    def parse_file(self, file_path: str) -> List[ExtractedPage]:
        """Convenience method to parse a local PDF file path."""
        with open(file_path, "rb") as f:
            return self.parse_stream(f)


def extract_pages_from_pdf(pdf_file: Union[BinaryIO, io.BytesIO]) -> List[Dict[str, Any]]:
    """
    Backwards-compatible wrapper matching the legacy retrieval.extract_pages_from_pdf signature.
    Returns list of dicts: [{'page_number': int, 'text': str, 'char_count': int, 'word_count': int}]
    """
    parser = PDFParser()
    extracted = parser.parse_stream(pdf_file)
    return [p.to_dict() for p in extracted]
