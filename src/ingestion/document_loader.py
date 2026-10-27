"""
Document loader managing file ingestion, hashing, and metadata extraction.
"""

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, BinaryIO
from src.ingestion.pdf_parser import PDFParser, ExtractedPage
from src.ingestion.validator import DocumentValidator


@dataclass
class LoadedDocument:
    """Represents an ingested document with full metadata and extracted pages."""
    filename: str
    sha256_hash: str
    file_size_bytes: int
    pages: List[ExtractedPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def total_words(self) -> int:
        return sum(p.word_count for p in self.pages)

    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)

    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


class DocumentLoader:
    """Orchestrates document loading, checksum computation, and parsing."""

    def __init__(self, parser: Optional[PDFParser] = None):
        self.parser = parser or PDFParser()

    def load_bytes(self, data: bytes, filename: str) -> LoadedDocument:
        """
        Loads and processes document from raw byte array.

        Args:
            data: Raw PDF bytes.
            filename: Original file name.

        Returns:
            LoadedDocument instance.
        """
        is_valid, error = DocumentValidator.validate_pdf_bytes(data)
        if not is_valid:
            raise ValueError(f"Document validation failed: {error}")

        sha256 = hashlib.sha256(data).hexdigest()
        stream = io.BytesIO(data)
        pages = self.parser.parse_stream(stream)

        return LoadedDocument(
            filename=filename,
            sha256_hash=sha256,
            file_size_bytes=len(data),
            pages=pages,
            metadata={"source": "bytes", "format": "pdf"}
        )

    def load_stream(self, stream: Union[BinaryIO, io.BytesIO], filename: str) -> LoadedDocument:
        """Loads and processes document from a readable stream."""
        if hasattr(stream, "seek"):
            stream.seek(0)
        data = stream.read()
        return self.load_bytes(data, filename)

    def load_file(self, file_path: Union[str, Path]) -> LoadedDocument:
        """Loads and processes document from filesystem path."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        data = path.read_bytes()
        return self.load_bytes(data, path.name)
