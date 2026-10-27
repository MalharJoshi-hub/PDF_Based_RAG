"""
Document validation utilities for incoming files and byte streams.
"""

import io
from typing import Tuple, Optional


class DocumentValidator:
    """Validates document payloads for type, structure, and integrity."""

    MAX_FILE_SIZE_BYTES = 64 * 1024 * 1024  # 64 MB
    PDF_MAGIC_BYTES = b"%PDF-"

    @classmethod
    def validate_pdf_bytes(cls, data: bytes) -> Tuple[bool, Optional[str]]:
        """
        Validates raw bytes representing a PDF file.

        Args:
            data: Raw byte payload.

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
        """
        if not data:
            return False, "Document payload is empty."

        if len(data) > cls.MAX_FILE_SIZE_BYTES:
            max_mb = cls.MAX_FILE_SIZE_BYTES / (1024 * 1024)
            return False, f"File size exceeds maximum allowed limit of {max_mb:.0f} MB."

        if not data.startswith(cls.PDF_MAGIC_BYTES):
            # Check within the first 1024 bytes in case of leading whitespace/comments
            header_slice = data[:1024]
            if cls.PDF_MAGIC_BYTES not in header_slice:
                return False, "Invalid PDF header: file does not match PDF signature '%PDF-'."

        return True, None

    @classmethod
    def validate_stream(cls, stream: io.IOBase) -> Tuple[bool, Optional[str]]:
        """
        Validates an I/O stream by inspecting initial magic bytes without consuming the stream.

        Args:
            stream: Seekable I/O stream.

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
        """
        if stream is None:
            return False, "Stream cannot be None."

        try:
            curr_pos = stream.tell() if hasattr(stream, "tell") else 0
            header = stream.read(1024)
            if hasattr(stream, "seek"):
                stream.seek(curr_pos)
        except Exception as e:
            return False, f"Unable to read stream: {str(e)}"

        if not header:
            return False, "Stream is empty or unreadable."

        if cls.PDF_MAGIC_BYTES not in header:
            return False, "Stream does not contain valid PDF magic bytes."

        return True, None
