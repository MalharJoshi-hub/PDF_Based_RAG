"""
Structural text chunker splitting documents by headers and structural section boundaries.
"""

import re
from typing import List, Dict, Any, Optional
from src.chunking.base import BaseChunker, DocumentChunk
from src.chunking.token_counter import TokenCounter


class StructuralChunker(BaseChunker):
    """
    Intelligent structural chunker.
    Uses regex to identify headers and section boundaries in documents.
    If a section exceeds the predefined chunk size, it applies a sliding window with overlap.
    """

    DEFAULT_HEADER_REGEX = r"(?:^|\n)(?=[A-Z0-9])([^\n]{2,55}(?<![\,\.\;\-\:]))\n"

    def __init__(
        self,
        chunk_size: int = 150,
        overlap: int = 30,
        min_chunk_len: int = 20,
        header_regex: Optional[str] = None
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_len = min_chunk_len
        self.header_regex = re.compile(header_regex or self.DEFAULT_HEADER_REGEX)

    def _split_words_sliding_window(self, text: str, page_number: int, start_idx: int) -> List[DocumentChunk]:
        """Applies sliding window over word tokens."""
        words = text.split()
        chunks: List[DocumentChunk] = []
        step = max(1, self.chunk_size - self.overlap)

        curr_idx = start_idx
        for i in range(0, len(words), step):
            window = words[i : i + self.chunk_size]
            chunk_text = " ".join(window).strip()
            if len(chunk_text) >= self.min_chunk_len:
                tokens = TokenCounter.count_tokens_approx(chunk_text)
                chunks.append(
                    DocumentChunk(
                        chunk_index=curr_idx,
                        page_number=page_number,
                        text=chunk_text,
                        token_count=tokens
                    )
                )
                curr_idx += 1

        return chunks

    def split_text(self, text: str, page_number: int = 1) -> List[DocumentChunk]:
        """
        Splits a single body of text based on structural headers and fallback sliding window.
        """
        if not text:
            return []

        # Find header match boundaries
        matches = list(self.header_regex.finditer(text))
        chunks: List[DocumentChunk] = []

        if not matches:
            # No clear headers detected: fallback to sliding window
            return self._split_words_sliding_window(text, page_number, 0)

        sections: List[str] = []
        last_end = 0

        for match in matches:
            start = match.start()
            if start > last_end:
                sec = text[last_end:start].strip()
                if sec:
                    sections.append(sec)
            last_end = start

        trailing = text[last_end:].strip()
        if trailing:
            sections.append(trailing)

        curr_idx = 0
        for section in sections:
            words = section.split()
            if len(words) <= self.chunk_size:
                if len(section) >= self.min_chunk_len:
                    chunks.append(
                        DocumentChunk(
                            chunk_index=curr_idx,
                            page_number=page_number,
                            text=section,
                            token_count=TokenCounter.count_tokens_approx(section)
                        )
                    )
                    curr_idx += 1
            else:
                sub_chunks = self._split_words_sliding_window(section, page_number, curr_idx)
                chunks.extend(sub_chunks)
                curr_idx += len(sub_chunks)

        return chunks

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a multi-page document into chunks matching the legacy chunk_document schema.
        """
        all_chunks: List[Dict[str, Any]] = []
        global_index = 0

        for page in pages:
            page_num = page.get("page_number", 1)
            text = page.get("text", "")
            if not text:
                continue

            page_chunks = self.split_text(text, page_num)
            for c in page_chunks:
                all_chunks.append({
                    "chunk_index": global_index,
                    "page_number": c.page_number,
                    "text": c.text,
                    "token_count": c.token_count
                })
                global_index += 1

        return all_chunks


def chunk_document(
    pages: List[Dict[str, Any]],
    chunk_size: int = 150,
    overlap: int = 30
) -> List[Dict[str, Any]]:
    """
    Backwards-compatible wrapper matching retrieval.chunk_document signature.
    """
    chunker = StructuralChunker(chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk_pages(pages)
