"""
Sliding window chunking strategy with configurable window size, overlap, and token thresholds.
"""

from typing import List, Dict, Any
from src.chunking.base import BaseChunker, DocumentChunk
from src.chunking.token_counter import TokenCounter


class SlidingWindowChunker(BaseChunker):
    """
    Fixed-size sliding window text chunker with configurable token or word overlap.
    Ensures continuity across chunk boundaries for optimal retrieval context.
    """

    def __init__(self, chunk_size: int = 150, overlap: int = 30, min_chunk_len: int = 20):
        if overlap >= chunk_size:
            raise ValueError("Overlap must be strictly smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_len = min_chunk_len

    def split_text(self, text: str, page_number: int = 1) -> List[DocumentChunk]:
        """Splits arbitrary text into overlapping chunks."""
        words = text.split()
        if not words:
            return []

        chunks: List[DocumentChunk] = []
        step = max(1, self.chunk_size - self.overlap)
        chunk_idx = 0

        for i in range(0, len(words), step):
            window = words[i : i + self.chunk_size]
            chunk_text = " ".join(window).strip()
            if len(chunk_text) >= self.min_chunk_len:
                tokens = TokenCounter.count_tokens_approx(chunk_text)
                chunks.append(
                    DocumentChunk(
                        chunk_index=chunk_idx,
                        page_number=page_number,
                        text=chunk_text,
                        token_count=tokens
                    )
                )
                chunk_idx += 1

        return chunks

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes list of pages using sliding window."""
        results: List[Dict[str, Any]] = []
        global_idx = 0
        for page in pages:
            page_num = page.get("page_number", 1)
            text = page.get("text", "")
            page_chunks = self.split_text(text, page_number=page_num)
            for c in page_chunks:
                item = c.to_dict()
                item["chunk_index"] = global_idx
                results.append(item)
                global_idx += 1
        return results


def chunk_raw_text(text: str, chunk_size: int = 150, overlap: int = 30) -> List[Dict[str, Any]]:
    """
    Backwards-compatible wrapper matching retrieval.chunk_raw_text signature.
    """
    chunker = SlidingWindowChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.split_text(text, page_number=1)
    return [c.to_dict() for c in chunks]
