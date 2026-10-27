"""
Recursive character text chunker splitting recursively across paragraphs, sentences, and words.
"""

from typing import List, Dict, Any, Optional
from src.chunking.base import BaseChunker, DocumentChunk
from src.chunking.token_counter import TokenCounter


class RecursiveCharacterChunker(BaseChunker):
    """
    Hierarchical chunker that splits text by attempting separators in order:
    ["\n\n", "\n", ". ", " ", ""]
    Keeps chunks under target character/word limits while preserving semantic cohesion.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text_with_separator(self, text: str, separators: List[str]) -> List[str]:
        """Recursive helper splitting text along separators."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._split_text_with_separator(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.append(s)

        # Merge splits up to chunk_size with overlap
        current_chunk: List[str] = []
        current_length = 0

        for piece in good_splits:
            piece_len = len(piece)
            sep_len = len(separator) if current_chunk else 0

            if current_length + piece_len + sep_len > self.chunk_size:
                if current_chunk:
                    doc = separator.join(current_chunk).strip()
                    if doc:
                        final_chunks.append(doc)

                    # Handle overlap
                    while current_chunk and current_length > self.chunk_overlap:
                        removed = current_chunk.pop(0)
                        current_length -= len(removed) + len(separator)

                current_chunk.append(piece)
                current_length = sum(len(p) for p in current_chunk) + len(separator) * (len(current_chunk) - 1)
            else:
                current_chunk.append(piece)
                current_length += piece_len + sep_len

        if current_chunk:
            doc = separator.join(current_chunk).strip()
            if doc:
                final_chunks.append(doc)

        return final_chunks

    def split_text(self, text: str, page_number: int = 1) -> List[DocumentChunk]:
        """Splits raw text recursively."""
        if not text:
            return []

        raw_chunks = self._split_text_with_separator(text, self.separators)
        chunks: List[DocumentChunk] = []

        for idx, c_text in enumerate(raw_chunks):
            tokens = TokenCounter.count_tokens_approx(c_text)
            chunks.append(
                DocumentChunk(
                    chunk_index=idx,
                    page_number=page_number,
                    text=c_text,
                    token_count=tokens
                )
            )

        return chunks

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes pages through recursive character splitting."""
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
