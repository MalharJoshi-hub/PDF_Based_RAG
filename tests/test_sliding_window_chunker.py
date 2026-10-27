"""
Unit tests for SlidingWindowChunker module.
"""

import pytest
from src.chunking.sliding_window_chunker import SlidingWindowChunker, chunk_raw_text


def test_sliding_window_invalid_overlap():
    with pytest.raises(ValueError):
        SlidingWindowChunker(chunk_size=10, overlap=10)


def test_sliding_window_basic_overlap():
    chunker = SlidingWindowChunker(chunk_size=5, overlap=2, min_chunk_len=5)
    text = "one two three four five six seven eight nine ten"
    chunks = chunker.split_text(text, page_number=1)
    assert len(chunks) >= 3
    # First chunk: one two three four five
    assert "one two three four five" == chunks[0].text
    # Second chunk: starts with four five six seven eight
    assert "four five" in chunks[1].text


def test_sliding_window_empty():
    chunker = SlidingWindowChunker()
    assert chunker.split_text("") == []


def test_chunk_raw_text_wrapper():
    text = "word " * 40
    chunks = chunk_raw_text(text, chunk_size=20, overlap=5)
    assert len(chunks) >= 2
    assert "chunk_index" in chunks[0]
    assert "token_count" in chunks[0]
