"""
Unit tests for TokenCounter module.
"""

from src.chunking.token_counter import TokenCounter


def test_count_words_basic():
    text = "Hibernate is an Object-Relational Mapping (ORM) solution."
    count = TokenCounter.count_words(text)
    assert count == 8



def test_count_words_empty():
    assert TokenCounter.count_words("") == 0
    assert TokenCounter.count_words(None) == 0


def test_count_tokens_approx():
    text = "Short sentence for token test."
    tokens = TokenCounter.count_tokens_approx(text)
    assert tokens >= 1
    assert tokens == len(text) // 4


def test_count_tokens_approx_empty():
    assert TokenCounter.count_tokens_approx("") == 0


def test_count_tokens_tiktoken_or_fallback():
    text = "Testing exact or estimated token counter."
    count = TokenCounter.count_tokens(text)
    assert count >= 5
