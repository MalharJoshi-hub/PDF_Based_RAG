"""
Prompt engineering templates, context compression, and key point extraction.
"""

import re
from typing import List, Dict, Any


class PromptBuilder:
    """Constructs focused, anti-hallucination prompts for sequence-to-sequence models."""

    @staticmethod
    def extract_key_points(query: str, text: str, max_points: int = 4) -> List[str]:
        """
        Extracts high-salience sentences from context chunks that directly relate to the query.
        Filters boilerplate and duplicates.
        """
        if not text:
            return []

        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 35]

        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        scored = []

        boost_keywords = [
            "is a", "provides", "allows", "enables", "eliminates",
            "acts as", "serves as", "features", "consists of", "designed to"
        ]

        for s in sentences:
            s_lower = s.lower()
            s_words = set(re.findall(r"\b\w{3,}\b", s_lower))
            overlap = len(query_words.intersection(s_words))

            boost = 1.5 if any(w in s_lower for w in boost_keywords) else 1.0
            score = (overlap + 1) * boost
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)

        picked: List[str] = []
        seen = set()
        for _, s in scored:
            clean_s = s.strip()
            first_words = " ".join(clean_s.split()[:4]).lower()
            if first_words not in seen:
                seen.add(first_words)
                picked.append(clean_s)
            if len(picked) >= max_points:
                break

        return picked

    @classmethod
    def build_qa_prompt(
        cls,
        query: str,
        context_chunks: List[Dict[str, Any]],
        template_style: str = "instruction",
        max_context_chars: int = 1500
    ) -> str:
        """
        Builds a grounded QA prompt from top context chunks.

        Args:
            query: User's question.
            context_chunks: List of chunk dictionaries containing 'text'.
            template_style: 'instruction', 'summarization', or 'direct'.
            max_context_chars: Character limit to prevent tokenizer truncation.

        Returns:
            Formatted prompt string.
        """
        context_parts = []
        chars_used = 0

        for idx, c in enumerate(context_chunks, start=1):
            raw_text = c.get("text", "")
            key_points = cls.extract_key_points(query, raw_text, max_points=3)
            segment_text = " ".join(key_points) if key_points else raw_text[:400]

            if chars_used + len(segment_text) > max_context_chars:
                budget_left = max_context_chars - chars_used
                if budget_left > 100:
                    context_parts.append(f"[Source {idx}]: {segment_text[:budget_left]}...")
                break

            context_parts.append(f"[Source {idx}]: {segment_text}")
            chars_used += len(segment_text)

        joined_context = "\n\n".join(context_parts)

        if template_style == "summarization":
            return (
                f"Context:\n{joined_context}\n\n"
                f"Question: {query}\n"
                f"Summarize the answer clearly and concisely in 2-3 sentences based strictly on the context:"
            )
        elif template_style == "direct":
            return (
                f"Read this context carefully:\n{joined_context}\n\n"
                f"What is the explanation for: {query}?"
            )
        else:  # default 'instruction'
            return (
                f"Context information:\n{joined_context}\n\n"
                f"Answer the question using only the facts from the context above: {query}"
            )
