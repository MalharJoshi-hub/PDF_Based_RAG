"""
Citation manager formatting source attribution and grounded markdown payloads.
"""

from typing import List, Dict, Any
from src.generation.base import Citation


class CitationManager:
    """Extracts, organizes, and formats source citations for RAG responses."""

    @staticmethod
    def extract_citations(chunks: List[Dict[str, Any]], max_chars: int = 300) -> List[Citation]:
        """Converts retrieved and reranked chunks into structured Citation models."""
        citations: List[Citation] = []
        for idx, c in enumerate(chunks, start=1):
            text = c.get("text", "")
            snippet = text[:max_chars].strip() + ("..." if len(text) > max_chars else "")
            citations.append(
                Citation(
                    citation_id=idx,
                    pdf_id=c.get("pdf_id", 0),
                    filename=c.get("filename", "Unknown Document"),
                    page_number=c.get("page_number", 1),
                    chunk_index=c.get("chunk_index", 0),
                    snippet=snippet,
                    score=float(c.get("score", 0.0)),
                    rerank_score=c.get("rerank_score")
                )
            )
        return citations

    @classmethod
    def format_markdown(cls, answer: str, citations: List[Citation]) -> str:
        """Formats the final answer and structured citations into a clean Markdown document."""
        sections = [
            "### AI Generated Answer",
            answer,
            "",
            "---",
            "### Grounded Citations & Sources"
        ]

        if not citations:
            sections.append("_No direct document sources were cited._")
        else:
            for cit in citations:
                score_label = f" (Rerank Score: {cit.rerank_score:.2f})" if cit.rerank_score is not None else f" (Similarity: {cit.score:.2f})"
                sections.append(
                    f"**[Source {cit.citation_id}]** `{cit.filename}` — Page {cit.page_number}{score_label}\n"
                    f"> \"{cit.snippet}\"\n"
                )

        return "\n".join(sections)
