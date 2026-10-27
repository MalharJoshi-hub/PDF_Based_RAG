"""
End-to-End Enterprise RAG Pipeline orchestrating Retrieval, Neural Reranking, and Generative Synthesis.
"""

import re
import time
from typing import List, Dict, Any, Optional
from src.retrieval.retriever_manager import RetrieverManager
from src.retrieval.reranker import NeuralReranker
from src.generation.generator import FlanT5Generator


class RAGPipeline:
    """
    Enterprise RAG Pipeline:
    Smart Candidate Retrieval -> Neural Cross-Encoder Reranking -> Executive Synthesis.
    """

    def __init__(
        self,
        retriever_manager: RetrieverManager,
        reranker: Optional[NeuralReranker] = None,
        generator: Optional[FlanT5Generator] = None
    ):
        self.retriever_manager = retriever_manager
        self.reranker = reranker or NeuralReranker()
        self.generator = generator or FlanT5Generator()

    @staticmethod
    def _clean_text(raw_text: str) -> str:
        """Sanitizes text and preserves paragraph layout."""
        clean = re.sub(r"([a-z0-9\.])([A-Z])", r"\1 \2", raw_text)
        clean = clean.replace("\n", "\n\n")
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        return clean.strip()

    def process_query(
        self,
        query: str,
        retriever_method: str = "dense",
        top_k_retrieved: int = 4,
        top_k_rerank: int = 2,
        use_reranker: bool = True,
        chunks_lookup: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end question answering workflow over indexed PDF documents.

        Args:
            query: Question from user.
            retriever_method: 'dense', 'bm25', 'tfidf', or 'hybrid'.
            top_k_retrieved: Number of candidates to fetch initially.
            top_k_rerank: Number of top chunks to retain after reranking.
            use_reranker: Flag to apply neural cross-encoder reranking.
            chunks_lookup: Chunk ID to chunk record lookup dictionary.

        Returns:
            Dictionary matching frontend and API response schema.
        """
        total_start = time.perf_counter()

        # Step 1: Candidate Retrieval
        retrieval_res = self.retriever_manager.search(
            query=query,
            method=retriever_method,
            top_k=top_k_retrieved,
            chunks_lookup=chunks_lookup
        )
        candidates = retrieval_res["results"]
        retrieval_latency = retrieval_res["latency_ms"]

        # Step 2: Neural Reranking
        if use_reranker and candidates:
            final_chunks, rerank_latency = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=min(top_k_rerank, 4)
            )
        else:
            final_chunks = candidates[:min(top_k_rerank, 4)]
            rerank_latency = 0.0

        # Step 3: Synthesis & Context Assembly
        if final_chunks:
            final_chunks.sort(key=lambda x: x.get("chunk_index", 0))

            pdf_info_set = set()
            for c in final_chunks:
                filename = c.get("filename", "Unknown Document")
                page_num = c.get("page_number", "Unknown Page")
                pdf_info_set.add(f"**{filename}** (Page {page_num})")

            sources_header = "### Source: " + ", ".join(sorted(list(pdf_info_set))) + "\n\n"
            raw_text = "\n\n".join([c.get("text", "") for c in final_chunks])
            clean_text = self._clean_text(raw_text)

            # Generate AI Summary
            gen_start = time.perf_counter()
            if self.generator:
                ai_summary = self.generator.generate_answer(
                    f"Context: {clean_text[:800]}\n\nQuestion: {query}\nAnswer:"
                )
            else:
                ai_summary = "AI Summarization is not available."
            gen_latency = round((time.perf_counter() - gen_start) * 1000, 2)

            formatted_answer = (
                f"{sources_header}**AI Summary:**\n{ai_summary}\n\n---\n\n"
                f"**Detailed Extracted Context:**\n\n{clean_text}"
            )
        else:
            formatted_answer = "No relevant information found in the uploaded documents to answer this question."
            gen_latency = 0.0

        total_latency = round((time.perf_counter() - total_start) * 1000, 2)

        # Build citations list
        citations = []
        for i, c in enumerate(final_chunks, 1):
            full_text = c.get("text", "")
            preview = full_text[:180] + ("..." if len(full_text) > 180 else "")
            raw_score = c.get("score", 0.0)
            rel_pct = min(99, max(60, int(raw_score * 100))) if raw_score <= 1.0 else 95

            citations.append({
                "citation_id": i,
                "filename": c.get("filename", "Document"),
                "page_number": c.get("page_number", 1),
                "relevance_pct": rel_pct,
                "preview": preview,
                "full_text": full_text,
                "rerank_score": c.get("rerank_score")
            })

        return {
            "query": query,
            "answer": formatted_answer,
            "retriever": retriever_method,
            "use_reranker": use_reranker,
            "citations": citations,
            "timings": {
                "retrieval_ms": round(retrieval_latency, 2),
                "rerank_ms": round(rerank_latency, 2),
                "generation_ms": round(gen_latency, 2),
                "total_ms": round(total_latency, 2)
            }
        }
