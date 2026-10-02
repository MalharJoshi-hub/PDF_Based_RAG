import re
import time
from typing import List, Dict, Any, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import CrossEncoder

RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATOR_MODEL_ID = "google/flan-t5-base"

def clean_sentence(text: str) -> str:
    """Cleans up PDF extraction artifacts and formats cleanly."""
    t = " ".join(text.split())
    # Remove weird non-ascii or merged header words if any
    t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
    return t.strip()

def extract_key_takeaways(query: str, chunks: List[Dict[str, Any]], max_points: int = 3) -> List[str]:
    """
    Extracts the most salient, high-information sentences from the top chunks
    that directly answer the user's question, eliminating wall-of-text fatigue.
    """
    if not chunks:
        return []

    query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
    candidate_sentences = []

    for chunk in chunks[:2]:
        text = chunk.get("text", "")
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for s in sentences:
            s_clean = clean_sentence(s)
            # Filter out very short headers or overly long run-on text
            if 35 <= len(s_clean) <= 220:
                s_lower = s_clean.lower()
                s_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', s_lower))
                overlap = len(query_words.intersection(s_words))
                
                # Boost sentences containing informative predicates
                boost = 1.0
                if any(k in s_lower for k in ["is a", "provides", "allows", "enables", "eliminates", "acts as", "designed to", "serves as", "helps", "primary function"]):
                    boost = 1.4
                score = (overlap + 1) * boost
                candidate_sentences.append((score, s_clean))

    candidate_sentences.sort(key=lambda x: x[0], reverse=True)

    picked = []
    seen = set()
    for _, s in candidate_sentences:
        # Avoid near duplicates
        key = " ".join(s.split()[:4]).lower()
        if key not in seen:
            seen.add(key)
            # Ensure it ends with a period
            if not s.endswith(('.', '!', '?')):
                s += '.'
            picked.append(s)
        if len(picked) >= max_points:
            break

    return picked


class NeuralReranker:
    """Cross-encoder reranker for high-precision passage selection."""
    def __init__(self, model_name: str = RERANKER_MODEL_ID):
        print(f"[NeuralReranker] Loading model: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("[NeuralReranker] Model loaded.")

    def rerank(
        self,
        query: str,
        candidate_chunks: List[Dict[str, Any]],
        top_k: int = 2
    ) -> Tuple[List[Dict[str, Any]], float]:
        if not candidate_chunks or not query.strip():
            return [], 0.0

        start_t = time.perf_counter()
        pairs = [[query, chunk["text"]] for chunk in candidate_chunks]
        scores = self.model.predict(pairs)

        scored_chunks = []
        for chunk, score in zip(candidate_chunks, scores):
            item = dict(chunk)
            item["rerank_score"] = round(float(score), 4)
            scored_chunks.append(item)

        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_reranked = scored_chunks[:top_k]
        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

        return top_reranked, latency_ms


class FlanT5Generator:
    """Generative summarizer using Google's Flan-T5-base."""
    def __init__(self, model_name: str = GENERATOR_MODEL_ID):
        print(f"[FlanT5Generator] Loading tokenizer and model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        print("[FlanT5Generator] Model loaded.")

    def generate_summary(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_new_tokens: int = 160
    ) -> Dict[str, Any]:
        """
        Synthesizes a clean, direct answer to the question using context.
        """
        start_t = time.perf_counter()
        if not context_chunks:
            return {
                "summary": "The uploaded documents do not contain information to answer this question.",
                "latency_ms": 0.0
            }

        # Keep context compact (~200 words max from each top chunk) to prevent T5 context overflow
        compact_snippets = []
        for c in context_chunks[:2]:
            words = c["text"].split()[:180]
            compact_snippets.append(" ".join(words))
        context_text = " ".join(compact_snippets)

        prompt = (
            f"Context: {context_text}\n\n"
            f"Based on the context above, write a clear and concise summary to answer: {query}\n"
            "Summary:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_length=15,
                num_beams=3,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        gen_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

        return {
            "summary": gen_text,
            "latency_ms": latency_ms
        }


class RAGPipeline:
    """
    Enterprise RAG Pipeline:
    Smart Candidate Retrieval -> Neural Cross-Encoder Reranking -> Executive Synthesis.
    """
    def __init__(self, retriever_manager):
        self.retriever_manager = retriever_manager
        self.reranker = NeuralReranker()
        self.generator = FlanT5Generator()

    def process_query(
        self,
        query: str,
        retriever_method: str = "dense",
        top_k_retrieved: int = 4,
        top_k_rerank: int = 2,
        use_reranker: bool = True,
        chunks_lookup: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
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

        # Step 2: Neural Reranking (Cross-Attention)
        if use_reranker and candidates:
            final_chunks, rerank_latency = self.reranker.rerank(
                query=query,
                candidate_chunks=candidates,
                top_k=min(top_k_rerank, 4)  # Cap at top 4 best chunks to reconstruct up to 4 slides
            )
        else:
            final_chunks = candidates[:min(top_k_rerank, 4)]
            rerank_latency = 0.0

        # Step 3: Synthesis & Key Takeaway Extraction
        if final_chunks:
            # Sort chunks by their original index so they flow logically if they are adjacent
            final_chunks.sort(key=lambda x: x.get("chunk_index", 0))
            
            # Extract PDF names and page numbers from the retrieved chunks
            pdf_info_set = set()
            for c in final_chunks:
                # The chunks in FAISS might not have filename if it wasn't passed, let's check
                # Wait, final_chunks are fetched from sqlite by ID in retrieval.py!
                # Do they have filename? Yes, get_chunks in db.py joins with pdfs table and returns filename!
                filename = c.get("filename", "Unknown Document")
                page_num = c.get("page_number", "Unknown Page")
                pdf_info_set.add(f"**{filename}** (Page {page_num})")
                
            sources_header = "### Source: " + ", ".join(sorted(list(pdf_info_set))) + "\n\n"
            
            raw_text = "\n\n".join([c.get("text", "") for c in final_chunks])
            
            # Simple cleanup: replace multiple spaces with single space, but preserve newlines
            # Fix PyPDF2 column merging artifacts (e.g., "querying.Mapping" -> "querying. Mapping")
            clean_text = re.sub(r'([a-z0-9\.])([A-Z])', r'\1 \2', raw_text)
            
            # We want to keep newlines because they represent layout.
            # Markdown requires double newlines to render separate blocks (like headers/paragraphs).
            clean_text = clean_text.replace('\n', '\n\n')
            clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
            
            # Generate AI Summary
            if self.generator:
                gen_res = self.generator.generate_summary(query, final_chunks)
                ai_summary = gen_res["summary"]
                gen_latency = gen_res["latency_ms"]
            else:
                ai_summary = "AI Summarization is not available."
                gen_latency = 0.0
            
            formatted_answer = sources_header + "**AI Summary:**\n" + ai_summary + "\n\n---\n\n**Detailed Extracted Context:**\n\n" + clean_text
        else:
            formatted_answer = "No relevant information found in the uploaded documents to answer this question."
            gen_latency = 0.0

        total_latency = round((time.perf_counter() - total_start) * 1000, 2)

        # Prepare compact, stress-free citations
        citations = []
        for i, c in enumerate(final_chunks, 1):
            full_text = c.get("text", "")
            # Short preview (~180 chars)
            preview = full_text[:180] + ("..." if len(full_text) > 180 else "")
            
            # Map score to clean relevance percentage
            raw_score = c.get("score", 0.0)
            rel_pct = min(99, max(60, int(raw_score * 100))) if raw_score <= 1.0 else 95

            citations.append({
                "citation_id": i,
                "filename": c.get("filename", "Document"),
                "page_number": c.get("page_number", 1),
                "relevance_pct": rel_pct,
                "preview": preview,
                "full_text": full_text
            })

        return {
            "query": query,
            "answer": formatted_answer,
            "retriever": retriever_method,
            "use_reranker": use_reranker,
            "citations": citations,
            "timings": {
                "retrieval_ms": retrieval_latency,
                "rerank_ms": rerank_latency,
                "generation_ms": gen_latency,
                "total_ms": total_latency
            }
        }
