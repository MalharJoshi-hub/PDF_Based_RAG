import os
import re
import time
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# ----------------- Configuration -----------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
FAISS_PATH = DATA_DIR / "faiss.index"
IDMAP_PATH = DATA_DIR / "id_map.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMB_DIM = 384

# ----------------- PDF Chunking -----------------
def extract_pages_from_pdf(file_obj) -> List[Dict[str, Any]]:
    """
    Extracts text per-page from a file object or path using PyPDF2.
    Returns a list of dicts: [{'page_num': 1, 'text': '...'}]
    """
    reader = PyPDF2.PdfReader(file_obj)
    pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        # Preserve newlines but clean up multiple spaces horizontally
        lines = []
        for line in page_text.split('\n'):
            line = " ".join(line.split())
            if line:
                lines.append(line)
        cleaned = "\n".join(lines)
        if cleaned:
            pages.append({"page_num": i + 1, "text": cleaned})
    return pages

def chunk_document(
    pages: List[Dict[str, Any]],
    chunk_size: int = 150,
    overlap: int = 30
) -> List[Dict[str, Any]]:
    """
    Splits document into logically bound sections using a generic structural header detector.
    Preserves newlines for robust layout-based splitting and extraction.
    """
    if not pages:
        return []

    # Reconstruct full text and map char index to page number
    full_text = ""
    char_to_page = []
    
    for page in pages:
        text = page["text"].strip() + "\n"
        p_num = page["page_num"]
        full_text += text
        char_to_page.extend([p_num] * len(text))

    # Generic header regex:
    # A header is usually a short line that starts with an Uppercase letter or Number.
    # It should not end with sentence-ending punctuation like period, comma, semicolon, or hyphen.
    # It can end in a lowercase letter or a colon.
    header_regex = re.compile(
        r'(?:^|\n)(?=[A-Z0-9])([^\n]{2,55}(?<![\,\.\;\-\:]))\n'
    )
    
    split_char_indices = [0]
    for m in header_regex.finditer(full_text):
        idx = m.start()
        if full_text[idx] == '\n':
            idx += 1
        split_char_indices.append(idx)
    split_char_indices.append(len(full_text))
    
    chunks = []
    
    for i in range(len(split_char_indices) - 1):
        start_c = split_char_indices[i]
        end_c = split_char_indices[i+1]
        
        section_text = full_text[start_c:end_c].strip()
        if not section_text:
            continue
            
        c_tokens = re.split(r'(\s+)', section_text)
        # Filter out empty strings from the edges if any
        c_words_only = [t for t in c_tokens if t.strip()]
        
        if not c_words_only:
            continue
            
        # Determine primary page by sampling char_to_page
        c_pages = char_to_page[start_c:end_c:10]
        primary_page = max(set(c_pages), key=c_pages.count) if c_pages else 1
        
        # If the section exceeds chunk_size, fallback to sliding window within it
        if len(c_words_only) > chunk_size + overlap:
            # We must preserve the original whitespace (newlines) when joining the chunk.
            # We can map word indices back to token indices.
            word_to_token = []
            for idx, t in enumerate(c_tokens):
                if t.strip():
                    word_to_token.append(idx)
                    
            s_start = 0
            step = max(1, chunk_size - overlap)
            while s_start < len(word_to_token):
                s_end = min(s_start + chunk_size, len(word_to_token))
                
                t_start = word_to_token[s_start]
                t_end = word_to_token[s_end - 1] + 1 # Include the last word
                
                sub_text = "".join(c_tokens[t_start:t_end]).strip()
                
                chunks.append({
                    "chunk_index": len(chunks),
                    "page_number": primary_page,
                    "text": sub_text,
                    "token_count": s_end - s_start
                })
                
                if s_end >= len(word_to_token):
                    break
                s_start += step
        else:
            chunks.append({
                "chunk_index": len(chunks),
                "page_number": primary_page,
                "text": section_text,
                "token_count": len(c_words_only)
            })

    # Renumber chunk_index just in case
    for idx, c in enumerate(chunks):
        c["chunk_index"] = idx

    return chunks

def chunk_raw_text(
    text: str,
    chunk_size: int = 150,
    overlap: int = 30
) -> List[Dict[str, Any]]:
    """Fallback chunker for plain text strings."""
    return chunk_document([{"page_num": 1, "text": text}], chunk_size, overlap)


# ----------------- Dense Retriever (FAISS) -----------------
class DenseRetriever:
    """Dense vector retriever using SentenceTransformer + FAISS IndexFlatIP (Cosine Similarity)."""
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.id_map: Dict[int, int] = {}  # faiss_idx -> chunk_id
        self.load()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return (vectors / norms).astype("float32")

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return self._normalize(embeddings)

    def load(self):
        if FAISS_PATH.exists() and IDMAP_PATH.exists():
            try:
                self.index = faiss.read_index(str(FAISS_PATH))
                with open(IDMAP_PATH, "rb") as f:
                    self.id_map = pickle.load(f)
                return
            except Exception as e:
                print(f"[DenseRetriever] Error loading index: {e}, initializing fresh.")
        self.index = faiss.IndexFlatIP(EMB_DIM)
        self.id_map = {}

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(FAISS_PATH))
            with open(IDMAP_PATH, "wb") as f:
                pickle.dump(self.id_map, f)

    def rebuild(self, chunks: List[Dict[str, Any]]):
        """Rebuilds the entire FAISS index from a list of chunks with 'id' and 'text'."""
        self.index = faiss.IndexFlatIP(EMB_DIM)
        self.id_map = {}
        if not chunks:
            self.save()
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.encode(texts)
        self.index.add(embeddings)
        self.id_map = {i: chunks[i]["id"] for i in range(len(chunks))}
        self.save()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0 or not query.strip():
            return []

        q_emb = self.encode([query])
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q_emb, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            chunk_id = self.id_map.get(int(idx))
            if chunk_id is not None:
                results.append({
                    "chunk_id": chunk_id,
                    "score": round(float(dist), 4),
                    "method": "dense"
                })
        return results


# ----------------- TF-IDF Baseline Retriever -----------------
class TfidfRetriever:
    """Sparse baseline retriever using Scikit-Learn TF-IDF + Cosine Similarity."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2)
        )
        self.matrix = None
        self.chunk_ids: List[int] = []

    def fit(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            self.matrix = None
            self.chunk_ids = []
            return

        self.chunk_ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        try:
            self.matrix = self.vectorizer.fit_transform(texts)
        except ValueError:
            # E.g. empty vocabulary
            self.matrix = None
            self.chunk_ids = []

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.matrix is None or not self.chunk_ids or not query.strip():
            return []

        try:
            q_vec = self.vectorizer.transform([query])
        except Exception:
            return []

        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score <= 0:
                continue
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "score": round(score, 4),
                "method": "tfidf"
            })
        return results


# ----------------- BM25 Baseline Retriever -----------------
class BM25Retriever:
    """Sparse baseline retriever using BM25Okapi."""
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[int] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            self.bm25 = None
            self.chunk_ids = []
            return

        self.chunk_ids = [c["id"] for c in chunks]
        tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.bm25 is None or not self.chunk_ids or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "score": round(score, 4),
                "method": "bm25"
            })
        return results


# ----------------- Unified Retriever Manager -----------------
class RetrieverManager:
    """Coordinates Dense (FAISS), TF-IDF, and BM25 retrievers and performs multi-retriever lookups."""
    def __init__(self):
        self.dense = DenseRetriever()
        self.tfidf = TfidfRetriever()
        self.bm25 = BM25Retriever()

    def rebuild_all(self, all_chunks: List[Dict[str, Any]]):
        """Rebuilds Dense, TF-IDF, and BM25 indices on current chunks."""
        self.dense.rebuild(all_chunks)
        self.tfidf.fit(all_chunks)
        self.bm25.fit(all_chunks)

    def search(
        self,
        query: str,
        method: str = "dense",
        top_k: int = 5,
        chunks_lookup: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Executes search using the requested retriever method ('dense', 'tfidf', 'bm25').
        Enriches results with chunk metadata if chunks_lookup is provided.
        """
        start_t = time.perf_counter()
        method = method.lower()
        if method == "tfidf":
            raw_hits = self.tfidf.search(query, top_k)
        elif method == "bm25":
            raw_hits = self.bm25.search(query, top_k)
        else:
            method = "dense"
            raw_hits = self.dense.search(query, top_k)
        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

        results = []
        for hit in raw_hits:
            item = dict(hit)
            if chunks_lookup and hit["chunk_id"] in chunks_lookup:
                chunk_data = chunks_lookup[hit["chunk_id"]]
                item.update({
                    "text": chunk_data["text"],
                    "filename": chunk_data.get("filename", "Unknown"),
                    "page_number": chunk_data.get("page_number", 1),
                    "chunk_index": chunk_data.get("chunk_index", 0),
                    "token_count": chunk_data.get("token_count", 0),
                    "pdf_id": chunk_data.get("pdf_id")
                })
            results.append(item)

        return {
            "method": method,
            "query": query,
            "top_k": top_k,
            "latency_ms": latency_ms,
            "results": results
        }

    def compare_all(
        self,
        query: str,
        top_k: int = 5,
        chunks_lookup: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Runs all three retrievers for comparative evaluation."""
        dense_res = self.search(query, method="dense", top_k=top_k, chunks_lookup=chunks_lookup)
        tfidf_res = self.search(query, method="tfidf", top_k=top_k, chunks_lookup=chunks_lookup)
        bm25_res = self.search(query, method="bm25", top_k=top_k, chunks_lookup=chunks_lookup)

        dense_ids = set(h["chunk_id"] for h in dense_res["results"])
        tfidf_ids = set(h["chunk_id"] for h in tfidf_res["results"])
        bm25_ids = set(h["chunk_id"] for h in bm25_res["results"])

        overlap_dense_bm25 = list(dense_ids.intersection(bm25_ids))
        overlap_dense_tfidf = list(dense_ids.intersection(tfidf_ids))
        overlap_all = list(dense_ids.intersection(bm25_ids).intersection(tfidf_ids))

        return {
            "query": query,
            "dense": dense_res,
            "tfidf": tfidf_res,
            "bm25": bm25_res,
            "comparison": {
                "overlap_dense_bm25_count": len(overlap_dense_bm25),
                "overlap_dense_tfidf_count": len(overlap_dense_tfidf),
                "overlap_all_count": len(overlap_all)
            }
        }
