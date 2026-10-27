"""
Repository layer for CRUD operations on documents, chunks, and queries.
"""

from typing import List, Dict, Any, Optional
from src.persistence.database import DatabaseManager
from src.persistence.models import PDFDocumentRecord, ChunkRecord


class DocumentRepository:
    """Handles database persistence for uploaded PDF documents."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def insert(self, filename: str, total_pages: int, total_chunks: int) -> int:
        """Inserts a new PDF record and returns its auto-generated ID."""
        with self.db.transaction() as cur:
            cur.execute("""
                INSERT INTO pdfs (filename, total_pages, total_chunks)
                VALUES (?, ?, ?)
            """, (filename, total_pages, total_chunks))
            return cur.lastrowid

    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieves all PDF records along with active chunk counts."""
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT p.id, p.filename, p.total_pages, p.total_chunks, p.created_at,
                       COUNT(c.id) AS active_chunks
                FROM pdfs p
                LEFT JOIN chunks c ON p.id = c.pdf_id
                GROUP BY p.id
                ORDER BY p.id DESC
            """)
            return [dict(row) for row in cur.fetchall()]

    def get_by_id(self, pdf_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves single PDF by ID."""
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM pdfs WHERE id = ?", (pdf_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def delete(self, pdf_id: int) -> bool:
        """Deletes a PDF and cascaded chunks."""
        with self.db.transaction() as cur:
            cur.execute("DELETE FROM chunks WHERE pdf_id = ?", (pdf_id,))
            cur.execute("DELETE FROM pdfs WHERE id = ?", (pdf_id,))
            return cur.rowcount > 0

    def update_chunk_count(self, pdf_id: int, total_chunks: int) -> None:
        """Updates total_chunks counter for a PDF."""
        with self.db.transaction() as cur:
            cur.execute("UPDATE pdfs SET total_chunks = ? WHERE id = ?", (total_chunks, pdf_id))


class ChunkRepository:
    """Handles database persistence for extracted text chunks."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def insert_batch(self, pdf_id: int, chunks: List[Dict[str, Any]]) -> List[int]:
        """Inserts batch of chunks within a single transaction."""
        chunk_ids: List[int] = []
        with self.db.transaction() as cur:
            for ch in chunks:
                cur.execute("""
                    INSERT INTO chunks (pdf_id, chunk_index, page_number, text, token_count)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    pdf_id,
                    ch.get("chunk_index", 0),
                    ch.get("page_number", 1),
                    ch.get("text", ""),
                    ch.get("token_count", 0)
                ))
                chunk_ids.append(cur.lastrowid)

            cur.execute("UPDATE pdfs SET total_chunks = ? WHERE id = ?", (len(chunks), pdf_id))
        return chunk_ids

    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieves all chunks joined with their source document filenames."""
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT c.id, c.pdf_id, c.chunk_index, c.page_number, c.text, c.token_count,
                       p.filename
                FROM chunks c
                JOIN pdfs p ON c.pdf_id = p.id
                ORDER BY c.id ASC
            """)
            return [dict(row) for row in cur.fetchall()]

    def get_by_ids(self, chunk_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Retrieves multiple chunks by primary key IDs as a lookup dictionary."""
        if not chunk_ids:
            return {}

        with self.db.cursor() as cur:
            placeholders = ",".join(["?"] * len(chunk_ids))
            cur.execute(f"""
                SELECT c.id, c.pdf_id, c.chunk_index, c.page_number, c.text, c.token_count,
                       p.filename
                FROM chunks c
                JOIN pdfs p ON c.pdf_id = p.id
                WHERE c.id IN ({placeholders})
            """, chunk_ids)
            return {row["id"]: dict(row) for row in cur.fetchall()}

    def get_by_pdf_id(self, pdf_id: int) -> List[Dict[str, Any]]:
        """Retrieves all chunks belonging to a specific PDF ID."""
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT id, pdf_id, chunk_index, page_number, text, token_count
                FROM chunks
                WHERE pdf_id = ?
                ORDER BY chunk_index ASC
            """, (pdf_id,))
            return [dict(row) for row in cur.fetchall()]
