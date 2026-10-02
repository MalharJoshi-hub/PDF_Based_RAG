import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "pdfs.db"

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db(db_path: Path = DB_PATH):
    """Initializes the database schema with pdfs and chunks tables and migrates existing schema."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pdfs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                total_pages INTEGER DEFAULT 1,
                total_chunks INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Ensure any missing columns in pdfs table from older versions are added
        cursor.execute("PRAGMA table_info(pdfs)")
        existing_cols = {col["name"] for col in cursor.fetchall()}
        if "total_pages" not in existing_cols:
            cursor.execute("ALTER TABLE pdfs ADD COLUMN total_pages INTEGER DEFAULT 1")
        if "total_chunks" not in existing_cols:
            cursor.execute("ALTER TABLE pdfs ADD COLUMN total_chunks INTEGER DEFAULT 0")
        if "created_at" not in existing_cols:
            cursor.execute("ALTER TABLE pdfs ADD COLUMN created_at TIMESTAMP DEFAULT NULL")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pdf_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                page_number INTEGER DEFAULT 1,
                text TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                FOREIGN KEY (pdf_id) REFERENCES pdfs(id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_pdf_id ON chunks(pdf_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_id ON chunks(id);")
        conn.commit()

def migrate_existing_data(db_path: Path = DB_PATH, chunk_func=None):
    """
    If existing 'pdfs' table has old columns (content) and 'chunks' is empty,
    re-chunk and populate the chunks table.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        # Check column names of pdfs
        cursor.execute("PRAGMA table_info(pdfs)")
        cols = [col["name"] for col in cursor.fetchall()]
        
        # Check if chunks table has data
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        
        if "content" in cols and chunk_count == 0:
            cursor.execute("SELECT id, filename, content FROM pdfs")
            rows = cursor.fetchall()
            for row in rows:
                pdf_id = row["id"]
                content = row["content"] or ""
                if not content.strip():
                    continue
                chunks = chunk_func(content) if chunk_func else []
                total_chunks = len(chunks)
                for ch in chunks:
                    cursor.execute("""
                        INSERT INTO chunks (pdf_id, chunk_index, page_number, text, token_count)
                        VALUES (?, ?, ?, ?, ?)
                    """, (pdf_id, ch["chunk_index"], ch.get("page_number", 1), ch["text"], ch.get("token_count", len(ch["text"].split()))))
                cursor.execute("""
                    UPDATE pdfs SET total_chunks = ? WHERE id = ?
                """, (total_chunks, pdf_id))
            conn.commit()

def insert_pdf(filename: str, total_pages: int, total_chunks: int, db_path: Path = DB_PATH) -> int:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO pdfs (filename, total_pages, total_chunks)
            VALUES (?, ?, ?)
        """, (filename, total_pages, total_chunks))
        conn.commit()
        return cursor.lastrowid

def insert_chunks(pdf_id: int, chunks: List[Dict[str, Any]], db_path: Path = DB_PATH) -> List[int]:
    chunk_ids = []
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        for ch in chunks:
            cursor.execute("""
                INSERT INTO chunks (pdf_id, chunk_index, page_number, text, token_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                pdf_id,
                ch.get("chunk_index", 0),
                ch.get("page_number", 1),
                ch.get("text", ""),
                ch.get("token_count", 0)
            ))
            chunk_ids.append(cursor.lastrowid)
        
        cursor.execute("UPDATE pdfs SET total_chunks = ? WHERE id = ?", (len(chunks), pdf_id))
        conn.commit()
    return chunk_ids

def get_all_pdfs(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.id, p.filename, p.total_pages, p.total_chunks, p.created_at,
                   COUNT(c.id) AS active_chunks
            FROM pdfs p
            LEFT JOIN chunks c ON p.id = c.pdf_id
            GROUP BY p.id
            ORDER BY p.id DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

def get_pdf(pdf_id: int, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pdfs WHERE id = ?", (pdf_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def delete_pdf(pdf_id: int, db_path: Path = DB_PATH) -> bool:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks WHERE pdf_id = ?", (pdf_id,))
        cursor.execute("DELETE FROM pdfs WHERE id = ?", (pdf_id,))
        conn.commit()
        return cursor.rowcount > 0

def get_all_chunks(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.id, c.pdf_id, c.chunk_index, c.page_number, c.text, c.token_count,
                   p.filename
            FROM chunks c
            JOIN pdfs p ON c.pdf_id = p.id
            ORDER BY c.id ASC
        """)
        return [dict(row) for row in cursor.fetchall()]

def get_chunks_by_ids(chunk_ids: List[int], db_path: Path = DB_PATH) -> Dict[int, Dict[str, Any]]:
    if not chunk_ids:
        return {}
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        placeholders = ",".join(["?"] * len(chunk_ids))
        cursor.execute(f"""
            SELECT c.id, c.pdf_id, c.chunk_index, c.page_number, c.text, c.token_count,
                   p.filename
            FROM chunks c
            JOIN pdfs p ON c.pdf_id = p.id
            WHERE c.id IN ({placeholders})
        """, chunk_ids)
        return {row["id"]: dict(row) for row in cursor.fetchall()}
