"""
Schema initialization and database migration management.
"""

from typing import Optional, Callable
from src.persistence.database import DatabaseManager


class MigrationManager:
    """Manages table creation, schema alters, and legacy data migration."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def initialize_schema(self) -> None:
        """Creates tables, indices, and adds missing columns if schema evolves."""
        with self.db.transaction() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdfs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    total_pages INTEGER DEFAULT 1,
                    total_chunks INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("PRAGMA table_info(pdfs);")
            existing_cols = {col["name"] for col in cur.fetchall()}
            if "total_pages" not in existing_cols:
                cur.execute("ALTER TABLE pdfs ADD COLUMN total_pages INTEGER DEFAULT 1;")
            if "total_chunks" not in existing_cols:
                cur.execute("ALTER TABLE pdfs ADD COLUMN total_chunks INTEGER DEFAULT 0;")
            if "created_at" not in existing_cols:
                cur.execute("ALTER TABLE pdfs ADD COLUMN created_at TIMESTAMP DEFAULT NULL;")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER DEFAULT 1,
                    text TEXT NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    FOREIGN KEY (pdf_id) REFERENCES pdfs(id) ON DELETE CASCADE
                );
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_pdf_id ON chunks(pdf_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_id ON chunks(id);")

            # Query audit logging table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    retriever_method TEXT NOT NULL,
                    latency_ms REAL,
                    top_k INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def migrate_legacy_content(self, chunk_func: Optional[Callable] = None) -> None:
        """Migrates legacy single-content column in pdfs to normalized chunks table."""
        self.initialize_schema()
        with self.db.transaction() as cur:
            cur.execute("PRAGMA table_info(pdfs);")
            cols = [col["name"] for col in cur.fetchall()]

            cur.execute("SELECT COUNT(*) FROM chunks;")
            chunk_count = cur.fetchone()[0]

            if "content" in cols and chunk_count == 0:
                cur.execute("SELECT id, filename, content FROM pdfs;")
                rows = cur.fetchall()
                for row in rows:
                    pdf_id = row["id"]
                    content = row["content"] or ""
                    if not content.strip():
                        continue

                    chunks = chunk_func(content) if chunk_func else []
                    for ch in chunks:
                        cur.execute("""
                            INSERT INTO chunks (pdf_id, chunk_index, page_number, text, token_count)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            pdf_id,
                            ch.get("chunk_index", 0),
                            ch.get("page_number", 1),
                            ch.get("text", ""),
                            ch.get("token_count", len(ch["text"].split()))
                        ))
                    cur.execute("UPDATE pdfs SET total_chunks = ? WHERE id = ?;", (len(chunks), pdf_id))
