"""
Unit tests for MigrationManager module.
"""

from src.persistence.database import DatabaseManager
from src.persistence.migrations import MigrationManager


def test_migration_manager_initialize_schema(tmp_path):
    db_file = tmp_path / "fresh.db"
    db_mgr = DatabaseManager(db_path=db_file)
    migration = MigrationManager(db_mgr)

    migration.initialize_schema()

    with db_mgr.cursor() as cur:
        # Check pdfs table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pdfs';")
        assert cur.fetchone() is not None

        # Check chunks table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks';")
        assert cur.fetchone() is not None

        # Check query_logs table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='query_logs';")
        assert cur.fetchone() is not None


def test_migration_legacy_content(tmp_path):
    db_file = tmp_path / "legacy.db"
    db_mgr = DatabaseManager(db_path=db_file)

    # Simulate old table with content column
    with db_mgr.transaction() as cur:
        cur.execute("""
            CREATE TABLE pdfs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content TEXT
            );
        """)
        cur.execute("INSERT INTO pdfs (filename, content) VALUES (?, ?)", ("old.pdf", "Legacy content text for testing."))

    migration = MigrationManager(db_mgr)

    def simple_chunker(text):
        return [{"chunk_index": 0, "page_number": 1, "text": text, "token_count": len(text.split())}]

    migration.migrate_legacy_content(chunk_func=simple_chunker)

    with db_mgr.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks;")
        assert cur.fetchone()[0] == 1
