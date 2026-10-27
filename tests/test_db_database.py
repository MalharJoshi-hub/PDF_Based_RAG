"""
Unit tests for DatabaseManager connection handling and transaction rollback.
"""

import pytest
from src.persistence.database import DatabaseManager


def test_database_connection(temp_db):
    conn = temp_db.get_connection()
    assert conn is not None
    # Check foreign keys enabled
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys;")
    assert cur.fetchone()[0] == 1
    conn.close()


def test_database_transaction_commit(temp_db):
    with temp_db.transaction() as cur:
        cur.execute("INSERT INTO pdfs (filename, total_pages, total_chunks) VALUES (?, ?, ?)", ("file1.pdf", 3, 5))
        last_id = cur.lastrowid

    with temp_db.cursor() as cur:
        cur.execute("SELECT filename FROM pdfs WHERE id = ?", (last_id,))
        assert cur.fetchone()[0] == "file1.pdf"


def test_database_transaction_rollback(temp_db):
    try:
        with temp_db.transaction() as cur:
            cur.execute("INSERT INTO pdfs (filename, total_pages, total_chunks) VALUES (?, ?, ?)", ("rollback.pdf", 1, 1))
            raise RuntimeError("Forced rollback")
    except RuntimeError:
        pass

    with temp_db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM pdfs WHERE filename = ?", ("rollback.pdf",))
        assert cur.fetchone()[0] == 0
