"""
Database connection management and transaction contexts for SQLite.
"""

import sqlite3
from pathlib import Path
from typing import Generator, Optional, Union
from contextlib import contextmanager
from src.config import config


class DatabaseManager:
    """Manages SQLite connections with thread safety, foreign keys, and WAL mode."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None, timeout: float = 30.0):
        self.db_path = Path(db_path) if db_path else config.database.db_path
        self.timeout = timeout
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Creates a configured SQLite connection with row factory and foreign keys."""
        conn = sqlite3.connect(str(self.db_path), timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        if config.database.enable_wal_mode:
            try:
                conn.execute("PRAGMA journal_mode = WAL;")
            except Exception:
                pass
        return conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager providing an atomic transaction block with auto-commit/rollback."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager providing a read-only or single-query cursor."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            yield cursor
        finally:
            conn.close()
