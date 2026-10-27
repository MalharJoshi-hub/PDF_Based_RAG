"""
Persistence package handling SQLite connections, models, repositories, and migrations.
"""

from src.persistence.database import DatabaseManager
from src.persistence.models import PDFDocumentRecord, ChunkRecord, QueryAuditRecord
from src.persistence.repository import DocumentRepository, ChunkRepository
from src.persistence.migrations import MigrationManager

__all__ = [
    "DatabaseManager",
    "PDFDocumentRecord",
    "ChunkRecord",
    "QueryAuditRecord",
    "DocumentRepository",
    "ChunkRepository",
    "MigrationManager"
]
