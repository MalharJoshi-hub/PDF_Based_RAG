"""
Enterprise PDF RAG Application Configuration Module.
Provides centralized, environment-aware configuration management for all system layers.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


@dataclass
class DatabaseConfig:
    """Configuration for SQLite storage and persistence."""
    db_path: Path = field(default_factory=lambda: DATA_DIR / "pdfs.db")
    timeout_seconds: float = 30.0
    enable_wal_mode: bool = True
    connection_pool_size: int = 5


@dataclass
class ChunkingConfig:
    """Configuration for text chunking algorithms."""
    default_chunk_size: int = 150
    default_overlap: int = 30
    min_chunk_length: int = 20
    header_regex_pattern: str = r"(?:^|\n)(?=[A-Z0-9])([^\n]{2,55}(?<![\,\.\;\-\:]))\n"
    sentence_boundary_regex: str = r"(?<=[.!?])\s+"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models and vector generation."""
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True
    embedding_dimension: int = 384
    cache_enabled: bool = True
    cache_max_size: int = 50000


@dataclass
class FAISSConfig:
    """Configuration for FAISS vector index."""
    index_file: Path = field(default_factory=lambda: DATA_DIR / "faiss.index")
    id_map_file: Path = field(default_factory=lambda: DATA_DIR / "id_map.pkl")
    index_type: str = "IndexFlatIP"
    metric_type: str = "INNER_PRODUCT"


@dataclass
class RerankerConfig:
    """Configuration for Cross-Encoder neural reranking."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    max_length: int = 512
    batch_size: int = 16
    score_threshold: float = -10.0


@dataclass
class GeneratorConfig:
    """Configuration for generative QA synthesis."""
    model_name: str = "google/flan-t5-base"
    device: str = "cpu"
    max_input_length: int = 1024
    max_new_tokens: int = 150
    min_length: int = 20
    num_beams: int = 3
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    temperature: float = 0.2


@dataclass
class AppConfig:
    """Root configuration object containing all sub-configs."""
    app_name: str = "Enterprise PDF RAG System"
    version: str = "2.4.0"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 5000
    upload_folder: Path = field(default_factory=lambda: DATA_DIR / "uploads")
    max_upload_size_mb: int = 64
    allowed_extensions: List[str] = field(default_factory=lambda: [".pdf"])
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    def __post_init__(self):
        """Ensure critical directories exist."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.upload_folder.mkdir(parents=True, exist_ok=True)


# Default singleton instance
config = AppConfig()
