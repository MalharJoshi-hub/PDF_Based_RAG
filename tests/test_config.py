"""
Unit tests for configuration module and sub-configurations.
"""

from pathlib import Path
from src.config import (
    AppConfig,
    DatabaseConfig,
    ChunkingConfig,
    EmbeddingConfig,
    FAISSConfig,
    RerankerConfig,
    GeneratorConfig,
    config
)


def test_app_config_defaults():
    cfg = AppConfig()
    assert cfg.app_name == "Enterprise PDF RAG System"
    assert cfg.version == "2.4.0"
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 5000
    assert ".pdf" in cfg.allowed_extensions
    assert cfg.upload_folder.exists()


def test_database_config_defaults():
    db_cfg = DatabaseConfig()
    assert db_cfg.db_path.name == "pdfs.db"
    assert db_cfg.timeout_seconds == 30.0
    assert db_cfg.enable_wal_mode is True


def test_chunking_config_defaults():
    c_cfg = ChunkingConfig()
    assert c_cfg.default_chunk_size == 150
    assert c_cfg.default_overlap == 30
    assert c_cfg.min_chunk_length == 20


def test_embedding_config_defaults():
    emb_cfg = EmbeddingConfig()
    assert emb_cfg.model_name == "all-MiniLM-L6-v2"
    assert emb_cfg.embedding_dimension == 384
    assert emb_cfg.normalize_embeddings is True
    assert emb_cfg.cache_enabled is True


def test_faiss_config_defaults():
    f_cfg = FAISSConfig()
    assert f_cfg.index_file.name == "faiss.index"
    assert f_cfg.id_map_file.name == "id_map.pkl"
    assert f_cfg.index_type == "IndexFlatIP"


def test_reranker_and_generator_config():
    r_cfg = RerankerConfig()
    assert "cross-encoder" in r_cfg.model_name
    assert r_cfg.batch_size == 16

    g_cfg = GeneratorConfig()
    assert "flan-t5" in g_cfg.model_name
    assert g_cfg.num_beams == 3
    assert g_cfg.repetition_penalty == 1.2


def test_global_config_instance():
    assert config is not None
    assert isinstance(config, AppConfig)
