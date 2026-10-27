"""
Enterprise PDF RAG Application Core Package.
Provides multi-document retrieval-augmented generation with FAISS, BM25, Neural Reranking, and Generative QA.
"""

__version__ = "2.4.0"
__author__ = "Enterprise AI Engineering Team"

from src.config import config, AppConfig

__all__ = ["config", "AppConfig", "__version__"]
