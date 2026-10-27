"""
Generation package providing Flan-T5 synthesis, prompt engineering, and end-to-end RAG pipeline.
"""

from src.generation.base import BaseGenerator, Citation, GenerationResponse
from src.generation.generator import FlanT5Generator
from src.generation.prompt_templates import PromptBuilder
from src.generation.citation_manager import CitationManager
from src.generation.pipeline import RAGPipeline

__all__ = [
    "BaseGenerator",
    "Citation",
    "GenerationResponse",
    "FlanT5Generator",
    "PromptBuilder",
    "CitationManager",
    "RAGPipeline"
]
