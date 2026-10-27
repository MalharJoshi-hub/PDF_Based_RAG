"""
Unit tests for RAGPipeline end-to-end question answering workflow.
"""

from unittest.mock import MagicMock
from src.generation.pipeline import RAGPipeline
from src.retrieval.retriever_manager import RetrieverManager
from src.embeddings.embedder import Embedder
from src.vector_store.index_manager import IndexManager


def test_rag_pipeline_process_query(mock_embedding_provider, sample_chunks):
    embedder = Embedder(provider=mock_embedding_provider)
    idx_mgr = IndexManager(embedder=embedder)
    retriever_mgr = RetrieverManager(embedder=embedder, index_manager=idx_mgr)
    retriever_mgr.rebuild_all(sample_chunks)

    mock_generator = MagicMock()
    mock_generator.generate_answer.return_value = "Hibernate simplifies relational database persistence in Java."

    pipeline = RAGPipeline(
        retriever_manager=retriever_mgr,
        generator=mock_generator
    )

    lookup = {c["id"]: c for c in sample_chunks}
    res = pipeline.process_query(
        query="What is Hibernate?",
        retriever_method="dense",
        top_k_retrieved=2,
        top_k_rerank=2,
        use_reranker=False,
        chunks_lookup=lookup
    )

    assert res["query"] == "What is Hibernate?"
    assert res["retriever"] == "dense"
    assert "AI Summary:" in res["answer"]
    assert "Detailed Extracted Context:" in res["answer"]
    assert len(res["citations"]) <= 2
    assert "timings" in res
    assert "total_ms" in res["timings"]


def test_rag_pipeline_no_results(mock_embedding_provider):
    embedder = Embedder(provider=mock_embedding_provider)
    idx_mgr = IndexManager(embedder=embedder)
    retriever_mgr = RetrieverManager(embedder=embedder, index_manager=idx_mgr)
    retriever_mgr.rebuild_all([])

    pipeline = RAGPipeline(retriever_manager=retriever_mgr)
    res = pipeline.process_query("Any question", chunks_lookup={})
    assert "No relevant information found" in res["answer"]
    assert len(res["citations"]) == 0
