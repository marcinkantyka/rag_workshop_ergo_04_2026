"""
RAG Workshop — reusable helper package for DKV Belgium.

Usage inside notebooks:
    import sys; sys.path.insert(0, "..")

    from src.document_loader import load_documents
    from src.chunker import build_chunk_records
    from src.embedder import DEFAULT_MODEL, FAST_MULTILINGUAL_MODEL
    from src.vector_store import get_collection, index_chunks
    from src.retriever import vector_retrieve, hybrid_retrieve, rerank
    from src.pipeline import RAGPipeline, AdvancedRAGPipeline
    from src.evaluator import run_evaluation, compare_pipelines
    from src.experiment_log import ExperimentLog
"""
