"""
Retrieval strategies: pure vector, BM25 keyword, and hybrid (RRF fusion).

Hybrid search is particularly valuable for multilingual insurance documents
where exact technical terms (RIZIV, NIHDI, INAMI) matter alongside semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .embedder import embed_query, DEFAULT_MODEL

if TYPE_CHECKING:
    import chromadb


def vector_retrieve(
    collection: "chromadb.Collection",
    question: str,
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL,
    source_filter: str | None = None,
    min_similarity: float = 0.0,
) -> list[dict]:
    """
    Pure vector (semantic) retrieval.

    Returns list of: {"text", "source", "similarity", "retrieval_method"}
    """
    q_emb = embed_query(question, model_name=model_name)
    kwargs: dict = dict(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if source_filter:
        kwargs["where"] = {"source": source_filter}

    results = collection.query(**kwargs)
    chunks = [
        {
            "text": text,
            "source": meta["source"],
            "similarity": 1.0 - dist,
            "retrieval_method": "vector",
        }
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
    return [c for c in chunks if c["similarity"] >= min_similarity]


def bm25_retrieve(
    chunks: list[dict],
    question: str,
    top_k: int = 5,
) -> list[dict]:
    """
    BM25 keyword retrieval over an in-memory list of chunk dicts.

    Note: operates on the full chunk list, not ChromaDB.
    Best used as the keyword arm of hybrid retrieval.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("Install rank-bm25: pip install rank-bm25")

    tokenised_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenised_corpus)
    scores = bm25.get_scores(question.lower().split())

    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)[:top_k]
    return [
        {**chunk, "bm25_score": float(score), "retrieval_method": "bm25"}
        for score, chunk in scored
        if score > 0
    ]


def hybrid_retrieve(
    collection: "chromadb.Collection",
    all_chunks: list[dict],
    question: str,
    top_k: int = 5,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    initial_k: int = 20,
    model_name: str = DEFAULT_MODEL,
) -> list[dict]:
    """
    Hybrid retrieval: combines vector and BM25 scores using Reciprocal Rank Fusion (RRF).

    RRF score = weight * 1/(rank + 60) — robust to score scale differences.
    """
    vector_results = vector_retrieve(collection, question, top_k=initial_k, model_name=model_name)
    bm25_results = bm25_retrieve(all_chunks, question, top_k=initial_k)

    rrf_scores: dict[str, float] = {}
    metadata: dict[str, dict] = {}
    rrf_k = 60

    for rank, chunk in enumerate(vector_results):
        key = chunk["text"]
        rrf_scores[key] = rrf_scores.get(key, 0) + vector_weight * (1 / (rank + rrf_k))
        metadata[key] = chunk

    for rank, chunk in enumerate(bm25_results):
        key = chunk["text"]
        rrf_scores[key] = rrf_scores.get(key, 0) + bm25_weight * (1 / (rank + rrf_k))
        if key not in metadata:
            metadata[key] = chunk

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {**metadata[text], "rrf_score": score, "retrieval_method": "hybrid"}
        for text, score in ranked
    ]


def rerank(
    question: str,
    candidates: list[dict],
    final_k: int,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[dict]:
    """
    Re-rank candidates using a cross-encoder. High precision, slower.

    Typical usage: over-retrieve 20 candidates, re-rank to top 5.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers")

    ce = CrossEncoder(model_name)
    pairs = [(question, c["text"]) for c in candidates]
    scores = ce.predict(pairs)

    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:final_k]
