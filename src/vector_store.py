"""
ChromaDB vector store helpers.
"""

from __future__ import annotations

import time
from pathlib import Path

import chromadb
from tqdm import tqdm

from .embedder import DEFAULT_MODEL, embed_query, embed_texts

DEFAULT_COLLECTION = "workshop_rag"
DEFAULT_CHROMA_PATH = str(Path(__file__).parent.parent / "chroma_db")
BATCH_SIZE = 500


def get_collection(
    collection_name: str = DEFAULT_COLLECTION,
    persist_path: str = DEFAULT_CHROMA_PATH,
) -> chromadb.Collection:
    """Open or create a persisted ChromaDB collection with cosine similarity."""
    client = chromadb.PersistentClient(path=persist_path)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(
    collection: chromadb.Collection,
    chunks: list[dict],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = BATCH_SIZE,
    force: bool = False,
) -> None:
    """
    Embed and index chunk records into a ChromaDB collection.

    Skips indexing if the collection already has >= len(chunks) documents,
    unless force=True.

    Each chunk dict must have: "text", "source", "id"
    """
    if not force and collection.count() >= len(chunks):
        print(f"Collection '{collection.name}' already has {collection.count()} docs — skipping.")
        return

    print(f"Indexing {len(chunks)} chunks into '{collection.name}'...")
    start = time.time()

    # TODO: Deduplicate chunks by id first (the dataset may contain duplicate documents).
    # Then iterate over unique chunks in batches of batch_size (use tqdm for a progress bar).
    # For each batch, extract lists of texts, ids, and metadata dicts ({"source": ...}).
    # Call embed_texts(texts, model_name=model_name) to get the embeddings.
    # Then call collection.upsert(documents=..., embeddings=..., ids=..., metadatas=...).
    # (upsert instead of add so re-running the cell is safe)
    # After the loop, print how long it took and the total doc count.
    raise NotImplementedError


def retrieve(
    collection: chromadb.Collection,
    question: str,
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL,
    source_filter: str | None = None,
    min_similarity: float = 0.0,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a question.

    Returns list of dicts: {"text", "source", "similarity"}
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
        {"text": text, "source": meta["source"], "similarity": 1.0 - dist}
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
    return [c for c in chunks if c["similarity"] >= min_similarity]
