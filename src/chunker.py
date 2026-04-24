"""
Document chunking strategies using LangChain text splitters.

Three production-ready approaches:
  chunk_recursive   — RecursiveCharacterTextSplitter (recommended default)
  chunk_fixed_size  — CharacterTextSplitter (simple baseline)
  chunk_by_tokens   — SentenceTransformersTokenTextSplitter (token-aware)
"""

from __future__ import annotations

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

DEFAULT_EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"


def chunk_recursive(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """
    Split text using RecursiveCharacterTextSplitter (recommended default).

    Tries separators in order: paragraph → newline → sentence → word → character.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def chunk_fixed_size(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """
    Split text using CharacterTextSplitter (simple fixed-size baseline).

    Splits on spaces — may cut in the middle of a sentence.
    """
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(text)


def chunk_by_tokens(
    text: str,
    chunk_size: int = 100,
    overlap: int = 10,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> list[str]:
    """
    Token-aware chunking using the embedding model's own tokenizer.

    Prevents silent truncation: the multilingual model caps at 128 tokens.
    """
    splitter = SentenceTransformersTokenTextSplitter(
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(text)


def build_chunk_records(
    articles: list[dict],
    strategy: str = "recursive",
    **kwargs,
) -> list[dict]:
    """
    Convert a list of article dicts into chunk records with metadata.

    Each record: {"text": str, "source": str, "id": str}

    Args:
        articles: list of {"title": str, "content": str, ...}
        strategy: "recursive" | "fixed" | "tokens"
        **kwargs: forwarded to the chosen chunking function
    """
    fn = {
        "recursive": chunk_recursive,
        "fixed": chunk_fixed_size,
        "tokens": chunk_by_tokens,
    }[strategy]

    # TODO: For each article, call fn(article["content"], **kwargs) to get chunks.
    # Build source as: f"{subfolder}/{title}" if article has a "subfolder" key, else just title.
    # Wrap each chunk in a dict with keys "text", "source", and "id".
    # Use f"{source}_{i}".replace("/", "_") as the id (i is the chunk index).
    # Collect all records into a list and return it.
    raise NotImplementedError
