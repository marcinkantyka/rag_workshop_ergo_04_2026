"""
Embedding helpers wrapping sentence-transformers.

Default model is multilingual — required for DKV Belgium documents
in French, Dutch, and English.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

# Default: multilingual, 768-dim, covers FR/NL/EN and 50+ languages (~500 MB)
DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Faster multilingual option, 384-dim (~250 MB) — good for low-RAM environments
FAST_MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# English-only, fast, 384-dim — keep for reference / ablation experiments
ENGLISH_FAST_MODEL = "all-MiniLM-L6-v2"


_cache: dict[str, SentenceTransformer] = {}


def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Return a cached SentenceTransformer instance."""
    if model_name not in _cache:
        _cache[model_name] = SentenceTransformer(model_name)
    return _cache[model_name]


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    show_progress: bool = False,
) -> list[list[float]]:
    """
    Embed a list of texts. Returns a list of float lists (not numpy arrays).

    Args:
        texts: strings to embed
        model_name: sentence-transformers model identifier
        batch_size: encoding batch size
        show_progress: show tqdm progress bar during encoding
    """
    # TODO: Call get_model(model_name), then use model.encode() with the provided
    # batch_size and show_progress_bar arguments. Set convert_to_numpy=True.
    # Return the result as a Python list (call .tolist() on the numpy array).
    raise NotImplementedError


def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> list[float]:
    """Embed a single query string. Returns a flat list of floats."""
    model = get_model(model_name)
    return model.encode(query, convert_to_numpy=True).tolist()
