"""
Document loading for DKV Belgium insurance documents.

Supports: PDF (digital + OCR fallback), plain text, Word (.docx).
Returns a list of document dicts ready for chunking.
"""

from __future__ import annotations

import re
from pathlib import Path


def load_text_file(path: Path) -> dict:
    """Load a plain text file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "title": path.stem,
        "content": text,
        "source_file": str(path),
        "format": "txt",
        "language": _detect_language(text[:500]),
    }


def load_pdf(path: Path) -> dict:
    """
    Load a PDF file. Tries pdfplumber first (better for structured PDFs),
    falls back to pypdf for simpler extraction.
    """
    text = _extract_pdf_pdfplumber(path) or _extract_pdf_pypdf(path)
    text = _clean_pdf_text(text)
    return {
        "title": path.stem,
        "content": text,
        "source_file": str(path),
        "format": "pdf",
        "language": _detect_language(text[:500]),
    }


def load_docx(path: Path) -> dict:
    """Load a Word document (.docx)."""
    try:
        import docx
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")

    doc = docx.Document(path)
    text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return {
        "title": path.stem,
        "content": text,
        "source_file": str(path),
        "format": "docx",
        "language": _detect_language(text[:500]),
    }


def load_documents(directory: str | Path, extensions: list[str] | None = None) -> list[dict]:
    """
    Recursively load all supported documents from a directory.

    Args:
        directory: path to the folder containing DKV documents
        extensions: list of file extensions to load (default: pdf, txt, docx)

    Returns:
        list of document dicts with keys: title, content, source_file, format, language
    """
    directory = Path(directory)
    if extensions is None:
        extensions = [".pdf", ".txt", ".docx"]

    loaders = {".pdf": load_pdf, ".txt": load_text_file, ".docx": load_docx}

    # TODO: For each extension, glob with directory.glob(f"**/*{ext}") (recursive).
    # Skip files starting with "." or named "README.txt".
    # Call the matching loader, skip docs with empty content.
    # After loading, enrich each doc with:
    #   "subfolder": path.parent.name if path.parent != directory else ""
    # Use the subfolder prefix (first 2 chars, e.g. "EN", "FR", "NL") to set
    # doc["language"] more reliably than the heuristic — if the prefix matches.
    # Print each loaded file as: "[LANG] subfolder/filename (N chars)"
    # Wrap each loader call in try/except and print a warning on failure.
    raise NotImplementedError


# ── private helpers ────────────────────────────────────────────────────────

def _extract_pdf_pdfplumber(path: Path) -> str | None:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text(layout=True) or ""
                if text.strip():
                    pages.append(text)
            return "\n\n".join(pages) if pages else None
    except ImportError:
        return None
    except Exception:
        return None


def _extract_pdf_pypdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p for p in pages if p.strip())
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")


def _clean_pdf_text(text: str) -> str:
    """Remove common PDF artefacts: repeated headers/footers, excessive whitespace."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"-\n(\w)", r"\1", text)
    return text.strip()


def _detect_language(sample: str) -> str:
    """
    Heuristic language detection based on stopword presence.
    Returns "fr", "nl", "en", or "unknown".
    """
    sample_lower = sample.lower()

    fr_words = {"les", "des", "est", "une", "pour", "dans", "que", "pas", "sur", "par"}
    nl_words = {"de", "het", "een", "van", "voor", "zijn", "met", "aan", "bij", "wordt"}
    en_words = {"the", "and", "for", "are", "this", "that", "with", "from", "have", "will"}

    # TODO: For each language, count how many of its stopwords appear in sample_lower.
    # Wrap each word with spaces (" word ") to avoid partial matches.
    # Return the language key with the highest count, or "unknown" if all scores are 0.
    # Hint: write a small score() helper that sums the membership checks.
    raise NotImplementedError
