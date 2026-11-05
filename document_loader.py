"""Module 1 - Document loading.

Reads a .txt file from disk and splits it into searchable documents.
A document is a blank-line separated paragraph; if the file has no blank
lines, every non-empty line becomes one document.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Document:
    doc_id: int  # 1-based identifier shown to the user
    text: str


def read_text_file(path) -> str:
    """Return the file's text, trying UTF-8 first and falling back to Latin-1."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    raw = p.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def split_into_documents(text: str) -> List[Document]:
    """Split raw text into Document objects (paragraphs, else lines)."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) <= 1:
        parts = [line.strip() for line in text.splitlines() if line.strip()]
    return [Document(i, part) for i, part in enumerate(parts, start=1)]


def load_documents(path) -> List[Document]:
    """Read a file and return its documents."""
    documents = split_into_documents(read_text_file(path))
    if not documents:
        raise ValueError("The file contains no text.")
    return documents
