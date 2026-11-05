"""TF-IDF ranking of matching documents."""
from __future__ import annotations

import math
from typing import Iterable, List, Tuple

from .indexer import InvertedIndex


def idf(index: InvertedIndex, term: str) -> float:
    df = index.df(term)
    return math.log10(1 + index.doc_count / df) if df else 0.0


def score_document(index: InvertedIndex, doc_id: int, terms: Iterable[str]) -> float:
    """score(d) = sum over terms of (1 + log10 tf) * log10(1 + N / df)."""
    total = 0.0
    for term in terms:
        tf = index.tf(term, doc_id)
        if tf:
            total += (1 + math.log10(tf)) * idf(index, term)
    return total


def rank(index: InvertedIndex, doc_ids: Iterable[int], terms: Iterable[str]) -> List[Tuple[int, float]]:
    terms = list(terms)
    scored = [(doc_id, score_document(index, doc_id, terms)) for doc_id in doc_ids]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored
