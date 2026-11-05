"""Module 4 - Inverted index construction.

InvertedIndex stores, for every normalized term, the documents containing it
and the word positions inside each document (a positional inverted index).
IndexBuilder fills it by running documents through the tokenizer and normalizer.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Set

from .document_loader import Document
from .normalizer import Normalizer
from .tokenizer import tokenize


class InvertedIndex:
    def __init__(self):
        self._postings: Dict[str, Dict[int, List[int]]] = {}
        self._surface: Dict[str, str] = {}  # case-folded word as written -> index term
        self._doc_ids: Set[int] = set()

    # ---- building ----
    def add_document(self, doc_id: int) -> None:
        self._doc_ids.add(doc_id)

    def add_posting(self, term: str, doc_id: int, position: int) -> None:
        self._postings.setdefault(term, {}).setdefault(doc_id, []).append(position)

    def register_surface(self, word: str, term: str) -> None:
        self._surface[word] = term

    # ---- reading ----
    @property
    def doc_count(self) -> int:
        return len(self._doc_ids)

    @property
    def all_doc_ids(self) -> Set[int]:
        return set(self._doc_ids)

    def __contains__(self, term: str) -> bool:
        return term in self._postings

    def terms(self) -> List[str]:
        return sorted(self._postings)

    def surface_words(self) -> List[str]:
        return list(self._surface)

    def term_for_surface(self, word: str) -> str:
        return self._surface[word]

    def postings(self, term: str) -> Dict[int, List[int]]:
        return self._postings.get(term, {})

    def doc_ids(self, term: str) -> Set[int]:
        return set(self._postings.get(term, {}))

    def positions(self, term: str, doc_id: int) -> List[int]:
        return self._postings.get(term, {}).get(doc_id, [])

    def df(self, term: str) -> int:
        """Document frequency: number of documents containing the term."""
        return len(self._postings.get(term, {}))

    def tf(self, term: str, doc_id: int) -> int:
        """Term frequency: occurrences of the term in one document."""
        return len(self.positions(term, doc_id))


class IndexBuilder:
    def __init__(self, normalizer: Normalizer):
        self.normalizer = normalizer

    def build(self, documents: Iterable[Document]) -> InvertedIndex:
        index = InvertedIndex()
        for doc in documents:
            index.add_document(doc.doc_id)
            for token in tokenize(doc.text):
                term = self.normalizer.normalize(token.text)
                if term is None:  # stop-word: skipped, but its position is still counted
                    continue
                index.add_posting(term, doc.doc_id, token.position)
                index.register_surface(self.normalizer.fold(token.text), term)
        return index
