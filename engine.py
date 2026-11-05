"""Facade that wires all modules together: load -> tokenize -> normalize -> index -> search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .document_loader import Document, load_documents, split_into_documents
from .indexer import IndexBuilder, InvertedIndex
from .kgram import KGramIndex
from .normalizer import Normalizer
from .phonetic import PhoneticIndex
from .query_processor import QueryProcessor, QueryResult
from .spelling import SpellingCorrector
from .tokenizer import tokenize
from .wildcard import WildcardMatcher


@dataclass(frozen=True)
class IndexStats:
    documents: int
    vocabulary: int  # distinct words as written (case-folded, stop-words removed)
    terms: int       # distinct index terms (after stemming)


class SearchEngine:
    def __init__(self, normalizer: Optional[Normalizer] = None):
        self.normalizer = normalizer or Normalizer()
        self.documents: List[Document] = []
        self.index: Optional[InvertedIndex] = None
        self.stats: Optional[IndexStats] = None
        self._by_id: Dict[int, Document] = {}
        self._processor: Optional[QueryProcessor] = None

    # ------------------------------------------------------------ loading
    def load_file(self, path) -> IndexStats:
        return self._build(load_documents(path))

    def load_text(self, text: str) -> IndexStats:
        documents = split_into_documents(text)
        if not documents:
            raise ValueError("The text is empty.")
        return self._build(documents)

    def _build(self, documents: List[Document]) -> IndexStats:
        index = IndexBuilder(self.normalizer).build(documents)
        kgram, phonetic = KGramIndex(), PhoneticIndex()
        for word in index.surface_words():
            kgram.add(word)
            phonetic.add(word)
        wildcard = WildcardMatcher(kgram, index.surface_words())
        speller = SpellingCorrector(kgram, lambda w: index.df(index.term_for_surface(w)))

        self.documents = documents
        self._by_id = {d.doc_id: d for d in documents}
        self.index = index
        self._processor = QueryProcessor(index, self.normalizer, wildcard, phonetic, speller)
        self.stats = IndexStats(len(documents), len(index.surface_words()), len(index.terms()))
        return self.stats

    # ----------------------------------------------------------- searching
    @property
    def is_ready(self) -> bool:
        return self._processor is not None

    def search(self, query: str) -> QueryResult:
        if not self.is_ready:
            raise RuntimeError("No file loaded. Load a .txt file first.")
        return self._processor.search(query)

    def document(self, doc_id: int) -> Document:
        return self._by_id[doc_id]

    def highlight_spans(self, doc_id: int, matched_terms: Set[str]) -> List[Tuple[int, int]]:
        """Character ranges in a document whose words matched the query."""
        spans = []
        for token in tokenize(self.document(doc_id).text):
            term = self.normalizer.normalize(token.text)
            if term is not None and term in matched_terms:
                spans.append((token.start, token.end))
        return spans
