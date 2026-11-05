"""Module 3 - Text normalization.

Case folding, stop-word removal and stemming. Both indexing and querying use
the same Normalizer so that words are always compared in the same form.
"""
from __future__ import annotations

from typing import Optional

STOP_WORDS = frozenset("""a an the and or not of to in on at by for with from as is are
was were be been being it its this that these those i you he she we they them his her
their our your my me do does did has have had but if then so than too very can will just
into over under about up out no nor""".split())


def _light_stem(word: str) -> str:
    """Small suffix-stripping stemmer used when NLTK is not installed."""
    if len(word) <= 3 or word.endswith("ss"):
        return word
    for suffix in ("ingly", "edly", "ing", "ed", "ly", "ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            root = word[: -len(suffix)]
            return root + "y" if suffix == "ies" else root
    return word


try:  # the real Porter stemmer gives the best word conflation
    from nltk.stem import PorterStemmer

    _stem_word = PorterStemmer().stem
    STEMMER_NAME = "Porter stemmer (NLTK)"
except Exception:  # pragma: no cover - depends on environment
    _stem_word = _light_stem
    STEMMER_NAME = "built-in light stemmer"


class Normalizer:
    def __init__(self, remove_stop_words: bool = True, use_stemming: bool = True,
                 stop_words=STOP_WORDS):
        self.remove_stop_words = remove_stop_words
        self.use_stemming = use_stemming
        self.stop_words = frozenset(stop_words)

    def fold(self, word: str) -> str:
        """Case folding: 'Cats' -> 'cats'."""
        return word.casefold()

    def is_stop_word(self, folded_word: str) -> bool:
        return self.remove_stop_words and folded_word in self.stop_words

    def stem(self, folded_word: str) -> str:
        return _stem_word(folded_word) if self.use_stemming else folded_word

    def normalize(self, word: str) -> Optional[str]:
        """Full pipeline for one word. Returns None for a stop-word."""
        folded = self.fold(word)
        if self.is_stop_word(folded):
            return None
        return self.stem(folded)
