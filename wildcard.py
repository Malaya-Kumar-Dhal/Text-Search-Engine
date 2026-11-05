"""Module 6 - Wildcard processing.

'*' matches any run of letters, '?' matches exactly one letter.
The k-gram index first narrows the vocabulary to words that could match, and
a regular expression then confirms each candidate.
"""
from __future__ import annotations

import re
from typing import Iterable, List

from .kgram import KGramIndex


class WildcardMatcher:
    def __init__(self, kgram_index: KGramIndex, vocabulary: Iterable[str]):
        self._kgrams = kgram_index
        self._vocabulary = set(vocabulary)

    @staticmethod
    def is_wildcard(term: str) -> bool:
        return "*" in term or "?" in term

    @staticmethod
    def _to_regex(pattern: str) -> "re.Pattern":
        parts = (".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern)
        return re.compile("".join(parts))

    def _candidates(self, pattern: str):
        k = self._kgrams.k
        candidates = None
        # Split '$pattern$' at wildcards; each literal piece must appear in the word.
        for piece in re.split(r"[*?]", f"${pattern}$"):
            for i in range(len(piece) - k + 1):
                words = self._kgrams.words_with_gram(piece[i:i + k])
                candidates = set(words) if candidates is None else candidates & words
                if not candidates:
                    return set()
        return self._vocabulary if candidates is None else candidates

    def match(self, pattern: str) -> List[str]:
        """Return the sorted vocabulary words matching the wildcard pattern."""
        pattern = pattern.casefold()
        regex = self._to_regex(pattern)
        return sorted(w for w in self._candidates(pattern) if regex.fullmatch(w))
