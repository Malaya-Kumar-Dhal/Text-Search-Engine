"""Spell correction: k-gram candidate search + edit distance."""
from __future__ import annotations

from typing import Callable, Optional

from .kgram import KGramIndex, kgrams


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: minimum inserts, deletes and replacements."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


class SpellingCorrector:
    def __init__(self, kgram_index: KGramIndex, popularity: Callable[[str], int],
                 min_overlap: float = 0.2):
        self._kgrams = kgram_index
        self._popularity = popularity  # word -> number of documents containing it
        self._min_overlap = min_overlap

    def correct(self, word: str) -> Optional[str]:
        """Best vocabulary word within a small edit distance, or None."""
        grams = kgrams(word, self._kgrams.k)
        limit = max(1, len(word) // 3)
        best = None
        for candidate, shared in self._kgrams.similar_words(word).items():
            union = len(grams) + len(kgrams(candidate, self._kgrams.k)) - shared
            if shared / union < self._min_overlap:
                continue
            distance = edit_distance(word, candidate)
            if distance > limit:
                continue
            key = (distance, -self._popularity(candidate), candidate)
            if best is None or key < best:
                best = key
        return best[2] if best else None
