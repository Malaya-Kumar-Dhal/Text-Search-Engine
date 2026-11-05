"""k-gram (bigram) index shared by wildcard matching and spell correction."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Set


def kgrams(word: str, k: int = 2) -> Set[str]:
    """Overlapping k-letter pieces of '$word$'. kgrams('cat') -> {$c, ca, at, t$}."""
    padded = f"${word}$"
    return {padded[i:i + k] for i in range(len(padded) - k + 1)}


class KGramIndex:
    def __init__(self, k: int = 2):
        self.k = k
        self._index: Dict[str, Set[str]] = defaultdict(set)

    def add(self, word: str) -> None:
        for gram in kgrams(word, self.k):
            self._index[gram].add(word)

    def words_with_gram(self, gram: str) -> Set[str]:
        return self._index.get(gram, set())

    def similar_words(self, word: str) -> Counter:
        """Words sharing at least one k-gram with `word`, with the shared count."""
        shared: Counter = Counter()
        for gram in kgrams(word, self.k):
            for candidate in self._index.get(gram, ()):
                shared[candidate] += 1
        return shared
