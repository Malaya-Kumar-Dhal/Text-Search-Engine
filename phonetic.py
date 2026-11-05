"""Soundex phonetic coding and a code -> words index."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Set, Tuple

_CODES = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
          **dict.fromkeys("dt", "3"), "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}


def soundex(word: str) -> str:
    """Return the 4-character Soundex code ('' if the word has no letters)."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return ""
    result, previous = word[0].upper(), _CODES.get(word[0], "")
    for ch in word[1:]:
        code = _CODES.get(ch, "")
        if code and code != previous:
            result += code
        if ch not in "hw":  # h and w do not separate equal codes
            previous = code
    return (result + "000")[:4]


class PhoneticIndex:
    def __init__(self):
        self._by_code: Dict[str, Set[str]] = defaultdict(set)

    def add(self, word: str) -> None:
        code = soundex(word)
        if code:
            self._by_code[code].add(word)

    def lookup(self, word: str) -> Tuple[str, Set[str]]:
        code = soundex(word)
        return code, set(self._by_code.get(code, ()))
