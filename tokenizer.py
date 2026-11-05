"""Module 2 - Tokenization.

Breaks text into word tokens. Tokens keep their ORIGINAL case; case folding
is the job of the normalizer module. Each token records its word position
(used by phrase search) and character span (used for highlighting).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

# letters/digits (any language), optionally with an apostrophe part: don't, it's
_TOKEN_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)?")


@dataclass(frozen=True)
class Token:
    text: str
    position: int  # 0-based word index inside the document
    start: int     # character offset where the token starts
    end: int       # character offset where the token ends


def tokenize(text: str) -> Iterator[Token]:
    for position, match in enumerate(_TOKEN_RE.finditer(text)):
        yield Token(match.group(), position, match.start(), match.end())
