"""Module 5 - Query processing.

Parses a query string and evaluates it against the index.

Supported syntax
    cat dog                 free text (implicit AND)
    cat AND dog / OR / NOT  boolean operators (UPPERCASE) and ( ) brackets
    "machine learning"      phrase (uses word positions)
    comp*  *tion  c?t       wildcard (delegated to the wildcard module)
    soundex:smith / ~smith  phonetic match
    informaton              unknown word -> spelling correction

Every search runs inside its own _QueryRun object, so no state is shared
between searches and the processor is safe to call from several threads.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from .indexer import InvertedIndex
from .normalizer import Normalizer
from .phonetic import PhoneticIndex
from .ranking import rank
from .spelling import SpellingCorrector
from .tokenizer import Token, tokenize
from .wildcard import WildcardMatcher

_QUERY_TOKEN_RE = re.compile(r'"[^"]*"|\(|\)|[^\s()"]+')


@dataclass
class QueryResult:
    query: str
    hits: List[Tuple[int, float]] = field(default_factory=list)  # (doc_id, score), best first
    notes: List[str] = field(default_factory=list)               # corrections, expansions...
    matched_terms: Set[str] = field(default_factory=set)         # index terms used (highlighting)
    query_types: List[str] = field(default_factory=list)

    @property
    def doc_ids(self) -> List[int]:
        return [doc_id for doc_id, _ in self.hits]


def _intersect(a: Optional[Set[int]], b: Optional[Set[int]]) -> Optional[Set[int]]:
    """AND where None means 'ignore this side' (e.g. a stop-word)."""
    return b if a is None else a if b is None else a & b


def _union(a: Optional[Set[int]], b: Optional[Set[int]]) -> Optional[Set[int]]:
    return b if a is None else a if b is None else a | b


class QueryProcessor:
    def __init__(self, index: InvertedIndex, normalizer: Normalizer,
                 wildcard: WildcardMatcher, phonetic: PhoneticIndex,
                 speller: SpellingCorrector):
        self.index = index
        self.normalizer = normalizer
        self.wildcard = wildcard
        self.phonetic = phonetic
        self.speller = speller

    def search(self, query: str) -> QueryResult:
        return _QueryRun(self, query).execute()


class _QueryRun:
    """One evaluation of one query (recursive-descent parser + evaluator)."""

    def __init__(self, processor: QueryProcessor, query: str):
        self.p = processor
        self.query = query
        self.tokens = _QUERY_TOKEN_RE.findall(query)
        self.i = 0
        self.notes: List[str] = []
        self.terms: Set[str] = set()
        self.types: List[str] = []

    # ---------------------------------------------------------- entry point
    def execute(self) -> QueryResult:
        docs: Optional[Set[int]] = None
        while self.i < len(self.tokens):
            if self._peek() == ")":  # stray closing bracket
                self.i += 1
                continue
            docs = _intersect(docs, self._parse_or())
        if docs is None:
            self.notes.append("Nothing to search for (empty query or only stop-words).")
            hits: List[Tuple[int, float]] = []
        else:
            hits = rank(self.p.index, docs, self.terms)
        return QueryResult(self.query, hits, self.notes, set(self.terms), self.types)

    # ------------------------------------------------------------- grammar
    #   or_expr  := and_expr ( "OR" and_expr )*
    #   and_expr := not_expr ( ["AND"] not_expr )*
    #   not_expr := "NOT" not_expr | atom
    #   atom     := "(" or_expr ")" | "phrase" | term
    def _peek(self) -> Optional[str]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def _mark(self, kind: str) -> None:
        if kind not in self.types:
            self.types.append(kind)

    def _parse_or(self) -> Optional[Set[int]]:
        left = self._parse_and()
        while self._peek() == "OR":
            self.i += 1
            self._mark("Boolean")
            left = _union(left, self._parse_and())
        return left

    def _parse_and(self) -> Optional[Set[int]]:
        left = self._parse_not()
        while self._peek() not in (None, "OR", ")"):
            if self._peek() == "AND":
                self.i += 1
                self._mark("Boolean")
            left = _intersect(left, self._parse_not())
        return left

    def _parse_not(self) -> Optional[Set[int]]:
        if self._peek() == "NOT":
            self.i += 1
            self._mark("Boolean")
            inner = self._parse_not()
            return None if inner is None else self.p.index.all_doc_ids - inner
        return self._parse_atom()

    def _parse_atom(self) -> Optional[Set[int]]:
        token = self._peek()
        if token is None:
            return None
        self.i += 1
        if token == "(":
            inner = self._parse_or()
            if self._peek() == ")":
                self.i += 1
            return inner
        if token == ")":
            return None
        if token.startswith('"'):
            return self._phrase(list(tokenize(token.strip('"'))))
        return self._term(token)

    # ------------------------------------------------------ term handlers
    def _docs_for_words(self, words) -> Set[int]:
        """Union of the documents containing any of the given vocabulary words."""
        docs: Set[int] = set()
        for word in words:
            term = self.p.index.term_for_surface(word)
            self.terms.add(term)
            docs |= self.p.index.doc_ids(term)
        return docs

    def _term(self, raw: str) -> Optional[Set[int]]:
        norm = self.p.normalizer
        folded = norm.fold(raw)

        # phonetic query
        if folded.startswith("soundex:") or folded.startswith("~"):
            word = folded[len("soundex:"):] if folded.startswith("soundex:") else folded[1:]
            if not word:
                return None
            self._mark("Soundex")
            code, words = self.p.phonetic.lookup(word)
            self.notes.append(f"Soundex {code} for '{word}': {', '.join(sorted(words)) or 'no matches'}")
            return self._docs_for_words(words)

        # wildcard query
        if WildcardMatcher.is_wildcard(folded):
            self._mark("Wildcard")
            words = self.p.wildcard.match(folded)
            shown = ", ".join(words[:15]) + (" ..." if len(words) > 15 else "")
            self.notes.append(f"Wildcard '{folded}': {shown or 'no matches'}")
            return self._docs_for_words(words)

        # ordinary word
        tokens = list(tokenize(raw))
        if not tokens:
            return None
        if len(tokens) > 1:  # e.g. e-mail -> treated as a phrase
            return self._phrase(tokens)
        word = norm.fold(tokens[0].text)
        if norm.is_stop_word(word):
            return None  # ignored
        self._mark("Keyword")
        term = norm.stem(word)
        if term in self.p.index:
            self.terms.add(term)
            return self.p.index.doc_ids(term)

        # unknown word -> try spelling correction
        suggestion = self.p.speller.correct(word)
        if suggestion:
            self._mark("Spell correction")
            self.notes.append(f"Spell-corrected '{word}' -> '{suggestion}'")
            return self._docs_for_words([suggestion])
        self.notes.append(f"No match for '{word}'")
        return set()

    def _phrase(self, tokens: List[Token]) -> Optional[Set[int]]:
        norm, index = self.p.normalizer, self.p.index
        keep: List[Tuple[str, int]] = []  # (term, original position); stop-words skipped
        for tok in tokens:
            folded = norm.fold(tok.text)
            if not norm.is_stop_word(folded):
                keep.append((norm.stem(folded), tok.position))
        if not keep:
            return None
        self._mark("Phrase")
        for term, _ in keep:
            if term not in index:
                self.notes.append(f"Phrase word '{term}' is not in the index")
                return set()
            self.terms.add(term)

        base = keep[0][1]
        matches: Set[int] = set()
        for doc_id, starts in index.postings(keep[0][0]).items():
            position_sets = []
            for term, _ in keep:
                positions = index.positions(term, doc_id)
                if not positions:
                    break
                position_sets.append(set(positions))
            else:
                # every word must sit at the same relative offset as in the phrase
                if any(all((start + pos - base) in found
                           for (_, pos), found in zip(keep, position_sets))
                       for start in starts):
                    matches.add(doc_id)
        return matches
