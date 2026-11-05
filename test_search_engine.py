"""Unit tests.  Run from the project root:   python -m unittest discover tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search_engine import SearchEngine
from search_engine.document_loader import split_into_documents
from search_engine.kgram import kgrams
from search_engine.normalizer import Normalizer
from search_engine.phonetic import soundex
from search_engine.spelling import edit_distance
from search_engine.tokenizer import tokenize

SAMPLE = """Information retrieval is the science of searching documents.

Smith and Smyth both work on machine learning systems.

Computers and computation are central to computing.

The cat chased the dog around the garden."""


class TestLoading(unittest.TestCase):
    def test_paragraphs_become_documents(self):
        docs = split_into_documents(SAMPLE)
        self.assertEqual([d.doc_id for d in docs], [1, 2, 3, 4])

    def test_lines_used_when_no_blank_lines(self):
        self.assertEqual(len(split_into_documents("one\ntwo\nthree")), 3)

    def test_empty_text(self):
        self.assertEqual(split_into_documents("  \n \n"), [])


class TestTokenizerAndNormalizer(unittest.TestCase):
    def test_tokens_keep_case_and_positions(self):
        tokens = list(tokenize("The Cat sat."))
        self.assertEqual([t.text for t in tokens], ["The", "Cat", "sat"])
        self.assertEqual([t.position for t in tokens], [0, 1, 2])
        self.assertEqual((tokens[1].start, tokens[1].end), (4, 7))

    def test_normalizer(self):
        n = Normalizer()
        self.assertEqual(n.fold("CaT"), "cat")
        self.assertIsNone(n.normalize("The"))
        self.assertEqual(n.normalize("Searching"), n.normalize("search"))


class TestHelpers(unittest.TestCase):
    def test_soundex(self):
        self.assertEqual(soundex("Smith"), "S530")
        self.assertEqual(soundex("Smyth"), "S530")
        self.assertEqual(soundex("Robert"), "R163")

    def test_edit_distance(self):
        self.assertEqual(edit_distance("informaton", "information"), 1)
        self.assertEqual(edit_distance("kitten", "sitting"), 3)

    def test_kgrams(self):
        self.assertEqual(kgrams("cat"), {"$c", "ca", "at", "t$"})


class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SearchEngine()
        cls.stats = cls.engine.load_text(SAMPLE)

    def ids(self, query):
        return self.engine.search(query).doc_ids

    def test_stats(self):
        self.assertEqual(self.stats.documents, 4)

    def test_inverted_index_positions(self):
        index = self.engine.index
        term = self.engine.normalizer.normalize("machine")
        self.assertEqual(index.positions(term, 2), [6])
        self.assertEqual(index.df(term), 1)

    def test_keyword_and_ranking(self):
        result = self.engine.search("information retrieval")
        self.assertEqual(result.doc_ids, [1])
        self.assertAlmostEqual(result.hits[0][1], 1.3979, places=3)

    def test_boolean(self):
        self.assertEqual(self.ids("cat AND NOT mouse"), [4])
        self.assertEqual(self.ids("cat AND NOT dog"), [])
        self.assertEqual(sorted(self.ids("cat OR smith")), [2, 4])
        self.assertEqual(self.ids("(cat OR smith) AND dog"), [4])

    def test_phrase(self):
        self.assertEqual(self.ids('"machine learning"'), [2])
        self.assertEqual(self.ids('"learning machine"'), [])

    def test_wildcard(self):
        self.assertEqual(self.ids("comput*"), [3])
        self.assertEqual(sorted(self.ids("*tion")), [1, 3])
        self.assertEqual(self.ids("c?t"), [4])
        self.assertEqual(self.engine.search("comput*").query_types, ["Wildcard"])

    def test_soundex_query(self):
        result = self.engine.search("~smith")
        self.assertEqual(result.doc_ids, [2])
        self.assertIn("smyth", result.notes[0])

    def test_spell_correction(self):
        result = self.engine.search("informaton")
        self.assertEqual(result.doc_ids, [1])
        self.assertIn("Spell correction", result.query_types)

    def test_stop_words_and_empty(self):
        self.assertEqual(self.ids(""), [])
        self.assertEqual(self.ids("the of"), [])

    def test_unknown_word(self):
        self.assertEqual(self.ids("zzzzqqqq"), [])

    def test_search_before_loading(self):
        with self.assertRaises(RuntimeError):
            SearchEngine().search("cat")

    def test_highlight_spans(self):
        result = self.engine.search("cat")
        spans = self.engine.highlight_spans(4, result.matched_terms)
        text = self.engine.document(4).text
        self.assertEqual([text[s:e] for s, e in spans], ["cat"])


if __name__ == "__main__":
    unittest.main()
