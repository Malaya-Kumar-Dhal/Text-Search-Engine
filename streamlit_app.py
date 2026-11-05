"""Web front end (Streamlit).

Run from the project root:   streamlit run web/streamlit_app.py
Tabs: "Search" (the working tool) and "How it works" (live, code-free explanation).
"""
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # lets this script import the search_engine package

import pandas as pd
import streamlit as st

from search_engine import SearchEngine
from search_engine.kgram import kgrams
from search_engine.normalizer import STEMMER_NAME
from search_engine.phonetic import soundex
from search_engine.spelling import edit_distance
from search_engine.tokenizer import tokenize

st.set_page_config(page_title="Text Search Engine", page_icon="🔎", layout="centered")

st.markdown(
    """
    <style>
      .block-container {max-width: 900px; padding-top: 2rem;}
      .title {text-align:center; font-size:2.2rem; font-weight:700; margin-bottom:0;}
      .sub {text-align:center; color:#6b7280; margin-bottom:1rem;}
      .card {background:#fff; border:1px solid #e5e7eb; border-radius:10px;
             padding:12px 16px; margin-bottom:12px; color:#1f2937;}
      .docid {color:#4f46e5; font-weight:700;}
      .muted {color:#6b7280; font-size:0.85rem;}
      .note {color:#047857;}
      mark {background:#fde68a; padding:0 2px; border-radius:3px;}
    </style>
    <div class="title">Text Search Engine</div>
    <div class="sub">Tokenization · Normalization · Inverted Index · Wildcard · Soundex · Spell Correction</div>
    """,
    unsafe_allow_html=True,
)

SAMPLE_PATH = ROOT / "sample_data" / "sample.txt"


@st.cache_resource(show_spinner="Indexing...")
def build_engine(text: str) -> SearchEngine:
    engine = SearchEngine()
    engine.load_text(text)
    return engine


# ---------------------------------------------------------------- load file
_, middle, _ = st.columns([1, 2, 1])
with middle:
    upload = st.file_uploader("Load .txt file", type=["txt"])

if upload is not None:
    text, source = upload.getvalue().decode("utf-8", errors="ignore"), upload.name
else:
    text, source = SAMPLE_PATH.read_text(encoding="utf-8"), "built-in sample text"
engine = build_engine(text)
norm, index = engine.normalizer, engine.index
st.markdown(f"<div class='sub'>{html.escape(source)} — {engine.stats.documents} documents, "
            f"{engine.stats.vocabulary} unique words indexed</div>", unsafe_allow_html=True)

tab_search, tab_flow = st.tabs(["🔎  Search", "🧭  How it works"])

# ================================================================ SEARCH TAB
EXAMPLES = [("Boolean", "cat AND NOT dog"), ("Phrase", '"machine learning"'),
            ("Wildcard", "comp*"), ("Soundex", "soundex:smith"), ("Spell check", "informaton")]


def set_query(value):
    st.session_state.q = value


with tab_search:
    st.markdown("**BOX 1 · SEARCH**")
    box, button = st.columns([5, 1])
    box.text_input("Query", key="q", label_visibility="collapsed",
                   placeholder='e.g.  cat AND NOT dog  |  "machine learning"  |  comp*  |  ~smith')
    button.button("Search", type="primary", width="stretch")  # a rerun runs the search
    for column, (label, example) in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        column.button(label, on_click=set_query, args=(example,), width="stretch",
                      key=f"chip_{label}")

    st.markdown("**BOX 2 · RESULTS**")
    query = st.session_state.get("q", "").strip()
    if not query:
        st.caption("Enter a query above.")
    else:
        result = engine.search(query)
        if result.query_types:
            st.caption("Query type: " + ", ".join(result.query_types))
        for note in result.notes:
            st.markdown(f"<div class='note'>• {html.escape(note)}</div>", unsafe_allow_html=True)
        st.markdown(f"### {len(result.hits)} document(s) found")
        for doc_id, score in result.hits[:50]:
            doc_text, last, parts = engine.document(doc_id).text, 0, []
            for start, end in engine.highlight_spans(doc_id, result.matched_terms):
                parts.append(html.escape(doc_text[last:start]))
                parts.append(f"<mark>{html.escape(doc_text[start:end])}</mark>")
                last = end
            parts.append(html.escape(doc_text[last:]))
            st.markdown(
                f"<div class='card'><span class='docid'>Document {doc_id}</span> "
                f"<span class='muted'>&nbsp; relevance {score:.3f}</span><br>{''.join(parts)}</div>",
                unsafe_allow_html=True)

# ========================================================== HOW IT WORKS TAB
INDEX_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#4f46e5",
        fontname="Helvetica", fontsize=11, margin="0.15,0.08"];
  edge [color="#6b7280"];
  A [label="1. Document loader\\n(.txt file → documents)"];
  B [label="2. Tokenizer\\n(words + positions)"];
  C [label="3. Normalizer\\n(case folding,\\nstop-words, stemming)"];
  D [label="4. Index builder\\n(inverted index)", fillcolor="#c7d2fe"];
  E [label="k-gram index\\n(wildcard, spelling)"];
  F [label="Soundex index\\n(phonetic)"];
  A -> B -> C -> D; D -> E; D -> F;
}
"""

QUERY_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#ecfdf5", color="#047857",
        fontname="Helvetica", fontsize=11, margin="0.15,0.08"];
  edge [color="#6b7280"];
  Q [label="User query\\n(Box 1)"];
  P [label="5. Query processor\\n(AND / OR / NOT,\\nbrackets, phrases)"];
  T [label="Term handler", shape=diamond, fillcolor="#d1fae5"];
  W [label="6. Wildcard\\nmodule"];
  S [label="Soundex\\nlookup"];
  C [label="Spell correct\\nedit distance"];
  N [label="Normal word\\nstem + lookup"];
  H [label="Phrase\\nposition check"];
  R [label="TF-IDF\\nranking"];
  O [label="7. GUI\\nResults (Box 2)", fillcolor="#a7f3d0"];
  Q -> P -> T;
  T -> W; T -> S; T -> C; T -> N; T -> H;
  W -> R; S -> R; C -> R; N -> R; H -> R;
  R -> O;
}
"""

with tab_flow:
    st.markdown("## How this search engine is built")
    st.write("The engine works in two phases. First it **reads and organises** the text file "
             "(indexing). Then it **answers questions** about it quickly (querying). "
             "Everything below runs live on the file you loaded.")

    # ---- 1. overview
    st.markdown("### 1. Overall flow")
    st.caption("Phase A: indexing (runs once when a file is loaded)")
    st.graphviz_chart(INDEX_DOT, width="stretch")
    st.caption("Phase B: querying (runs every time you press Search)")
    st.graphviz_chart(QUERY_DOT, width="stretch")

    # ---- 2. indexing walkthrough
    st.markdown("### 2. Indexing steps, live")
    demo = st.text_area("Type any sentence to watch it move through the pipeline",
                        value="The Cats were chasing computers happily", height=80, key="demo_text")
    tokens = list(tokenize(demo))
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Step 1: Tokenization**")
        st.caption("Split the text into words and remember each word's position.")
        st.dataframe(pd.DataFrame([(t.text, t.position) for t in tokens],
                                  columns=["Token", "Position"]),
                     hide_index=True, width="stretch")
    with col_b:
        st.markdown("**Step 2: Case folding + stop-words**")
        st.caption("Lowercase every word. Very common words (the, were, and...) are dropped.")
        st.dataframe(pd.DataFrame(
            [(t.text, norm.fold(t.text),
              "removed" if norm.is_stop_word(norm.fold(t.text)) else "kept") for t in tokens],
            columns=["Token", "Folded", "Status"]), hide_index=True, width="stretch")
    with col_c:
        st.markdown("**Step 3: Stemming**")
        st.caption(f"Reduce words to their root ({STEMMER_NAME}).")
        st.dataframe(pd.DataFrame(
            [(t.text, norm.normalize(t.text)) for t in tokens if norm.normalize(t.text) is not None],
            columns=["Word", "Stem"]), hide_index=True, width="stretch")

    # ---- 3. the index
    st.markdown("### 3. The inverted index built from your file")
    st.write("Like the index at the back of a book: for each root word, the engine stores "
             "**which documents contain it and at which positions**. Searching becomes a lookup "
             "instead of reading every document.")
    term_filter = st.text_input("Filter terms (optional)", key="idx_filter", placeholder="e.g. comput")
    rows = [{"Term (stem)": term, "Found in # docs": index.df(term),
             "Postings (document: positions)": "; ".join(
                 f"D{d}: {','.join(map(str, pos))}" for d, pos in sorted(index.postings(term).items()))}
            for term in index.terms() if term_filter.lower() in term]
    st.dataframe(pd.DataFrame(rows).head(60), hide_index=True, width="stretch")
    st.caption(f"{engine.stats.terms} unique terms indexed from {engine.stats.documents} documents.")

    # ---- 4. query walkthrough
    st.markdown("### 4. Query processing steps, live")
    demo_query = st.text_input("Try a query", value="comp*", key="flow_q",
                               help='Examples: informaton | cat AND NOT dog | "machine learning" | ~smith')
    demo_result = engine.search(demo_query) if demo_query.strip() else None
    if demo_result is not None:
        st.markdown("**Detected query type:** "
                    + (", ".join(demo_result.query_types) or "none"))
        for note in demo_result.notes:
            st.markdown(f"<div class='note'>• {html.escape(note)}</div>", unsafe_allow_html=True)
        st.markdown("**Matching documents:** "
                    + (", ".join(f"D{d}" for d in demo_result.doc_ids) or "none"))
    words = [w for w in re.findall(r"[A-Za-z0-9']+", demo_query)
             if w not in ("AND", "OR", "NOT", "soundex")]
    if words:
        st.caption("What the engine knows about each word you typed:")
        st.dataframe(pd.DataFrame(
            [{"Word": w, "Soundex code": soundex(w), "Stem": norm.stem(norm.fold(w)),
              "Is stop-word": "yes" if norm.is_stop_word(norm.fold(w)) else "no",
              "Found in index": "yes" if norm.stem(norm.fold(w)) in index else "no"}
             for w in words]), hide_index=True, width="stretch")

    # ---- 5. techniques explained
    st.markdown("### 5. The techniques in plain language")
    with st.expander("Case folding and stemming (normalizer)"):
        st.write("Case folding makes 'Cat' and 'cat' the same word. Stemming reduces a word to its "
                 "root so that related forms (search, searching, searched) match each other.")
    with st.expander("Wildcard search (k-gram index)"):
        st.write("Every word is cut into overlapping two-letter pieces, with $ marking the word "
                 "start and end. A wildcard query is cut the same way, and the engine only "
                 "checks words that share those pieces, which is much faster than scanning "
                 "the whole vocabulary.")
        st.write("Example: **cat** becomes " + ", ".join(f"`{g}`" for g in sorted(kgrams("cat"))))
    with st.expander("Soundex (search by sound)"):
        st.write("Soundex turns a word into a 4-character code: first letter plus three digits "
                 "that stand for similar-sounding consonants. Words that sound alike get the "
                 "same code, so spelling differences do not matter.")
        st.write(f"Example: Smith → **{soundex('Smith')}**, Smyth → **{soundex('Smyth')}**, "
                 f"Schmidt → **{soundex('Schmidt')}**")
    with st.expander("Spell correction (edit distance)"):
        st.write("If a word is not in the index, the engine finds similar words and picks the one "
                 "needing the fewest single-letter edits (insert, delete, replace). Ties are "
                 "broken by choosing the more common word.")
        st.write(f"Example: informaton → information needs "
                 f"**{edit_distance('informaton', 'information')}** edit.")
    with st.expander("Phrase search (positions)"):
        st.write("Because the index remembers positions, the engine can check that the words of a "
                 "phrase appear next to each other, in order.")
    with st.expander("Boolean search"):
        st.write("AND keeps documents that match both sides, OR keeps documents that match either, "
                 "NOT removes documents. Brackets control the order.")
    with st.expander("Ranking (TF-IDF)"):
        st.write("Results are ordered by relevance. A word counts more when it appears often in a "
                 "document (TF) and less when it appears in almost every document (IDF).")
        st.latex(r"score(d)=\sum_{t}\left(1+\log_{10} tf_{t,d}\right)\cdot"
                 r"\log_{10}\left(1+\frac{N}{df_t}\right)")
        st.caption("tf = times term t appears in document d, df = documents containing t, "
                   "N = total documents.")

    # ---- 6. module map
    st.markdown("### 6. Project modules")
    st.dataframe(pd.DataFrame([
        ("document_loader", "Document loading", "Read the .txt file and split it into documents"),
        ("tokenizer", "Tokenization", "Turn text into words with positions"),
        ("normalizer", "Normalization", "Case folding, stop-word removal, stemming"),
        ("indexer", "Inverted index construction", "Term → documents and positions"),
        ("query_processor", "Query processing", "Parse and evaluate keyword, boolean and phrase queries"),
        ("wildcard (+ kgram)", "Wildcard processing", "Handle patterns like comp*"),
        ("phonetic / spelling", "Soundex and spell correction", "Match by sound, fix typos"),
        ("ranking", "TF-IDF", "Show the most relevant documents first"),
        ("gui / web", "Frontend", "Search box, Search button, read-only result area"),
    ], columns=["Module", "Stage", "Purpose"]), hide_index=True, width="stretch")
