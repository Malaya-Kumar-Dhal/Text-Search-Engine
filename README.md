# Text Search Engine

Load a `.txt` file and search it with keyword, boolean, phrase, wildcard,
phonetic (Soundex) and spell-corrected queries. Results are ranked with TF-IDF.

## Project structure

```
text_search_project/
├── main.py                      # starts the desktop GUI
├── requirements.txt
├── sample_data/sample.txt
├── search_engine/               # all search logic (no GUI code)
│   ├── document_loader.py       # 1. document loading
│   ├── tokenizer.py             # 2. tokenization
│   ├── normalizer.py            # 3. normalization: case folding, stop-words, stemming
│   ├── indexer.py               # 4. inverted index construction
│   ├── query_processor.py       # 5. query processing (boolean, phrase, terms)
│   ├── wildcard.py              # 6. wildcard processing
│   ├── kgram.py                 #    k-gram index (used by wildcard + spelling)
│   ├── phonetic.py              #    Soundex
│   ├── spelling.py              #    edit distance + spell correction
│   ├── ranking.py               #    TF-IDF
│   └── engine.py                #    facade tying the modules together
├── gui/app.py                   # 7. frontend GUI (Tkinter)
├── web/streamlit_app.py         #    web frontend + "How it works" tab
└── tests/test_search_engine.py
```

Data flow: `document_loader → tokenizer → normalizer → indexer`, and at query time
`query_processor → (wildcard | phonetic | spelling | index lookup) → ranking → GUI`.

## Run the desktop app

```
python main.py
```

Click **Load .txt File**, type a query in the search box and press **Search**
(or Enter). Results appear in the read-only result area. Try `sample_data/sample.txt`.

## Run the web app

```
pip install -r requirements.txt
streamlit run web/streamlit_app.py
```

The **How it works** tab explains the whole pipeline with diagrams and live
examples, without showing code.

To host it free: push this folder to GitHub, open share.streamlit.io, choose the
repository and set the main file to `web/streamlit_app.py`.

## Query syntax

| Query | Meaning |
|---|---|
| `information retrieval` | keyword search (words combined with AND), ranked |
| `cat AND dog`, `cat OR dog`, `cat AND NOT dog` | boolean (operators in UPPERCASE) |
| `(cat OR dog) AND food` | brackets control the order |
| `"machine learning"` | exact phrase |
| `comp*`, `*tion`, `c?t` | wildcard (`*` any letters, `?` one letter) |
| `soundex:smith` or `~smith` | words that sound alike |
| `informaton` | unknown word is spell-corrected automatically |

## Tests

```
python -m unittest discover tests -v
```

## Notes

- Install `nltk` for the Porter stemmer. Without it a simpler built-in stemmer is
  used, which conflates fewer word forms.
- Stop-words are not indexed, but their positions are still counted so phrase
  search stays correct.
