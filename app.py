"""Module 7 - Frontend GUI (Tkinter).

Layout
    header                      title and feature line
    [ Load .txt File ]          centred button + status line
    Box 1:  search input box  +  Search button  (+ Clear and example chips)
    Box 2:  read-only result area
The GUI contains no search logic: it only calls search_engine.SearchEngine.
"""
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from search_engine import SearchEngine, QueryResult

BG = "#f3f5fb"
CARD = "#ffffff"
ACCENT = "#4f46e5"
ACCENT_DARK = "#4338ca"
TEXT = "#1f2937"
MUTED = "#6b7280"
BORDER = "#e5e7eb"
GREEN = "#047857"
FONT = "Segoe UI"
MAX_RESULTS = 50


class SearchApp:
    EXAMPLES = [
        ("Boolean", "cat AND NOT dog"),
        ("Phrase", '"machine learning"'),
        ("Wildcard", "comp*"),
        ("Soundex", "soundex:smith"),
        ("Spell check", "informaton"),
    ]

    def __init__(self, root: tk.Tk, engine: SearchEngine = None):
        self.root = root
        self.engine = engine or SearchEngine()
        root.title("Text Search Engine")
        root.geometry("1000x740")
        root.minsize(780, 580)
        root.configure(bg=BG)
        self._configure_styles()
        self._build_header()
        self._build_load_section()
        self._build_search_box()
        self._build_result_area()
        self._show([("Load a text file, then type a query above.\n", "muted"),
                    ("Supports boolean, phrase, wildcard, Soundex and "
                     "spell-corrected queries.\n", "muted")])

    # ------------------------------------------------------------- styling
    def _configure_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        for name, font_size, pad in (("Load", 12, (30, 11)), ("Search", 11, (22, 8))):
            style.configure(f"{name}.TButton", background=ACCENT, foreground="white",
                            font=(FONT, font_size, "bold"), padding=pad,
                            borderwidth=0, focusthickness=0)
            style.map(f"{name}.TButton",
                      background=[("active", ACCENT_DARK), ("pressed", ACCENT_DARK)])
        style.configure("Ghost.TButton", background=BORDER, foreground=TEXT,
                        font=(FONT, 11), padding=(16, 8), borderwidth=0, focusthickness=0)
        style.map("Ghost.TButton", background=[("active", "#d1d5db"), ("pressed", "#d1d5db")])

    def _card(self, parent, title: str) -> tk.Frame:
        card = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        tk.Label(card, text=title, font=(FONT, 9, "bold"), bg=CARD, fg=MUTED,
                 anchor="w").pack(fill="x", padx=14, pady=(10, 6))
        return card

    # ------------------------------------------------------------ sections
    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", pady=(20, 4))
        tk.Label(header, text="Text Search Engine", font=(FONT, 24, "bold"),
                 bg=BG, fg=TEXT).pack()
        tk.Label(header, text="Tokenization  ·  Normalization  ·  Inverted Index  ·  "
                              "Wildcard  ·  Soundex  ·  Spell Correction",
                 font=(FONT, 10), bg=BG, fg=MUTED).pack(pady=(3, 0))

    def _build_load_section(self) -> None:
        middle = tk.Frame(self.root, bg=BG)
        middle.pack(pady=(14, 10))
        ttk.Button(middle, text="Load .txt File", style="Load.TButton",
                   command=self.load_file, cursor="hand2").pack()
        self.status = tk.Label(middle, text="No file loaded yet", font=(FONT, 10),
                               bg=BG, fg=MUTED)
        self.status.pack(pady=(8, 0))

    def _build_search_box(self) -> None:
        card = self._card(self.root, "BOX 1  ·  SEARCH")
        card.pack(fill="x", padx=24, pady=(4, 8))

        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", padx=14, pady=(2, 6))
        self.query = tk.Entry(row, font=(FONT, 13), relief="flat", bg="#f9fafb", fg=TEXT,
                              insertbackground=TEXT, highlightthickness=2,
                              highlightbackground=BORDER, highlightcolor=ACCENT)
        self.query.pack(side="left", fill="x", expand=True, ipady=8)
        self.query.bind("<Return>", lambda _e: self.search())
        ttk.Button(row, text="Clear", style="Ghost.TButton", command=self.clear,
                   cursor="hand2").pack(side="right", padx=(8, 0))
        ttk.Button(row, text="Search", style="Search.TButton", command=self.search,
                   cursor="hand2").pack(side="right", padx=(10, 0))

        chips = tk.Frame(card, bg=CARD)
        chips.pack(pady=(0, 12))
        tk.Label(chips, text="Try:", font=(FONT, 9), bg=CARD, fg=MUTED).pack(side="left", padx=(0, 6))
        for label, example in self.EXAMPLES:
            self._make_chip(chips, label, example).pack(side="left", padx=3)

    def _make_chip(self, parent, label: str, example: str) -> tk.Button:
        chip = tk.Button(parent, text=label, font=(FONT, 9), relief="flat", bd=0,
                         bg="#eef2ff", fg=ACCENT, activebackground="#e0e7ff",
                         activeforeground=ACCENT_DARK, padx=10, pady=3, cursor="hand2",
                         command=lambda: self._use_example(example))
        chip.bind("<Enter>", lambda _e: chip.config(bg="#e0e7ff"))
        chip.bind("<Leave>", lambda _e: chip.config(bg="#eef2ff"))
        return chip

    def _build_result_area(self) -> None:
        card = self._card(self.root, "BOX 2  ·  RESULTS (read-only)")
        card.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        self.output = scrolledtext.ScrolledText(
            card, wrap="word", font=("Consolas", 11), relief="flat", bg="#fbfbfe", fg=TEXT,
            padx=14, pady=10, borderwidth=0, highlightthickness=1,
            highlightbackground=BORDER, state="disabled")
        self.output.pack(fill="both", expand=True, padx=14, pady=(2, 14))
        self.output.tag_config("hit", background="#fde68a", foreground="#111827")
        self.output.tag_config("docid", font=("Consolas", 11, "bold"), foreground=ACCENT)
        self.output.tag_config("count", font=(FONT, 12, "bold"), foreground=TEXT)
        self.output.tag_config("note", foreground=GREEN)
        self.output.tag_config("muted", foreground=MUTED)

    # ------------------------------------------------------------- actions
    def _show(self, parts) -> None:
        """Replace the read-only result area with [(text, tag), ...]."""
        self.output.config(state="normal")
        self.output.delete("1.0", "end")
        for text, tag in parts:
            self.output.insert("end", text, tag)
        self.output.config(state="disabled")

    def _use_example(self, example: str) -> None:
        self.query.delete(0, "end")
        self.query.insert(0, example)
        self.search()

    def clear(self) -> None:
        self.query.delete(0, "end")
        self._show([("Cleared. Type a new query above.\n", "muted")])
        self.query.focus_set()

    def load_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            stats = self.engine.load_file(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not load file", str(error))
            return
        name = path.replace("\\", "/").split("/")[-1]
        self.status.config(
            text=f"{name}  -  {stats.documents} documents, {stats.vocabulary} unique words indexed",
            fg=GREEN)
        self._show([("File indexed successfully. Enter a query and press Search.\n", "note")])
        self.query.focus_set()

    def search(self) -> None:
        if not self.engine.is_ready:
            messagebox.showinfo("No data", "Load a .txt file first.")
            return
        query = self.query.get().strip()
        if not query:
            return
        self._show(self._format_result(self.engine.search(query)))

    def _format_result(self, result: QueryResult):
        parts = [(f"Query:  {result.query}\n", "muted")]
        if result.query_types:
            parts.append((f"Type:   {', '.join(result.query_types)}\n", "muted"))
        for note in result.notes:
            parts.append(("•  " + note + "\n", "note"))
        parts.append((f"\n{len(result.hits)} document(s) found\n", "count"))
        parts.append(("─" * 60 + "\n\n", "muted"))
        for doc_id, score in result.hits[:MAX_RESULTS]:
            text = self.engine.document(doc_id).text
            parts.append((f"Document {doc_id}", "docid"))
            parts.append((f"    relevance {score:.3f}\n", "muted"))
            last = 0
            for start, end in self.engine.highlight_spans(doc_id, result.matched_terms):
                parts.append((text[last:start], ""))
                parts.append((text[start:end], "hit"))
                last = end
            parts.append((text[last:] + "\n\n", ""))
        return parts


def run() -> None:
    root = tk.Tk()
    SearchApp(root)
    root.mainloop()
