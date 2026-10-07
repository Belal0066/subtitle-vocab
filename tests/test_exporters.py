"""Tests: CSV generation and Anki deck building."""
import csv
from collections import Counter
from pathlib import Path

from subtitle_vocab import (
    CSV_COLUMNS,
    build_candidates,
    build_deck,
    highlight_forms,
    write_csv,
)


def _sample_candidates():
    counts = Counter({"mechanism": 3, "peculiar": 1})
    forms = {"mechanism": Counter({"mechanism": 2, "mechanisms": 1}),
             "peculiar": Counter({"peculiar": 1})}
    contexts = {"mechanism": [("I found the mechanism peculiar", "00:00:01"),
                              ("The mechanism was concealed", "00:00:04")],
                "peculiar": [("rather peculiar indeed", "00:00:01")]}
    cefr = {"mechanism": "C1", "peculiar": "B2"}
    out = build_candidates(counts, forms, contexts, cefr,
                           learner_level="A2", min_zipf=4.5, min_count=1)
    return out, contexts


def test_csv_columns_and_content(tmp_path):
    candidates, _ = _sample_candidates()
    assert candidates, "expected non-empty candidates at level A2"
    p = tmp_path / "vocab.csv"
    write_csv(candidates, p)
    with p.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_COLUMNS
        rows = list(reader)
    assert len(rows) == len(candidates)
    row = rows[0]
    assert row["rank"] == "1"
    assert row["lemma"]
    assert row["surface_forms"]  # e.g. "mechanism, mechanisms"
    assert row["cefr"] in ("B2", "C1")
    for col in ("zipf", "rarity", "recurrence", "score"):
        float(row[col])  # must be numeric
    assert row["example"]
    assert row["timestamp"]


def test_csv_surface_forms_show_variants(tmp_path):
    candidates, _ = _sample_candidates()
    mech = next(c for c in candidates if c["lemma"] == "mechanism")
    assert "mechanism" in mech["surface_forms"]
    p = tmp_path / "v.csv"
    write_csv(candidates, p)
    text = p.read_text(encoding="utf-8")
    assert "mechanism" in text


def test_highlight_escapes_html_and_bolds_form():
    out = highlight_forms("a <b>mechanism</b> & co", ["mechanism"])
    assert "&lt;b&gt;" in out  # original tags escaped, not injected
    assert "<b>mechanism</b>" in out
    assert "&amp;" in out


def test_build_deck_structure_and_escaping():
    candidates, contexts = _sample_candidates()
    deck = build_deck(candidates, contexts, "Test Deck")
    assert len(deck.notes) == len(candidates)
    for note in deck.notes:
        word, info, context = note.fields
        assert word and info and context
        assert "CEFR:" in info
        assert "Seen in this show:" in info
        # Word field is escaped; context contains highlight markup.
        assert "<script" not in word + info + context


def test_deck_ids_deterministic():
    _, contexts = _sample_candidates()
    d1 = build_deck([], contexts, "Same Name")
    d2 = build_deck([], contexts, "Same Name")
    d3 = build_deck([], contexts, "Other Name")
    assert d1.deck_id == d2.deck_id
    assert d1.deck_id != d3.deck_id
