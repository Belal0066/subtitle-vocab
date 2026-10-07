"""Tests: SRT parsing, subtitle cleaning, lemmatization, entity filtering."""
from pathlib import Path

import pytest
import spacy

from subtitle_vocab import (
    analyze_subtitles,
    clean_subtitle,
    dominant_upos,
    parse_srt_file,
    should_skip_token,
)

FIXTURE = Path(__file__).resolve().parent.parent / "examples" / "test.srt"


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm", disable=["parser"])


# --- 1. SRT parsing -------------------------------------------------------

def test_parse_fixture_srt():
    subs = parse_srt_file(FIXTURE)
    assert len(subs) >= 5
    texts = [s.content for s in subs]
    assert any("mechanism" in t for t in texts)


def test_parse_empty_file_returns_empty(tmp_path):
    p = tmp_path / "empty.srt"
    p.write_text("", encoding="utf-8")
    assert parse_srt_file(p) == []


def test_parse_bom_and_malformed_gracefully(tmp_path):
    raw = (
        "\ufeff1\n00:00:01,000 --> 00:00:02,000\nHello world\n\n"
        "GARBAGE BLOCK WITHOUT TIMESTAMP\n\n"
        "3\nnot-a-timestamp\nBroken entry\n\n"
        "4\n00:00:05,000 --> 00:00:06,000\nSecond good line\n"
    )
    p = tmp_path / "bom.srt"
    p.write_bytes(raw.encode("utf-8"))
    subs = parse_srt_file(p)
    joined = " ".join(s.content for s in subs)
    assert "Hello world" in joined
    assert "Second good line" in joined
    assert "GARBAGE" not in joined


def test_consecutive_duplicate_lines_counted_once(nlp):
    class Fake:
        def __init__(self, content, start):
            from datetime import timedelta
            self.content = content
            self.start = start

    from datetime import timedelta
    subs = [
        Fake("The mechanism was concealed.", timedelta(seconds=1)),
        Fake("The mechanism was concealed.", timedelta(seconds=3)),
        Fake("Something else peculiar.", timedelta(seconds=5)),
    ]
    counts, _, _, _ = analyze_subtitles(subs, nlp)
    # "mechanism" appears in one unique consecutive block -> count 1
    assert counts.get("mechanism", 0) == 1


# --- 2. Subtitle cleaning ---------------------------------------------------

@pytest.mark.parametrize("raw,expected_part,absent", [
    ("<i>hello</i> world", "hello world", ["<i>"]),
    ("{\\an8}hello", "hello", ["{"]),
    ("[Music] hello (laughs)", "hello", ["Music", "laughs"]),
    ("JOHN: hello there", "hello there", ["JOHN:"]),
    ("- hello", "hello", []),
    (">> hello ♪", "hello", [">>", "♪"]),
    ("I found it\\Nunderneath", "I found it underneath", ["\\N"]),
    ("", "", []),
    ("   ", "", []),
])
def test_clean_subtitle(raw, expected_part, absent):
    out = clean_subtitle(raw)
    assert expected_part in out
    for token in absent:
        assert token not in out


def test_clean_unicode_kept():
    out = clean_subtitle("café naïve उम्मीदवार hello")
    assert "caf" in out and "hello" in out


def test_speaker_prefix_unicode_and_ocr_tolerant():
    # Proper Unicode all-caps label stripped.
    assert clean_subtitle("BÖRI KHAN: Good.") == "Good."
    # OCR-mangled label (lowercase l for I, 7/8 uppercase) stripped.
    assert clean_subtitle("BÖRl KHAN: So, you have news.") == "So, you have news."
    # Ordinary dialogue with a colon is kept.
    assert clean_subtitle("Listen: do this now") == "Listen: do this now"
    assert clean_subtitle("Note: something odd") == "Note: something odd"


# --- 3. Lemmatization -------------------------------------------------------

def test_lemmatization_groups_surface_forms(nlp):
    class Fake:
        def __init__(self, content):
            from datetime import timedelta
            self.content = content
            self.start = timedelta(seconds=1)

    subs = [Fake("She settled it. They are settling now. It was settled.")]
    counts, forms, ctx, pos = analyze_subtitles(subs, nlp)
    assert counts.get("settle", 0) >= 3
    assert "settled" in forms["settle"] or "settling" in forms["settle"]
    assert ctx["settle"], "expected stored example context"
    assert dominant_upos(pos, "settle") == "VERB"


def test_contractions_punctuation_numbers_short_tokens_filtered(nlp):
    class Fake:
        def __init__(self, content):
            from datetime import timedelta
            self.content = content
            self.start = timedelta(seconds=1)

    subs = [Fake("Don't go! 12345... ah oh eh a I me.")]
    counts, _, _, _ = analyze_subtitles(subs, nlp)
    # Noise tokens must not become lemmas.
    for bad in ("n't", "12345", "ah", "oh", "a"):
        assert bad not in counts


# --- 4. Name / entity filtering ---------------------------------------------

def test_obvious_names_filtered_but_common_words_kept(nlp):
    doc = nlp("Barack Obama visited Paris with Alice.")
    skipped = {t.text for t in doc if should_skip_token(t)}
    # Proper-noun entities should be skipped (conservative rule).
    assert "Obama" in skipped or "Barack" in skipped
    assert "Paris" in skipped or "Alice" in skipped

    doc2 = nlp("I found the mechanism rather peculiar.")
    kept = [t.text for t in doc2 if not should_skip_token(t)]
    assert "mechanism" in kept or "peculiar" in kept


def test_ner_alone_does_not_aggressively_remove_non_propn(nlp):
    # A verb/noun that happens to carry no PROPN tag must survive even
    # if NER fires spuriously; the common-word path must stay open.
    doc = nlp("Please mark my words and grace us with patience.")
    kept_lemmas = {t.lemma_.lower() for t in doc if not should_skip_token(t)}
    assert "mark" in kept_lemmas or "grace" in kept_lemmas
