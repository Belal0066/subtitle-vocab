"""Tests: define.py caching, validation, deck rebuild (no network)."""
import json

import define
from define import (
    build_cache_key,
    define_all,
    load_cache,
    row_contexts,
    rows_to_candidates,
    save_cache,
    validate_definition,
    write_markdown,
)
from subtitle_vocab import build_deck


def _row(lemma="dishonor", example="risk shame, dishonor, exile?"):
    return {"rank": "1", "lemma": lemma, "surface_forms": lemma,
            "count": "4", "cefr": "UNKNOWN", "zipf": "2.88",
            "rarity": "5.12", "recurrence": "1.61", "score": "8.00",
            "example": example, "timestamp": "00:01:16"}


# --- cache keys ---------------------------------------------------------------

def test_cache_key_stable_and_sensitive():
    k1 = build_cache_key("Dishonor", "b2", ["ctx a"])
    k2 = build_cache_key("dishonor", "B2", ["ctx a"])
    assert k1 == k2  # case-insensitive
    assert build_cache_key("dishonor", "C1", ["ctx a"]) != k1  # level matters
    assert build_cache_key("dishonor", "B2", ["ctx b"]) != k1  # context matters
    assert build_cache_key("other", "B2", ["ctx a"]) != k1     # lemma matters


def _good_def(**over):
    d = {"definition": "loss of respect",
         "sense": "loss of respect",
         "example": "They faced dishonor.",
         "synonyms": {"easier": "shame", "harder": "ignominy"},
         "antonym": "honor"}
    d.update(over)
    return d


# --- validation ------------------------------------------------------------------

def test_validate_definition():
    good = _good_def()
    assert validate_definition(good) == good
    assert validate_definition({"definition": "x"}) is None       # missing keys
    assert validate_definition({"definition": " ", "sense": "s",
                                "example": "e"}) is None          # empty
    assert validate_definition({"definition": "x" * 501, "sense": "s",
                                "example": "e"}) is None          # too long
    assert validate_definition("not a dict") is None
    assert validate_definition(None) is None
    # synonyms/antonym required, distinct, length-capped
    assert validate_definition(_good_def(synonyms={"easier": "shame"})) is None
    assert validate_definition(_good_def(antonym=" ")) is None
    assert validate_definition(_good_def(
        synonyms={"easier": "Shame", "harder": "shame"})) is None
    assert validate_definition(_good_def(antonym="-")) is not None  # no-opposite ok
    assert validate_definition(_good_def(antonym="x" * 61)) is None


# --- cache roundtrip ---------------------------------------------------------------

def test_load_save_cache_roundtrip(tmp_path):
    p = tmp_path / "d.json"
    assert load_cache(p) == {}
    save_cache(p, {"k": _good_def(definition="d")})
    assert load_cache(p)["k"]["definition"] == "d"


def test_load_cache_corrupt_returns_empty(tmp_path):
    p = tmp_path / "d.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_cache(p) == {}


# --- define_all -----------------------------------------------------------------------

def test_define_all_cache_hit_makes_no_api_call(monkeypatch):
    rows = [_row()]
    key = build_cache_key("dishonor", "B2", ["risk shame, dishonor, exile?"])
    cache = {key: _good_def(definition="loss of honour and respect.",
                             example="The scandal brought dishonor on them.")}

    def _boom(*a, **k):
        raise AssertionError("API must not be called on cache hit")

    monkeypatch.setattr(define, "fetch_definition", _boom)
    defs, _ = define_all(rows, "B2", None, dict(cache),
                         "http://x", "KEY", "m", "groq",
                         verbose=False)
    assert defs["dishonor"]["definition"].startswith("loss of honour")


def test_define_all_fetches_on_miss_and_caches(monkeypatch):
    rows = [_row()]
    calls = []

    def _fake_fetch(base_url, api_key, model, lemma, level, contexts):
        calls.append((lemma, level))
        return _good_def(definition="loss of respect.",
                         example="They faced dishonor.")

    monkeypatch.setattr(define, "fetch_definition", _fake_fetch)
    monkeypatch.setattr(define.time, "sleep", lambda s: None)
    defs, cache = define_all(rows, "B2", None, {}, "http://x", "KEY",
                             "m", "groq", verbose=False)
    assert len(calls) == 1
    assert defs["dishonor"]["sense"] == "loss of respect"
    assert len(cache) == 1

    # Second run hits cache: no more calls.
    defs2, _ = define_all(rows, "B2", None, cache, "http://x", "KEY",
                          "m", "groq", verbose=False)
    assert len(calls) == 1
    assert defs2 == defs


def test_define_all_failure_still_renders_card(monkeypatch):
    def _fail(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(define, "fetch_definition", _fail)
    defs, _ = define_all([_row()], "B2", None, {}, "http://x", "KEY",
                         "m", "groq", verbose=False)
    assert defs == {}  # no definition, but no crash either


def test_define_all_no_key_skips_gracefully():
    defs, _ = define_all([_row()], "B2", None, {}, "http://x", None,
                         "m", "groq", verbose=False)
    assert defs == {}


def test_define_all_max_definitions_cap(monkeypatch):
    rows = [_row("alpha"), _row("beta")]
    monkeypatch.setattr(
        define, "fetch_definition",
        lambda *a, **k: _good_def())
    monkeypatch.setattr(define.time, "sleep", lambda s: None)
    defs, _ = define_all(rows, "B2", None, {}, "http://x", "KEY", "m",
                         "groq", max_definitions=1, verbose=False)
    assert len(defs) == 1


def test_old_schema_cache_entry_is_refetched(monkeypatch):
    rows = [_row()]
    key = build_cache_key("dishonor", "B2", ["risk shame, dishonor, exile?"])
    stale = {key: {"definition": "old.", "sense": "old", "example": "old ex."}}
    calls = []
    monkeypatch.setattr(
        define, "fetch_definition",
        lambda *a, **k: (calls.append(1), _good_def())[1])
    monkeypatch.setattr(define.time, "sleep", lambda s: None)
    defs, _ = define_all(rows, "B2", None, stale, "http://x", "KEY",
                         "m", "groq", verbose=False)
    assert calls == [1]  # stale 3-field entry refetched, not reused
    assert "synonyms" in [k for k in defs["dishonor"]]


def test_row_contexts_prefers_full_analysis():
    full = {"dishonor": [("s1", "t1"), ("s2", "t2"), ("s3", "t3"),
                         ("s4", "t4")]}
    assert row_contexts(_row(), full) == ["s1", "s2", "s3"]
    assert row_contexts(_row(), None) == ["risk shame, dishonor, exile?"]
    assert row_contexts({"lemma": "x"}, None) == []


# --- deck rebuild ----------------------------------------------------------------------

def test_rows_to_candidates_parses_and_skips_bad():
    rows = [_row(), {"lemma": "bad", "count": "notanint"}]
    cands = rows_to_candidates(rows)
    assert len(cands) == 1
    assert cands[0]["lemma"] == "dishonor"
    assert isinstance(cands[0]["score"], float)


def test_build_deck_with_and_without_definitions():
    cands = rows_to_candidates([_row()])
    ctx = {"dishonor": [("risk shame, dishonor, exile?", "00:01:16")]}
    plain = build_deck(cands, ctx, "T")
    assert "loss of respect" not in plain.notes[0].fields[2]

    withdef = build_deck(
        cands, ctx, "T",
        definitions={"dishonor": _good_def(
            definition="loss of respect.",
            example="They faced dishonor.")})
    html = withdef.notes[0].fields[2]
    assert "loss of respect" in html
    assert "They faced dishonor." in html
    assert "ignominy" in html and "honor" in html  # synonyms + antonym shown
    # Order preserved, same note count.
    assert len(withdef.notes) == len(plain.notes) == 1


# --- markdown study sheet ---------------------------------------------------------------
def test_write_markdown_with_definitions(tmp_path):
    cands = rows_to_candidates([_row("dishonor"), _row("brave", "brave, and true.")])
    ctx = {"dishonor": [("risk shame, dishonor, exile?", "00:01:16")],
           "brave": [("brave, and true.", "00:23:31")]}
    defs = {"dishonor": _good_def(definition="loss of respect.",
                                     example="They faced dishonor.")}
    p = tmp_path / "vocab.md"
    write_markdown(cands, ctx, defs, "C1", "Mulan", p)
    text = p.read_text(encoding="utf-8")
    assert "# Mulan — vocabulary (C1)" in text
    assert text.index("## 1. dishonor") < text.index("## 2. brave")  # order kept
    assert "loss of respect" in text
    assert "They faced dishonor." in text
    assert "ignominy" in text and "Synonyms" in text
    assert "Antonym" in text
    assert "00:01:16" in text
    assert "not generated yet" in text  # brave has contexts but no definition


def test_write_markdown_without_anything(tmp_path):
    cands = rows_to_candidates([_row()])
    p = tmp_path / "v.md"
    write_markdown(cands, {}, {}, "C1", "T", p)
    text = p.read_text(encoding="utf-8")
    assert "## 1. dishonor" in text
    assert "Forms: dishonor" in text


# --- http client ------------------------------------------------------------------

def test_post_chat_sends_browser_user_agent(monkeypatch):
    import io
    import urllib.request

    captured = {}

    class _FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return (b'{"choices": [{"message": {"content": '
                    b'"hi"}}]}')

    def _fake_urlopen(req, timeout=None):
        captured["ua"] = req.get_header("User-agent")
        captured["url"] = req.full_url
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    out = define._post_chat("https://x.test/v1", "KEY", "m",
                            [{"role": "user", "content": "hi"}])
    assert out == "hi"
    assert captured["url"] == "https://x.test/v1/chat/completions"
    assert captured["ua"] and "Python-urllib" not in captured["ua"]
