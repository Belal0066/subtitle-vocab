"""Tests: define.py caching, validation, deck rebuild (no network)."""
import json

import define
from define import (
    RateLimiter,
    build_cache_key,
    define_all,
    estimate_tokens,
    fetch_definition,
    load_cache,
    parse_retry_after,
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

    def _fake_fetch(base_url, api_key, model, lemma, level, contexts,
                      **kw):
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
    out, usage = define._post_chat("https://x.test/v1", "KEY", "m",
                                   [{"role": "user", "content": "hi"}])
    assert out == "hi"
    assert usage == {}
    assert captured["url"] == "https://x.test/v1/chat/completions"
    assert captured["ua"] and "Python-urllib" not in captured["ua"]


def test_post_chat_returns_usage():
    import io
    import json as _json
    import urllib.request

    class _FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return _json.dumps(
                {"choices": [{"message": {"content": "hi"}}],
                 "usage": {"total_tokens": 123}}).encode()

    monkeypatch_urlopen = _FakeResp()
    import unittest.mock as _mock
    with _mock.patch.object(urllib.request, "urlopen",
                            return_value=monkeypatch_urlopen):
        out, usage = define._post_chat("https://x/v1", None, "m", [])
    assert usage == {"total_tokens": 123}


# --- rate limiting -------------------------------------------------------------------

class _Clock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, secs):
        self.slept.append(secs)
        self.now += secs


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") >= 1
    assert estimate_tokens("x" * 400) == 100
    assert estimate_tokens("x" * 400, "y" * 400) == 200


def test_rate_limiter_request_cap_blocks_31st_call():
    clock = _Clock()
    lim = RateLimiter(max_requests_per_min=30, max_tokens_per_min=10**9,
                      clock=clock, sleep=clock.sleep)
    for _ in range(30):
        assert lim.reserve(10) == 0.0
    waited = lim.reserve(10)
    assert 59.0 < waited <= 60.0  # oldest call must age out of the window
    assert clock.now >= 1060.0


def test_rate_limiter_token_cap_blocks_over_budget():
    clock = _Clock()
    lim = RateLimiter(max_requests_per_min=10**9, max_tokens_per_min=8000,
                      clock=clock, sleep=clock.sleep)
    assert lim.reserve(7000) == 0.0
    waited = lim.reserve(2000)  # 7000 + 2000 > 8000 -> must wait
    assert waited > 0
    assert lim.reserve(100) == 0.0  # old entry aged out while waiting


def test_rate_limiter_correct_replaces_estimate():
    clock = _Clock()
    lim = RateLimiter(max_requests_per_min=10**9, max_tokens_per_min=1000,
                      clock=clock, sleep=clock.sleep)
    lim.reserve(900)
    lim.correct(100)  # real usage was far smaller...
    assert lim.reserve(900) == 0.0  # ...so this now fits


def test_parse_retry_after():
    import urllib.error
    from email.message import Message

    def _err(val):
        headers = Message()
        if val is not None:
            headers["Retry-After"] = val
        return urllib.error.HTTPError("http://x", 429, "slow", headers, None)

    assert parse_retry_after(_err("7")) == 7.0
    assert parse_retry_after(_err(None)) is None
    assert parse_retry_after(_err("bogus")) is None
    assert parse_retry_after(_err("9999")) is None  # capped
    assert parse_retry_after(ValueError()) is None


def test_fetch_definition_retries_429_then_succeeds(monkeypatch):
    import urllib.error
    from email.message import Message

    headers = Message()
    headers["Retry-After"] = "0"
    err429 = urllib.error.HTTPError("http://x", 429, "slow", headers, None)
    calls = []
    slept = []
    good = {"definition": "d", "sense": "s", "example": "e",
            "synonyms": {"easier": "a", "harder": "b"}, "antonym": "c"}

    def _fake_post(base_url, api_key, model, messages, timeout=60):
        calls.append(1)
        if len(calls) == 1:
            raise err429
        import json as _json
        return _json.dumps(good), {"total_tokens": 50}

    monkeypatch.setattr(define, "_post_chat", _fake_post)
    monkeypatch.setattr(define.time, "sleep", slept.append)
    out = fetch_definition("http://x", "K", "m", "w", "B2", ["ctx"])
    assert out["definition"] == "d"
    assert len(calls) == 2
    assert slept  # backed off on the 429


def test_fetch_definition_gives_up_after_retries(monkeypatch):
    import urllib.error
    from email.message import Message

    err429 = urllib.error.HTTPError("http://x", 429, "slow", Message(), None)
    monkeypatch.setattr(define, "_post_chat",
                        lambda *a, **k: (_ for _ in ()).throw(err429))
    monkeypatch.setattr(define.time, "sleep", lambda s: None)
    try:
        fetch_definition("http://x", "K", "m", "w", "B2", ["c"],
                         max_retries=3)
    except RuntimeError as exc:
        assert "definition failed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_define_all_uses_limiter_only_on_miss(monkeypatch):
    import json as _json
    rows = [_row("alpha"), _row("beta")]
    key = build_cache_key("alpha", "B2", ["risk shame, dishonor, exile?"])
    cache = {key: dict(_good_def(), definition="cached.")}

    reserved, corrected = [], []

    class _Lim:
        def reserve(self, est):
            reserved.append(est)
            return 0.0

        def correct(self, actual):
            corrected.append(actual)

    def _fake_post(base_url, api_key, model, messages, timeout=60):
        return _json.dumps(_good_def()), {"total_tokens": 432}

    monkeypatch.setattr(define, "_post_chat", _fake_post)
    monkeypatch.setattr(define.time, "sleep", lambda s: None)
    defs, _ = define_all(rows, "B2", None, cache, "http://x", "KEY",
                         "m", "groq", verbose=False, rate_limiter=_Lim())
    assert defs["alpha"]["definition"] == "cached."
    assert len(reserved) == 1  # only the beta miss reserved budget
    assert corrected == [432]  # real usage corrected the estimate
