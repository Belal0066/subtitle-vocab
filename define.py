#!/usr/bin/env python3
"""Add LLM definitions to an existing subtitle-vocab CSV + rebuild the Anki deck.

Pipeline:
    existing CSV (from subtitle_vocab.py)
     -> define.py
     -> Groq API (default model: openai/gpt-oss-120b)
     -> definitions.json (cache, no database)
     -> updated Anki .apkg (same ranking/order; cards gain definition cards)

The ranking CSV and scorer are never modified here -- this only ADDS
definition/sense/example to cards. If generation fails for a word (or the
API key is missing), its card is still generated normally, without a
definition.

Configuration (all optional, all overridable -- never hardcoded to Groq):
    GROQ_API_KEY   API key, read ONLY from the environment, never CLI/files.
    --provider     groq | openai | ollama (selects default base URL + key env)
    --model        model id (default: openai/gpt-oss-120b)
    --api-base     override the provider's chat-completions base URL
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# Reuse ranking/Anki code; the scorer itself is untouched.
from subtitle_vocab import (
    CEFR_ORDER,
    MAX_CONTEXTS,
    build_deck,
)

DEFAULT_MODEL = "openai/gpt-oss-120b"

PROVIDERS = {
    # provider: (default base url, env var holding the key, key required?)
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", True),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", True),
    "ollama": ("http://localhost:11434/v1", "OLLAMA_API_KEY", False),
}

SYSTEM_PROMPT = (
    "You are a vocabulary assistant for language learners. "
    "Given a target word, the learner's CEFR level, and real subtitle "
    "lines where the word occurs, reply with STRICT JSON only -- no "
    "markdown, no commentary -- shaped exactly like: "
    '{"definition": "...", "sense": "...", "example": "...", '
    '"synonyms": {"easier": "...", "harder": "..."}, "antonym": "..."}. '
    "Rules: definition must be understandable at the learner's CEFR level; "
    "sense must name the meaning actually used in the subtitles "
    "(a few words, e.g. \"loss of respect\"); "
    "example must be one natural, clean sentence using the word in that "
    "sense (not copied verbatim from the subtitles); "
    "synonyms.easier must be a MORE COMMON, easier synonym of that sense; "
    "synonyms.harder must be a RARER or more formal synonym of that sense; "
    "antonym must be an opposite of that sense, or \"-\" when no natural "
    "opposite exists (proper names and concrete nouns often have none)."
)


# ---------------------------------------------------------------------------
# LLM client (minimal OpenAI-compatible chat over stdlib urllib)
# ---------------------------------------------------------------------------

def _post_chat(base_url: str, api_key: str | None, model: str,
               messages: list[dict], timeout: int = 60) -> str:
    """POST one chat-completions request; return the assistant's text."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    # NOTE: a browser-like UA is required -- Groq's Cloudflare edge answers
    # urllib's default "Python-urllib/..." UA with HTTP 403 / error 1010.
    headers = {"Content-Type": "application/json",
               "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/126.0 Safari/537.36"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=payload, headers=headers,
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected chat response shape: {body!r}") from exc


def build_user_prompt(lemma: str, level: str, contexts: list[str]) -> str:
    lines = "\n".join(f"- {c}" for c in contexts[:MAX_CONTEXTS])
    return (f"Word: {lemma}\nLearner CEFR level: {level}\n"
            f"Subtitle contexts:\n{lines}")


def validate_definition(data) -> dict | None:
    """Return a cleaned definition dict, or None if bad.

    Schema: {definition, sense, example,
             synonyms: {easier, harder}, antonym}.
    """
    if not isinstance(data, dict):
        return None
    try:
        definition = str(data["definition"]).strip()
        sense = str(data["sense"]).strip()
        example = str(data["example"]).strip()
        synonyms = data["synonyms"]
        easier = str(synonyms["easier"]).strip()
        harder = str(synonyms["harder"]).strip()
        antonym = str(data["antonym"]).strip()
    except (KeyError, TypeError, AttributeError):
        return None
    if not (definition and sense and example and easier and harder
            and antonym):
        return None
    if len(definition) > 500 or len(example) > 300 or len(sense) > 160:
        return None
    if len(easier) > 60 or len(harder) > 60 or len(antonym) > 60:
        return None
    if len({easier.lower(), harder.lower()}) < 2:
        return None
    return {"definition": definition, "sense": sense, "example": example,
            "synonyms": {"easier": easier, "harder": harder},
            "antonym": antonym}


def fetch_definition(base_url: str, api_key: str | None, model: str,
                     lemma: str, level: str, contexts: list[str]) -> dict:
    """Ask the model for a definition; retry once on bad JSON. Raises."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": build_user_prompt(lemma, level, contexts)},
    ]
    last_error: Exception | None = None
    for _ in range(2):
        try:
            raw = _post_chat(base_url, api_key, model, messages)
            clean = validate_definition(json.loads(raw))
            if clean is not None:
                return clean
            last_error = ValueError(f"invalid definition JSON: {raw[:200]!r}")
        except Exception as exc:  # network, JSON, shape -- retry once
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"definition failed for {lemma!r}: {last_error}")


# ---------------------------------------------------------------------------
# Cache (plain JSON file, no database)
# ---------------------------------------------------------------------------

def build_cache_key(lemma: str, level: str, contexts: list[str]) -> str:
    """Cache key accounts for lemma + learner level + exact contexts.

    Same word in a different movie (different contexts => possibly a
    different sense) hashes differently and is never wrongly reused.
    """
    canonical = "|".join([
        lemma.strip().lower(),
        level.strip().upper(),
        "\n".join(c.strip() for c in contexts[:MAX_CONTEXTS]),
    ])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(path)


def define_default_paths(csv_path: Path) -> tuple[Path, Path, Path]:
    """Default (cache, apkg, md) paths for a ranking CSV.

    Inside an ``outputs/<slug>/`` folder (csv named ``vocab.csv``) this
    yields short stable names: ``definitions.json`` / ``vocab_defined.apkg``
    / ``vocab.md``. For standalone CSVs it falls back to stem-based names
    next to the CSV.
    """
    folder = csv_path.parent
    if csv_path.stem == "vocab":
        return (folder / "definitions.json",
                folder / "vocab_defined.apkg",
                folder / "vocab.md")
    stem = csv_path.stem
    return (folder / f"{stem}_definitions.json",
            folder / f"{stem}_defined.apkg",
            folder / f"{stem}.md")


# ---------------------------------------------------------------------------
# CSV -> definitions -> deck
# ---------------------------------------------------------------------------

def load_csv_rows(path: Path, top: int | None = None) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if top:
        rows = rows[:top]
    return rows


def row_contexts(row: dict, full_contexts: dict[str, list] | None) -> list[str]:
    """Up to MAX_CONTEXTS subtitle lines for a CSV row.

    Prefers full re-analysed contexts (via --srt); falls back to the
    single example stored in the CSV.
    """
    if full_contexts and row.get("lemma") in full_contexts:
        return [s for s, _ in full_contexts[row["lemma"]][:MAX_CONTEXTS]]
    return [row["example"]] if row.get("example") else []


def analyse_srt_for_contexts(srt_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Re-run subtitle analysis to recover all contexts per lemma."""
    import spacy
    from subtitle_vocab import analyze_subtitles, parse_srt_file
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser"])
    except OSError:
        raise SystemExit(
            "spaCy model 'en_core_web_sm' not found. Run:\n"
            "  python -m spacy download en_core_web_sm")
    subtitles = parse_srt_file(srt_path)
    _, _, contexts, _ = analyze_subtitles(subtitles, nlp)
    return contexts


def define_all(rows: list[dict], level: str,
               full_contexts: dict[str, list] | None,
               cache: dict, base_url: str, api_key: str | None,
               model: str, provider: str,
               max_definitions: int | None = None,
               verbose: bool = True,
               on_each=None) -> tuple[dict[str, dict], dict]:
    """Fill the cache for every CSV row; return (definitions, cache).

    definitions maps lemma -> {definition, sense, example, ...}. Rows whose
    generation fails keep no entry (their cards render normally).
    ``on_each(lemma, ok)`` is called after every row (for TUI progress).
    """
    definitions: dict[str, dict] = {}
    made = 0
    for row in rows:
        lemma = (row.get("lemma") or "").strip()
        if not lemma:
            continue
        contexts = row_contexts(row, full_contexts)
        key = build_cache_key(lemma, level, contexts)
        entry = cache.get(key)
        if not (isinstance(entry, dict) and validate_definition(entry)):
            if api_key is None and PROVIDERS[provider][2]:
                if verbose:
                    print(f"  ! {lemma}: no API key; skipping",
                          file=sys.stderr)
                if on_each:
                    on_each(lemma, False)
                continue
            if max_definitions is not None and made >= max_definitions:
                if verbose:
                    print(f"  ! {lemma}: --max-definitions reached; skipping",
                          file=sys.stderr)
                if on_each:
                    on_each(lemma, False)
                continue
            try:
                clean = fetch_definition(base_url, api_key, model,
                                         lemma, level, contexts)
            except Exception as exc:
                print(f"  ! {lemma}: {exc}", file=sys.stderr)
                if on_each:
                    on_each(lemma, False)
                continue
            cache[key] = {"lemma": lemma, "level": level.upper(),
                          "contexts": contexts, **clean,
                          "model": model, "provider": provider}
            entry = cache[key]
            made += 1
            time.sleep(0.2)  # be polite to rate limits
        valid = validate_definition(entry)
        if valid:
            definitions[lemma] = valid
        elif verbose:
            print(f"  ! {lemma}: cached entry invalid; skipping",
                  file=sys.stderr)
        if on_each:
            on_each(lemma, lemma in definitions)
    return definitions, cache


def rows_to_candidates(rows: list[dict]) -> list[dict]:
    """Rebuild subtitle_vocab-style candidate dicts from CSV rows."""
    out = []
    for row in rows:
        try:
            forms = [s.strip() for s in (row.get("surface_forms") or "")
                     .split(",") if s.strip()] or [row.get("lemma", "")]
            out.append({
                "lemma": row.get("lemma", ""),
                "surface_forms": forms,
                "count": int(row.get("count") or 0),
                "cefr": row.get("cefr") or "UNKNOWN",
                "zipf": float(row.get("zipf") or 0.0),
                "rarity": float(row.get("rarity") or 0.0),
                "recurrence": float(row.get("recurrence") or 0.0),
                "score": float(row.get("score") or 0.0),
                "example": row.get("example") or "",
                "timestamp": row.get("timestamp") or "",
            })
        except (ValueError, TypeError):
            continue
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Add LLM definitions to a subtitle_vocab CSV and rebuild the "
            "Anki deck (ranking/order unchanged).\n\n"
            "Needs an API key in the environment, e.g.:\n"
            "  export GROQ_API_KEY=... && python define.py vocab.csv --srt movie.srt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("csv_file", type=Path, help="CSV from subtitle_vocab.py")
    p.add_argument("--level", choices=sorted(CEFR_ORDER, key=CEFR_ORDER.get),
                   default="B2",
                   help="Learner level definitions are worded for "
                        "(default: B2). Must match the CSV's --level for "
                        "sense consistency.")
    p.add_argument("--srt", type=Path, default=None,
                   help="Original .srt: recovers up to 3 contexts per word. "
                        "Without it, only the CSV's single example is sent.")
    p.add_argument("--top", type=int, default=None,
                   help="Only define the first N CSV rows (default: all).")
    p.add_argument("--max-definitions", type=int, default=None,
                   help="Cap fresh API calls (cache hits are free).")
    p.add_argument("--cache", type=Path, default=None,
                   help="JSON cache path "
                        "(default: definitions.json next to the CSV).")
    p.add_argument("--output", type=Path, default=None,
                   help="Output .apkg path "
                        "(default: vocab_defined.apkg next to the CSV).")
    p.add_argument("--deck-name", type=str, default=None,
                   help="Anki deck title (default: derived from --output).")
    p.add_argument("--provider", choices=sorted(PROVIDERS), default="groq",
                   help="LLM provider preset (default: groq).")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help=f"Model id (default: {DEFAULT_MODEL}).")
    p.add_argument("--api-base", type=str, default=None,
                   help="Override provider chat-completions base URL.")
    p.add_argument("--md", type=Path, default=None,
                   help="Also write a human-readable Markdown study sheet "
                        "(default: vocab.md next to the CSV). "
                        "Use --no-md to skip.")
    p.add_argument("--no-md", action="store_true",
                   help="Do not write the Markdown study sheet.")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be defined (cache hits vs API "
                        "calls) without calling the API or writing files.")
    return p.parse_args(argv)


def write_markdown(candidates: list[dict],
                   contexts: dict[str, list[tuple[str, str]]],
                   definitions: dict[str, dict],
                   level: str, title: str, path: Path) -> None:
    """Write a readable study sheet: one section per word, rank order kept.

    Includes definition/sense/example when generated; otherwise a short
    note that the card currently has subtitle context only.
    """
    out = [f"# {title} — vocabulary ({level})", "",
           f"{len(candidates)} words, ranked by show-specific usefulness. "
           f"CEFR levels from CEFR-J/Octanove profiles; "
           f"UNKNOWN = absent from the profiles (Zipf rarity fallback).", ""]
    for i, cand in enumerate(candidates, start=1):
        lemma = cand["lemma"]
        forms = ", ".join(cand["surface_forms"])
        out.append(f"## {i}. {lemma}")
        out.append("")
        out.append(f"- CEFR: **{cand['cefr']}** · seen **{cand['count']}×** "
                   f"· Zipf {cand['zipf']:.2f} · score {cand['score']:.2f}")
        out.append(f"- Forms: {forms}")
        defn = (definitions or {}).get(lemma) or {}
        if defn.get("definition"):
            out.append(f"- Definition: {defn['definition']}")
            out.append(f"- Used here as: _{defn.get('sense') or '—'}_")
            out.append(f"- Example: {defn.get('example') or '—'}")
            syns = defn.get("synonyms") or {}
            if isinstance(syns, dict) and (syns.get("easier")
                                           or syns.get("harder")):
                out.append(f"- Synonyms: easier — {syns.get('easier') or '—'} "
                           f"· harder — {syns.get('harder') or '—'}")
            if defn.get("antonym") and defn.get("antonym") != "-":
                out.append(f"- Antonym: {defn['antonym']}")
        else:
            out.append("- Definition: _not generated yet — subtitle context only_")
        lines = contexts.get(lemma, [])[:MAX_CONTEXTS]
        if lines:
            out.append("- From the film:")
            for sentence, ts in lines:
                shown = sentence.replace("\n", " ").strip()
                if len(shown) > 220:
                    shown = shown[:217] + "…"
                out.append(f"  - `{ts}` {shown}")
        out.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.csv_file.exists():
        raise SystemExit(f"CSV not found: {args.csv_file}")

    base_url, key_env, key_required = PROVIDERS[args.provider]
    base_url = args.api_base or base_url
    api_key = os.environ.get(key_env)

    stem = args.csv_file.stem
    default_cache, default_output, default_md = define_default_paths(
        args.csv_file)
    cache_path = args.cache or default_cache
    output_path = args.output or default_output
    md_path = args.md or default_md

    rows = load_csv_rows(args.csv_file, top=args.top)
    if not rows:
        raise SystemExit("CSV contains no rows.")

    full_contexts = None
    if args.srt:
        if not args.srt.exists():
            raise SystemExit(f"SRT not found: {args.srt}")
        print(f"Re-analysing {args.srt} for contexts...")
        full_contexts = analyse_srt_for_contexts(args.srt)

    cache = load_cache(cache_path)

    if args.dry_run:
        hits = misses = 0
        for row in rows:
            lemma = (row.get("lemma") or "").strip()
            if not lemma:
                continue
            key = build_cache_key(lemma, args.level,
                                  row_contexts(row, full_contexts))
            if key in cache:
                hits += 1
            else:
                misses += 1
        print(f"Rows: {len(rows)}; cache hits: {hits}; "
              f"API calls needed: {misses} (model {args.model})")
        return 0

    if api_key is None and key_required:
        print(f"Warning: ${key_env} not set -- rebuilding the deck "
              f"from cache only (no new definitions).", file=sys.stderr)

    definitions, cache = define_all(
        rows, args.level, full_contexts, cache,
        base_url, api_key, args.model, args.provider,
        max_definitions=args.max_definitions)
    save_cache(cache_path, cache)

    # Rebuild candidate/context structures rank-order preserved.
    candidates = rows_to_candidates(rows)
    contexts: dict[str, list[tuple[str, str]]] = {}
    for row, cand in zip(rows, candidates):
        lemma = cand["lemma"]
        if full_contexts and lemma in full_contexts:
            contexts[lemma] = full_contexts[lemma][:MAX_CONTEXTS]
        elif row.get("example"):
            contexts[lemma] = [(row["example"], row.get("timestamp") or "")]

    deck_name = args.deck_name or output_path.stem.replace(
        "_", " ").replace("-", " ").title()
    deck = build_deck(candidates, contexts, deck_name,
                      definitions=definitions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import genanki
    genanki.Package(deck).write_to_file(str(output_path))

    md_path = args.md or default_md
    if not args.no_md:
        write_markdown(candidates, contexts, definitions, args.level,
                       deck_name, md_path)

    print(f"CSV:         {args.csv_file} ({len(rows)} rows)")
    print(f"Definitions: {len(definitions)}/{len(candidates)} cards")
    print(f"Cache:       {cache_path} ({len(cache)} entries)")
    print(f"Anki deck:   {output_path}")
    if not args.no_md:
        print(f"Study sheet: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
