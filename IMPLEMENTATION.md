# Implementation Notes — subtitle_vocab weekend CLI prototype

This file documents **everything** about both implementation passes: the weekend
CLI baseline (§1–§7) and the definitions + real-CEFR + `--exclude` step (§8).

> Product statement (also in README): **this is a baseline
> vocabulary-ranking system, not a scientifically validated CEFR classifier.**
> CEFR-J data © Tono Laboratory, TUFS (cite properly); Octanove C1/C2 data
> © Octanove Labs, CC BY-SA 4.0 — see README for full attribution.

---

## 8. Next step (definitions + real CEFR + --exclude)

### 8.1 Files changed / created (this pass)

| File | Status | Purpose |
|------|--------|---------|
| `subtitle_vocab.py` | **extended** (scorer untouched) | `--exclude` filter, `CefrProfile` POS-aware lookup, `analyze_subtitles` now also returns POS counts, `build_deck(..., definitions=...)`, repeatable `--cefr` with format auto-detect |
| `define.py` | **created** | CSV → Groq GPT-OSS 120B → `definitions.json` cache → rebuilt `.apkg`; stdlib-only HTTP (no new deps), provider/model configurable |
| `tests/test_scoring.py` | **extended** | exclude parsing/filtering, POS mapping, profile quirks (`vern`, wildcards, variants), merge priority, dominant-POS lookup, real-data spot checks |
| `tests/test_define.py` | **created** | cache keys, validation, cache-hit/miss, failure tolerance, caps, deck rebuild with/without definitions (LLM mocked — no network) |
| `README.md` | **extended** | `--exclude`, CEFR-J/Octanove usage + attribution, `define.py` usage |

Suite is now **59 passed, ~4s**.

### 8.2 CEFR integration notes

- Sources (NOT vendored; at `../olp-en-cefrj/`): `cefrj-vocabulary-profile-1.5.csv`
  (7,799 rows, A1–B2) + `octanove-vocabulary-profile-c1c2-1.0.csv` (2,136 rows,
  C1–C2) → **8,690 lemmas** loaded. Grammar profile (`cefrj-grammar-profile-*`)
  present in the data dir but intentionally not integrated (grammar ≠ vocab).
- `(lemma, POS) → CEFR` preserved: spaCy UPOS mapped onto the profile `pos`
  vocabulary (`NOUN→noun`, `AUX+be→be-verb`, modals→`modal auxiliary`, …);
  exact POS match wins, else TSV-style wildcard, else hardest listed sense.
- Real-data checks: `sword→B1`, `warrior→B1`, `deceit→C1`, `matchmaker→None`,
  `mulan→None`. Note `dishonor` is absent from both profiles (UNKNOWN path) —
  the data has gaps; the CSV makes them visible.
- Mulan 2020 effect (B2, min-count 2, 6 excludes): candidates 93 → **26** —
  `sword/warrior/emperor/ancestor` now correctly gate out at B2 via real levels.

### 8.3 define.py notes

- Key via `$GROQ_API_KEY` only; provider presets groq/openai/ollama + `--api-base`.
- Cache key = sha256(lemma | level | contexts); entries validated pre-write,
  one retry on bad JSON, per-word failure never breaks the deck.
- Verified end-to-end against a local OpenAI-compatible mock (incl. retry on
  malformed JSON + sqlite inspection of the `.apkg` — definition/sense/example
  present, order preserved).

### 8.4 Blocked: live Groq call from this machine
`https://api.groq.com` returns **HTTP 403, Cloudflare error 1010** (network/IP
policy block) even for `/models` — the failure is at the Cloudflare edge, not
authentication, so the key itself is untested. **Action for Belal: run
`define.py` on your own machine/network** (README has the exact commands).
`--dry-run` previews cache hits vs calls first. Also: the key was pasted in
chat — consider rotating it in the Groq console (code never stores it).

---

## 1. Files changed / created

> Product statement (also in README): **this is a baseline
> vocabulary-ranking system, not a scientifically validated CEFR classifier.**
> CEFR datasets may have licensing restrictions — do not redistribute
> scraped/copyrighted vocabulary data.

---

## 1. Files changed / created

| File | Status | Purpose |
|------|--------|---------|
| `subtitle_vocab.py` | **rewritten** (was 251 lines, now ~480) | Entire pipeline, still a single modular file (no package refactor — single file was cleaner) |
| `requirements.txt` | **edited** | Added `pytest>=7` so `pytest` works right after `pip install -r requirements.txt` |
| `cefr.tsv` | **extended** | Demo CEFR rows covering the fixture vocabulary (`mechanism`, `peculiar`, `conceal`, `perfunctory`, `coerce`, `obfuscate`, …) |
| `README.md` | **rewritten** | Fresh-developer guide: what/why/install/usage/CEFR format/output/ranking/limitations/roadmap |
| `examples/test.srt` | **created** | Tiny 7-subtitle fixture (mechanism/peculiar/conceal/perfunctory/coerce/…, plus formatting edge cases) |
| `tests/__init__.py` | **created** | Makes `tests/` a package |
| `tests/test_parser.py` | **created** | SRT parsing, cleaning, lemmatization, entity filtering (17 tests) |
| `tests/test_scoring.py` | **created** | CEFR lookup, scoring formula, level gating, ranking (12 tests) |
| `tests/test_exporters.py` | **created** | CSV generation, Anki deck, highlighting (6 tests) |
| `IMPLEMENTATION.md` | **created** | This file |
| `examples/test_vocab.apkg` / `examples/test_vocab.csv` | **generated** | Build artefacts from the definition-of-done run (regenerable, git-ignorable) |

Architecture decision: kept the **single-file** layout (`subtitle_vocab.py` with
well-separated functions) instead of a `subtitle_vocab/` package, per the brief
("don't refactor just for the sake of abstraction"). Entry point stays
`python subtitle_vocab.py …`, so the definition of done is unaffected.

---

## 2. Pipeline (what the code does, in order)

```text
SRT bytes (utf-8-sig, BOM-tolerant, errors=replace)
 → parse_srt_file()          # lenient block parser first, strict `srt` fallback
 → clean_subtitle()          # per-block text normalisation
 → analyze_subtitles()       # spaCy tokenize+lemmatize, should_skip_token(),
                             # consecutive-duplicate collapse, surface forms + contexts
 → zipf_frequency()          # wordfreq, per lemma
 → load_cefr() / is_eligible() # level gate; missing → UNKNOWN + --min-zipf fallback
 → score_lemma()             # rarity / recurrence / score (standalone fn)
 → build_candidates()        # gate + score + sort
 → write_csv() + build_deck()# movie_vocab.csv + movie_vocab.apkg
```

### 2.1 Robustness fixes (section 1 of the brief)

- **UTF-8 BOM**: files read as `utf-8-sig`; stray `\ufeff` stripped per line.
- **Malformed SRT**: `_lenient_parse()` splits on blank lines, requires a `-->`
  timestamp line, validates timestamps, skips bad blocks. Chosen as the
  *primary* parser because strict `srt.parse()` can silently merge garbage
  lines into a neighbour's content (reproduced in tests).
- **HTML/formatting tags**: `<i>…</i>` etc. stripped via regex.
- **ASS/SSA**: `{\\an8}` overrides removed; `\N`, `\n`, `\h` escapes → space.
- **Hearing-impaired cues**: `[Music]`, `(laughs)` removed.
- **Speaker labels / dialogue marks**: `JOHN:`, `- `, `>>`, `♪/♫` stripped
  per line (speaker regex is intentionally narrow: `^[A-Z][A-Z0-9 .'-]{1,30}:`).
- **HTML entities**: `html.unescape()` first.
- **Duplicate lines**: consecutive identical cleaned lines counted once
  (retiming duplicates must not inflate recurrence); identical example
  sentences stored once per lemma.
- **Contractions**: left to spaCy tokenisation; fragments (`n't`, `'s`) are
  non-alpha → filtered. Documented, not special-cased.
- **Punctuation / numbers / short tokens**: `not tok.is_alpha`, `like_num`,
  `is_stop`, `len(lemma) < MIN_TOKEN_LEN (=3)`, non-`isalpha` lemmas dropped.
- **Names/places**: `should_skip_token()` skips **only** `ent_type_ in
  EXCLUDED_ENT_TYPES AND pos_ == PROPN` — deliberately conservative so common
  words ("mark", "grace", "will") survive when used as ordinary words.
- **Empty subtitles**: cleaned to `""` → skipped; empty file → `[]`, no crash.
- **Unicode**: `is_alpha`/`isalpha` keep `café`, `naïve`, Devanagari, etc.
- **Timestamps**: `HH:MM:SS` (milliseconds dropped by design).

### 2.2 Scoring model (section 2 of the brief)

Standalone, replaceable function:

```python
def score_lemma(count, zipf, weight=0.35):
    rarity     = max(0.0, 8.0 - zipf)
    recurrence = math.log1p(max(0, count))
    score      = rarity * (1.0 + weight * recurrence)
    return rarity, recurrence, score
```

All five per-lemma values (`count, zipf, cefr, rarity, recurrence, score`)
flow into `build_candidates()` dicts and are all exposed in the CSV.

### 2.3 Learner level (section 3 of the brief)

- `--level C1` means *"show vocabulary likely above C1"*.
- CEFR known → keep iff `CEFR_ORDER[word] > CEFR_ORDER[learner]`
  (equal-or-below excluded).
- CEFR missing → labelled **`UNKNOWN`** (never `UNK`, never "predicted CEFR"),
  kept iff `0 < --min-zipf` and `zipf <= --min-zipf`.
- Behaviour is stated in `--help`, README, and the no-CEFR console note.

### 2.4 CSV export (section 4 of the brief)

Columns (exact order): `rank, lemma, surface_forms, count, cefr, zipf, rarity,
recurrence, score, example, timestamp`. Floats formatted to 2 decimals.
`surface_forms` ordered by frequency then alphabetically
(e.g. `conceal` → `concealed`; demo shows real variant grouping).
`example` is the first real subtitle line containing the lemma.

### 2.5 Anki cards (section 5 of the brief)

- Front: escaped word. Back: word + `CEFR: … · Zipf: …. · Seen in this show:
  N×` + up to `MAX_CONTEXTS (=3)` deduplicated contexts with timestamps.
- Highlighting escapes the sentence **first**, then `<b>`-wraps every known
  surface form (longest-first, word-boundary, case-insensitive) — no tag
  injection (covered by test).
- Fixed a double-escaping bug in the old `info` field (escaped components
  once, into an unescaped template).
- Deck id is now deterministic per deck name (`md5 → 31-bit`), not a hardcoded
  constant that collides across decks. Model id unchanged.

### 2.6 CLI (section 6 of the brief)

```bash
python subtitle_vocab.py movie.srt --level C1 --top 50 \
  --output movie_vocab.apkg --csv movie_vocab.csv --cefr cefr.tsv
```

- Defaults: `level=C1`, `top=50`, `output=<input>_vocab.apkg`,
  `csv=<input>_vocab.csv` (same directory as input); `-o` kept as alias.
- Plus `--min-count`, `--min-zipf`, `--cefr` as specified.
- Prints: input path, subtitle count, unique lemmas, candidates
  (pre-`--top`), selected, both output paths, top-10 list
  `(CEFR, count, zipf, score)`, and a no-CEFR guidance note.

---

## 3. Commands run

Environment used for verification (system python is externally managed, so a
venv was used; equivalent to the README venv):

```bash
python3 -m venv /tmp/opencode/.venv --system-site-packages
/tmp/opencode/.venv/bin/pip install -r requirements.txt
/tmp/opencode/.venv/bin/python -m spacy download en_core_web_sm

# definition-of-done run (no CEFR file, as specified)
/tmp/opencode/.venv/bin/python subtitle_vocab.py examples/test.srt --level C1 --top 20
# → examples/test_vocab.apkg + examples/test_vocab.csv

# CEFR runs
/tmp/opencode/.venv/bin/python subtitle_vocab.py examples/test.srt \
  --level C1 --top 20 --cefr cefr.tsv
/tmp/opencode/.venv/bin/python subtitle_vocab.py examples/test.srt \
  --level B2 --top 20 --cefr cefr.tsv --csv /tmp/b2.csv --output /tmp/b2.apkg

# tests
/tmp/opencode/.venv/bin/python -m pytest tests/ -q
```

README-equivalent fresh setup (for a real clone) is:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python subtitle_vocab.py examples/test.srt --level C1 --top 20
pytest
```

---

## 4. Test results

```text
35 passed in ~2–4s
```

Breakdown: `test_parser.py` 17 (parse fixture/empty/BOM+malformed, consecutive
dedup, 9 cleaning cases, unicode, surface-form grouping, noise filtering,
2 entity-filtering tests) · `test_scoring.py` 12 (CEFR normalisation/empty/bad
header, 4 scoring tests, level gate, UNKNOWN labelling+gating, eligibility
matrix, ranking order, min-count) · `test_exporters.py` 6 (CSV columns/content,
variant surfacing, highlight escaping, deck structure, deterministic ids).

Two failures were found and fixed during development:

1. `test_parse_bom_and_malformed_gracefully` — strict `srt.parse()` merged the
   garbage block into the previous subtitle instead of raising. Fix: lenient
   parser became primary, strict demoted to fallback.
2. `test_deck_ids_deterministic` — used `deck.id`; genanki's attribute is
   `deck.deck_id`. Test fixed.

---

## 5. Example top-ranked vocabulary

Fixture `examples/test.srt`, `--level C1 --top 20 --cefr cefr.tsv`:

```text
1. obfuscate   (CEFR C2, count 1, zipf 2.28, score 7.11)
2. perfunctory (CEFR C2, count 1, zipf 2.28→2.45, score 6.90)
3. coerce      (CEFR C2, count 1, zipf 2.86, score 6.39)
```

Sanity: at C1 only C2 lemmas survive; ordering follows rarity. At `--level B2`
the list correctly extends with `meticulous (C1)`, `conceal (C1, surface
"concealed")`, `mechanism (C1, count 2, recurrence-boosted)`. Without `--cefr`,
the same rarity order appears labelled `UNKNOWN`. CSV inspection (`surface_forms,
example, timestamp` columns) confirms the ranking is intuitively sensible.

---

## 6. Known issues (not fixed — out of scope or by design)

1. **spaCy lemmatizer artefacts** (e.g. `malformed` → `malforme` when tagged
   VERB). Visible in CSV; fixture avoids showcasing it. A curated
   lemma-override list would fix it but adds maintenance burden for a weekend
   prototype.
2. **Proper-noun leakage**: conservative NER rule lets some unrecognised names
   through (`Alice` is caught, but obscure names may not be).
3. **Parenthetical stripping is wholesale**: real words inside `(…)` are lost
   along with sound cues.
4. **No phrase detection**: `get away with` etc. are scored as single words.
5. **No sense handling**: one lemma = one card; polysemy ignored.
6. **Timestamps drop milliseconds**; missing start → `??:??:??`.
7. **wordfreq returns 0.0 for OOV** → `rarity = 8.0` (maximally rare). Usually
   right for typos/jargon, but typos get top-ranked — CSV makes this auditable.
8. **`--min-zipf <= 0` excludes ALL unknown words**, which can yield an empty
   deck when no CEFR file is given — intended, but surprising; the console note
   explains it.

## 7. Deliberately NOT built (per brief §10)

Web UI, database, accounts, embeddings, LLM calls, vector DB, ML training,
pronunciation/TTS, browser extension, subtitle-sync UI, Anki sync, cloud
deploy. Next step when the baseline proves useful: sense-aware definitions
from subtitle context, then phrase detection as a separate scorer.

---

## 9. Output tree + Rich TUI (later pass)

- Every input now gets `outputs/<slug>/` (`slugify_stem`: strips `.srt`
  suffixes, maps junk to underscores) containing `vocab.csv`, `vocab.apkg`,
  `definitions.json`, `vocab_defined.apkg`, `vocab.md`. Base moved with
  `--outdir`; explicit paths still override. `define.py` uses short names
  inside such folders, stem-based names elsewhere.
- `tui.py` (requires `rich>=13`): .srt discovery table, settings prompts,
  ranking with summary + top-10 table, optional definitions with progress bar
  (API key pasted once, session-only, never stored), outputs table. Needs a
  tty; degrades with a clear message otherwise. Verified end-to-end via a
  pseudo-terminal run.
- Loose Mulan/Bolt outputs migrated into `Lingo/outputs/`; superseded Mulan
  runs under `.../archive/`. Lingo root now holds only SRTs, data, outputs,
  project.
- Suite: **72 passed**. `define_all` gained an optional `on_each` progress
  callback (additive, default off).
