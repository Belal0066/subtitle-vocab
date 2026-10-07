# Subtitle Vocab → Anki

Find the most useful vocabulary to learn **before watching a specific movie/show**,
and export it as an Anki deck + an inspectable CSV.

```bash
python subtitle_vocab.py movie.srt
# → movie_vocab.apkg + movie_vocab.csv
```

> **This is a baseline vocabulary-ranking system, not a scientifically validated
> CEFR classifier.** It ranks words by general rarity × in-show recurrence and
> applies an optional CEFR lookup. Nothing here predicts your personal
> difficulty or replaces a real placement test.

## Why frequency + recurrence?

- **Rarity** (wordfreq Zipf): uncommon words in general English are more likely
  worth pre-learning than "house" or "think".
- **Recurrence** (log1p count in *this* file): a rare word used 14× in this
  episode pays off more than a rarer word said once.
- **Score = rarity × (1 + 0.35 × recurrence)** — deliberately boring and
  interpretable. Inspect it in the CSV, argue with it, then improve it.

No LLM, no embeddings, no training, no database. Baseline first.

## Installation

Requires Python 3.10+. One command — first launch sets everything up:

```bash
cd subtitle_vocab_mvp
./run.sh [movie.srt]
```

`run.sh` creates `.venv/`, installs `requirements.txt`, fetches the spaCy
model on first launch, then starts the TUI. (Prefer manual control?
`./setup.sh` does the environment part alone.) An optional `.env` file
containing `GROQ_API_KEY=...` is picked up automatically — never committed
(it's gitignored).

## Usage

```bash
# Minimal: defaults are --level C1 --top 50, outputs go to outputs/<slug>/
python subtitle_vocab.py movie.srt
# → outputs/movie/vocab.apkg + outputs/movie/vocab.csv

# Interactive TUI (Rich): pick a file, tune settings, rank, define
python tui.py [movie.srt]

# Explicit (with real CEFR data + name exclusions)
python subtitle_vocab.py movie.srt \
    --level B2 \
    --top 50 \
    --min-count 2 \
    --output movie_vocab.apkg \
    --csv movie_vocab.csv \
    --cefr ../olp-en-cefrj/cefrj-vocabulary-profile-1.5.csv \
    --cefr ../olp-en-cefrj/octanove-vocabulary-profile-c1c2-1.0.csv \
    --exclude "mulan,hua,qiang,tung,xianniang"

# Add LLM definitions to the CSV's words and rebuild the deck
export GROQ_API_KEY=...
python define.py movie_vocab.csv --level C1 --srt movie.srt
# → movie_vocab_defined.apkg + movie_vocab_definitions.json
#   + movie_vocab_defined.md (readable study sheet, always written)

# Try the fixture (no CEFR file needed)
python subtitle_vocab.py examples/test.srt --level C1 --top 20
# → examples/test_vocab.apkg + examples/test_vocab.csv
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--level A1…C2` | `C1` | "Show words at/above my level." `--level B2` keeps B2+C1+C2; `--level C1` keeps C1+C2. Combine with `--min-zipf 0` for exactly those bands (no UNKNOWN). |
| `--top` | `50` | Max cards/rows; `0` = no limit, keep all candidates. |
| `--output` (`-o`) | `outputs/<slug>/vocab.apkg` | Anki deck path. |
| `--csv` | `outputs/<slug>/vocab.csv` | CSV path. |
| `--outdir` | `outputs/` next to input | Base folder for per-input output folders. |
| `--cefr` | none (repeatable) | CEFR file. Simple TSV (`lemma<TAB>cefr`) or CEFR-J/Octanove profile CSV (`headword,pos,CEFR`); later files win. Without it, every word is `UNKNOWN` and gated by `--min-zipf`. |
| `--exclude` | none | Comma-separated lemmas dropped after analysis, before ranking (case-insensitive; scorer untouched). For names the NER filter misses. |
| `--min-count` | `1` | Minimum occurrences in this file. |
| `--min-zipf` | `4.5` | Zipf fallback gate for `UNKNOWN` words: keep only `zipf <= this` (lower = rarer). `<= 0` excludes all `UNKNOWN` words. |

Example summary output:

```text
Subtitle file: movie.srt
Subtitles:     1823
Unique lemmas: 1432
Candidates:    117
Selected:      50
Anki deck:     movie_vocab.apkg
CSV:           movie_vocab.csv

Top vocabulary:
1. obfuscate (CEFR C2, count 3, zipf 2.28, score 7.89)
2. ...
```

## Output files

Every input gets its own folder so nothing gets messy:

```text
outputs/<slug>/
    vocab.csv            # ranking (from subtitle_vocab.py)
    vocab.apkg           # Anki deck, contexts only
    definitions.json     # LLM cache (from define.py)
    vocab_defined.apkg   # Anki deck with definitions
    vocab.md             # readable study sheet
```

`<slug>` is the SRT filename sanitised (`Mulan (1998)-en.srt` →
`Mulan_1998-en`). Explicit `--output/--csv/--cache/--md` paths always
override the defaults; `--outdir` moves the whole tree.

## CEFR data format

Two formats are accepted (auto-detected, `--cefr` is repeatable):

**1. Simple TSV** (demo / overrides) with exactly these columns:

```text
lemma	cefr
obfuscate	C2
perfunctory	C2
settle	B1
```

**2. CEFR-J / Octanove profile CSVs** (`headword,pos,CEFR[,...]`, POS-aware):

```bash
--cefr ../olp-en-cefrj/cefrj-vocabulary-profile-1.5.csv \   # A1–B2
--cefr ../olp-en-cefrj/octanove-vocabulary-profile-c1c2-1.0.csv  # C1–C2
```

POS-specific mappings are preserved as `(lemma, POS) → CEFR`: each lemma is
looked up with its dominant spaCy part-of-speech in *this* file (so
`light` the noun and `light` the adjective can differ). On a POS miss the
hardest listed sense is used, so a word is only gated out when *every*
listed sense is at or below your level. Spelling variants (`favourably /
favorably`), the Octanove `vern` typo, and multiword phrases (skipped —
no phrase detection yet) are handled.

- Matching is case-insensitive on the **lemma** (`Settle` → `settle`).
- Unknown/invalid levels are ignored.
- Words absent from every file get `CEFR = UNKNOWN` — Zipf frequency is **not**
  a CEFR prediction, it is only a rarity fallback.

> **Licensing / attribution (required).** The CEFR-J Wordlist v1.5 was compiled
> by Yukio Tono, Tokyo University of Foreign Studies (TUFS); copyright belongs
> to Tono Laboratory at TUFS. The CEFR-J vocabulary and grammar profiles may be
> used for research and commercial purposes free of charge **provided the
> dataset is cited properly**; neither CEFR-J nor Open Language Profiles is
> liable for inaccuracies. The Octanove C1/C2 profile is © Octanove Labs under
> **CC BY-SA 4.0**. Sources: <http://www.cefr-j.org/download.html>,
> <https://github.com/openlanguageprofiles/olp-en-cefrj>. The repo does not
> vendor this data — point `--cefr` at your local copy. The tiny `cefr.tsv`
> shipped here is a hand-made demo sample, not real data.

## Definitions layer (`define.py`)

Ranking stays frozen; this step only *adds* meaning to cards:

```bash
export GROQ_API_KEY=...   # key ONLY via environment, never CLI/files
python define.py movie_vocab.csv --level B2 --srt movie.srt
```

- Sends each word (lemma + level + up to 3 subtitle contexts; `--srt`
  recovers all 3, otherwise the CSV's single example) to the model
  (default provider `groq`, default model `openai/gpt-oss-120b`;
  `--provider openai|ollama`, `--model`, `--api-base` to reconfigure).
- The model must return strict JSON `{definition, sense, example,
  synonyms: {easier, harder}, antonym}`;
  responses are validated before caching, with one retry. `antonym` may be
  `"-"` when no natural opposite exists (correctly omitted from cards).
- Cache is a plain JSON file (`movie_vocab_definitions.json`): no database,
  no repeat API spend. The key hashes **lemma + level + exact contexts**, so
  the same word in another movie (possibly another sense) is never wrongly
  reused. `--max-definitions N` caps fresh calls; `--dry-run` previews
  cache hits vs calls.
- Rate limits are enforced client-side as sliding 60-second windows:
  `--rpm` (default 30) and `--tpm` (default 8000, estimated chars/4 and
  corrected with the API's real usage). HTTP 429s honour `Retry-After`
  with backoff instead of crashing the run.
- Rebuilds the `.apkg` in the same order; cards gain definition + "used here
  as" sense + clean example + easier/harder synonyms + antonym. If generation
  fails (or the key is missing),
  cards still render normally — just without definitions.
- Always writes a Markdown study sheet (`--md PATH`, default next to the
  `.apkg`, `--no-md` to skip): one section per word with rank, stats,
  definition if present, and timestamped film contexts — for reading without
  Anki.

## Output format

### CSV (`movie_vocab.csv`)

One row per selected word, ranked:

```text
rank,lemma,surface_forms,count,cefr,zipf,rarity,recurrence,score,example,timestamp
1,conceal,"concealed",1,C1,3.52,4.48,0.69,5.57,The mechanism was concealed underneath.,00:00:04
```

- `surface_forms`: spellings actually seen (`settle, settled, settling`).
- `example` / `timestamp`: a real subtitle line containing the word.
- `rarity / recurrence / score`: the interpretable components — sort, filter,
  and sanity-check the ranking here without opening Anki.

### Anki (`movie_vocab.apkg`)

- **Front:** the word.
- **Back:** the word, `CEFR: … · Zipf: … · Seen in this show: N×`, plus up to
  3 real subtitle contexts with timestamps and the word **bolded**.
- No generated definitions — intentionally the next layer, not this one.

## How ranking works

Per lemma (dominant POS taken from *this* file for the CEFR lookup):

```text
rarity     = max(0, 8 - zipf)          # uncommon in general English?
recurrence = log1p(count)             # repeats in THIS video?
score      = rarity * (1 + 0.35 * recurrence)
```

```text
rarity     = max(0, 8 - zipf)          # uncommon in general English?
recurrence = log1p(count)             # repeats in THIS video?
score      = rarity * (1 + 0.35 * recurrence)
```

`score_lemma()` is a standalone function — swap the formula without touching
parsing or export code. Level gating happens *before* ranking:

- CEFR known → keep iff `CEFR(word) > learner level`.
- CEFR unknown → keep iff `zipf <= --min-zipf`.

## Known limitations

- CEFR lookup uses the *dominant* POS per file; a word used in a minority
  sense keeps the majority sense's level. POS misses fall back to the hardest
  listed sense (documented in `CefrProfile`).

- spaCy's lemmatizer has quirks (e.g. "malformed" → "malforme"); rare
  artefacts can surface as lemmas — the CSV makes them visible.
- NER filtering is conservative (proper noun + excluded label); some names
  still slip through, some common words used as names may be kept.
- Sound cues in parentheses are stripped wholesale — occasional real words in
  `(…)` are lost.
- Single-word frequency ignores phrases (`get away with`, `call it a day`).
- No sense disambiguation: one lemma = one card, even for polysemous words.
- Timestamps drop milliseconds (`HH:MM:SS`).

## Future roadmap (explicitly NOT in this prototype)

Phrase detection as a separate scorer → per-learner difficulty → sense
re-checking against a bigger CEFR (grammar profile) → pronunciation/TTS →
sync services / web UI / database. Only after this baseline proves useful.

## Tests

```bash
pytest   # 72 tests, ~4s, no network
```

Covers SRT parsing, cleaning, lemmatization, entity filtering, CEFR lookup
(simple + POS-aware profiles), scoring, ranking, `--exclude`, output paths,
CSV generation, definition validation/caching, deck rebuild, and TUI
discovery helpers — the LLM HTTP layer is mocked, so tests never touch the
network. Interactive TUI steps are exercised manually, not in pytest.
