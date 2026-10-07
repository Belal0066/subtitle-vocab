#!/usr/bin/env python3
"""Subtitle vocabulary baseline: SRT -> frequency+recurrence ranking -> Anki + CSV.

Philosophy: establish a strong frequency + recurrence baseline first.
No LLM, embeddings, model training, or database.

Pipeline:
    SRT
     -> parse subtitles (robust / lenient)
     -> clean subtitle text
     -> spaCy tokenize + lemmatize + conservative NER filtering
     -> wordfreq Zipf frequency
     -> CEFR lookup, POS-aware when a CEFR-J/Octanove profile is provided
     -> difficulty / relevance score
     -> rank (minus --exclude filter)
     -> Anki (.apkg) + CSV
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import genanki
import srt
import spacy
from wordfreq import zipf_frequency

# ---------------------------------------------------------------------------
# Configurable constants (keep understandable; tweak, don't over-abstract)
# ---------------------------------------------------------------------------

CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
CEFR_UNKNOWN = "UNKNOWN"

#: Only skip a token for NER reasons when BOTH conditions hold:
#: its entity label is in this set AND spaCy tagged it as a proper noun.
#: This is intentionally conservative: we do NOT drop words merely
#: because an NER label fired (e.g. "mark" the verb vs "Mark" the name).
EXCLUDED_ENT_TYPES = {"PERSON", "GPE", "LOC", "FAC", "ORG", "NORP"}

#: Minimum lemma length. Filters noise like "ah", "oh", "eh", "yo".
MIN_TOKEN_LEN = 3

#: Weight of recurrence in the score. Exposed so it can be tuned later.
RECURRENCE_WEIGHT = 0.35

#: How many subtitle contexts to store / show per card.
MAX_CONTEXTS = 3

MODEL_ID = 1607392319


# ---------------------------------------------------------------------------
# CEFR
# ---------------------------------------------------------------------------

#: spaCy universal POS -> CEFR-J/Octanove `pos` vocabulary. AUX verbs are
#: resolved per-lemma (be/have/do/modal) by _cefr_pos_for(). Anything
#: unmapped returns None and the lookup falls back to lemma-only matching.
UPOS_TO_CEFR_POS = {
    "NOUN": "noun",
    "VERB": "verb",
    "ADJ": "adjective",
    "ADV": "adverb",
    "ADP": "preposition",
    "DET": "determiner",
    "PRON": "pronoun",
    "NUM": "number",
    "INTJ": "interjection",
}

_CONJUNCTION_POS = {"CCONJ", "SCONJ"}
_MODAL_LEMMAS = {"can", "could", "may", "might", "must", "shall",
                 "should", "will", "would", "ought", "dare", "need"}


def _cefr_pos_for(upos: str | None, lemma: str) -> str | None:
    """Map (spaCy UPOS, lemma) onto the CEFR profile `pos` vocabulary."""
    if not upos:
        return None
    if upos in UPOS_TO_CEFR_POS:
        # PROPN behaves like a common noun for lookup purposes; names are
        # almost never listed, so this simply yields "not found" for them.
        return UPOS_TO_CEFR_POS[upos]
    if upos == "PROPN":
        return "noun"
    if upos in _CONJUNCTION_POS:
        return "conjunction"
    if upos == "AUX":
        if lemma == "be":
            return "be-verb"
        if lemma == "have":
            return "have-verb"
        if lemma == "do":
            return "do-verb"
        if lemma in _MODAL_LEMMAS:
            return "modal auxiliary"
        return "verb"
    if upos == "PART":
        return "infinitive-to" if lemma == "to" else None
    return None


class CefrProfile:
    """POS-aware CEFR lookup: (lemma, pos) -> level, with lemma fallback.

    Exact (lemma, profile-pos) matches win. On a POS miss, falls back to
    the HIGHEST level listed for that lemma (assume the hardest known
    sense, so a word is only gated out when every listed sense is at or
    below the learner's level -- false exclusion is worse than an extra
    card). Plain ``lemma -> level`` TSV rows are stored as pos=None
    wildcards and lose to any POS-specific entry.
    """

    def __init__(self) -> None:
        self._pos_map: dict[tuple[str, str], str] = {}
        self._lemma_map: dict[str, set[str]] = defaultdict(set)

    def add(self, lemma: str, level: str, pos: str | None = None) -> None:
        lemma = lemma.strip().lower()
        level = level.strip().upper()
        if not lemma or level not in CEFR_ORDER:
            return
        self._lemma_map[lemma].add(level)
        if pos:
            self._pos_map[(lemma, pos.strip().lower())] = level
        elif (lemma, "") not in self._pos_map:
            # Wildcard only if no POS-specific entry claims the slot yet;
            # a later POS-specific add() still wins in lookup().
            self._pos_map[(lemma, "")] = level

    def lookup(self, lemma: str, upos: str | None = None) -> str | None:
        lemma = (lemma or "").strip().lower()
        if not lemma:
            return None
        if upos:
            pos = _cefr_pos_for(upos, lemma)
            if pos and (lemma, pos) in self._pos_map:
                return self._pos_map[(lemma, pos)]
        if (lemma, "") in self._pos_map:
            return self._pos_map[(lemma, "")]
        levels = self._lemma_map.get(lemma)
        if not levels:
            return None
        return max(levels, key=lambda lv: CEFR_ORDER[lv])

    def __len__(self) -> int:
        return len(self._lemma_map)

    def __bool__(self) -> bool:
        return bool(self._lemma_map)


def load_cefr(path: Path | None) -> dict[str, str]:
    """Load TSV with columns ``lemma<TAB>cefr``. Keys are lowercased lemmas."""
    if not path:
        return {}
    if not path.exists():
        raise SystemExit(f"CEFR file not found: {path}")
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "lemma" not in reader.fieldnames \
                or "cefr" not in reader.fieldnames:
            raise ValueError("CEFR TSV must contain columns: lemma, cefr")
        for row in reader:
            lemma = (row.get("lemma") or "").strip().lower()
            level = (row.get("cefr") or "").strip().upper()
            if lemma and level in CEFR_ORDER:
                result[lemma] = level
    return result


def _profile_variants(headword: str) -> list[str]:
    """Split a profile headword into single-word lookup variants.

    Handles ``favorably/favourably`` spelling pairs and surrounding
    whitespace (``laud ``). Multiword phrases (``air force``) are
    skipped -- there is no phrase detection yet.
    """
    variants: list[str] = []
    for part in (headword or "").split("/"):
        word = part.strip().lower()
        if word and " " not in word:
            variants.append(word)
    return variants


def load_cefr_profile_csv(path: Path, profile: CefrProfile) -> int:
    """Load a CEFR-J / Octanove CSV (headword,pos,CEFR[, ...]) into *profile*.

    Returns the number of (lemma, pos) entries added. Quirks handled:
    the ``vern`` typo for ``verb``, empty ``pos`` (wildcard), trailing
    spaces, slash-separated spelling variants, skipped multiword phrases.
    """
    if not path.exists():
        raise SystemExit(f"CEFR file not found: {path}")
    added = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {(c or "").strip().lower() for c in (reader.fieldnames or [])}
        if not {"headword", "pos", "cefr"} <= fields:
            raise ValueError(
                f"{path}: expected CEFR profile CSV with columns "
                "headword,pos,CEFR")
        # Resolve the real key spellings once (headers may vary in case).
        hw_key = next(c for c in reader.fieldnames
                      if (c or "").strip().lower() == "headword")
        pos_key = next(c for c in reader.fieldnames
                       if (c or "").strip().lower() == "pos")
        cefr_key = next(c for c in reader.fieldnames
                        if (c or "").strip().lower() == "cefr")
        for row in reader:
            level = (row.get(cefr_key) or "").strip().upper()
            if level not in CEFR_ORDER:
                continue
            pos = (row.get(pos_key) or "").strip().lower() or None
            if pos == "vern":  # typo in the Octanove file
                pos = "verb"
            for variant in _profile_variants(row.get(hw_key) or ""):
                profile.add(variant, level, pos)
                added += 1
    return added


def load_cefr_profiles(paths: list[Path] | None) -> CefrProfile:
    """Load --cefr files (auto-detected format) into one CefrProfile.

    Simple ``lemma<TAB>cefr`` TSVs contribute wildcard entries;
    CEFR-J/Octanove CSVs contribute POS-specific entries. Later files
    win on exact (lemma, pos) conflicts.
    """
    profile = CefrProfile()
    for path in paths or []:
        if not path.exists():
            raise SystemExit(f"CEFR file not found: {path}")
        # Peek at the header to pick a loader.
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            sniff = f.read(4096)
        first = sniff.splitlines()[0] if sniff.splitlines() else ""
        if "headword" in first.lower() and "\t" not in first:
            load_cefr_profile_csv(path, profile)
        else:
            for lemma, level in load_cefr(path).items():
                profile.add(lemma, level)
    return profile


# ---------------------------------------------------------------------------
# SRT parsing (robust)
# ---------------------------------------------------------------------------

def _parse_timestamp_lenient(ts: str):
    """Parse 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm'. Returns timedelta or raises."""
    from datetime import timedelta
    ts = ts.strip().replace(".", ",")
    m = re.match(r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})$", ts)
    if not m:
        m2 = re.match(r"(\d+):(\d{1,2}):(\d{1,2})$", ts)
        if not m2:
            raise ValueError(f"bad timestamp: {ts!r}")
        h, mi, sec = map(int, m2.groups())
        ms = 0
    else:
        h, mi, sec, ms = map(int, m.groups())
        if len(m.group(4)) == 1:
            ms *= 100
        elif len(m.group(4)) == 2:
            ms *= 10
    return timedelta(hours=h, minutes=mi, seconds=sec, milliseconds=ms)


def _lenient_parse(content: str) -> list:
    """Very forgiving SRT fallback: skip malformed blocks instead of failing."""
    from datetime import timedelta
    subs: list = []
    # Normalise newlines, split on blank lines.
    blocks = re.split(r"\r?\n\s*\r?\n", content.strip())
    index = 1
    for block in blocks:
        lines = [ln.strip("\ufeff").rstrip() for ln in block.strip().splitlines()]
        lines = [ln for ln in lines if ln != ""]
        if not lines:
            continue
        # Optional numeric index on first line.
        if re.match(r"^\d+$", lines[0] or ""):
            lines = lines[1:]
        if not lines:
            continue
        # Timestamp line must contain '-->'.
        if "-->" not in lines[0]:
            continue
        try:
            start_s, _, end_s = lines[0].partition("-->")
            start = _parse_timestamp_lenient(start_s)
            end = _parse_timestamp_lenient(end_s)
        except ValueError:
            continue
        text = "\n".join(lines[1:]).strip()
        if not text:
            continue
        try:
            sub = srt.Subtitle(
                index=index, start=start, end=end, content=text,
            )
        except Exception:
            continue
        subs.append(sub)
        index += 1
    return subs


def parse_srt_file(path: Path) -> list:
    """Read *path* (UTF-8-SIG to swallow BOM) and return subtitle entries.

    Tries the strict ``srt`` parser first; on failure falls back to a
    lenient block parser that skips malformed entries gracefully.
    Never raises on malformed content -- returns whatever could be parsed
    (possibly an empty list).
    """
    raw = path.read_bytes().decode("utf-8-sig", errors="replace")
    if not raw.strip():
        return []
    # Lenient block parser first: it skips malformed blocks gracefully.
    # The strict ``srt`` parser can silently merge garbage lines into a
    # neighbouring subtitle's content, so only use it as a fallback.
    try:
        subs = _lenient_parse(raw)
        if subs:
            return subs
    except Exception:
        pass
    try:
        return [s for s in srt.parse(raw) if (s.content or "").strip()]
    except Exception:
        return []


def format_timestamp(td) -> str:
    """Format a datetime.timedelta as HH:MM:SS (drops milliseconds)."""
    total = int(td.total_seconds()) if hasattr(td, "total_seconds") else 0
    total = max(0, total)
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_ASS_OVERRIDE_RE = re.compile(r"\{[^}]*\}")
_SQUARE_CUE_RE = re.compile(r"\[[^\]]*\]")
_PAREN_CUE_RE = re.compile(r"\([^)]*\)")
_WS_RE = re.compile(r"\s+")
# Speaker label like "JOHN:" / "MARY-JANE:" / "BÖRI KHAN:" at start of a line.
_SPEAKER_CANDIDATE_RE = re.compile(r"^([^:]{1,30}):\s*(.+)$")


def _strip_speaker_prefix(line: str) -> str:
    """Remove a leading ``SPEAKER:`` label, tolerating OCR noise.

    Strict pass drops all-caps (Unicode-aware via str.isupper, so
    ``BÖRI KHAN:`` works) labels. Fallback pass catches OCR-mangled
    labels like ``BÖRl KHAN:`` (lowercase ``l`` for ``I``): if the
    text before the first colon is short (<=30 chars) and >=70% of
    its letters are uppercase, treat it as a label. Ordinary
    dialogue like ``Listen: do this`` (~17% uppercase) is kept.
    """
    m = _SPEAKER_CANDIDATE_RE.match(line)
    if not m:
        return line
    prefix, rest = m.group(1), m.group(2)
    letters = [c for c in prefix if c.isalpha()]
    if not letters or not rest:
        return line
    stripped = prefix.strip()
    if stripped and stripped.isupper():
        return rest.strip()
    if len(prefix) <= 30 and \
            sum(c.isupper() for c in letters) / len(letters) >= 0.7:
        return rest.strip()
    return line


def clean_subtitle(text: str) -> str:
    """Clean one subtitle block into plain text.

    Handles: HTML tags (``<i>``), ASS/SSA overrides (``{\\an8}``),
    ASS newlines (``\\N``/``\\n``/``\\h``), hearing-impaired cues
    (``[Music]``, ``(laughs)``), speaker prefixes (``JOHN:``),
    dialogue dashes / ``>>`` / music notes, HTML entities, Unicode
    whitespace. Returns ``""`` for empty subtitles.
    """
    if not text:
        return ""
    text = html.unescape(text)
    # ASS/SSA override blocks and newline escapes.
    text = _ASS_OVERRIDE_RE.sub(" ", text)
    text = text.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
    # HTML-style tags.
    text = _TAG_RE.sub(" ", text)
    # Hearing-impaired cues.
    text = _SQUARE_CUE_RE.sub(" ", text)
    text = _PAREN_CUE_RE.sub(" ", text)

    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        # Strip dialogue markers: "- ", ">>", music notes.
        line = re.sub(r"^(?:>>|>>>)\s*", "", line)
        line = re.sub(r"^[-–—_]+(\s+|$)", "", line)
        line = line.strip("♪♫#").strip()
        line = _strip_speaker_prefix(line).strip()
        if line:
            lines.append(line)
    text = " ".join(lines)
    text = _WS_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Token filtering
# ---------------------------------------------------------------------------

def should_skip_token(tok) -> bool:
    """Return True when *tok* should be excluded from vocabulary.

    Conservative by design:
    - punctuation / numbers / stopwords / very short tokens are skipped;
    - named entities are skipped ONLY when the token is a proper noun
      with an excluded entity label (avoids killing common words like
      "mark", "will", "grace" when used as ordinary words).
    """
    if not tok.is_alpha:
        return True
    if tok.is_stop:
        return True
    if tok.like_num:
        return True
    lemma = tok.lemma_.lower().strip()
    if not lemma or len(lemma) < MIN_TOKEN_LEN or not lemma.isalpha():
        return True
    if tok.ent_type_ in EXCLUDED_ENT_TYPES and tok.pos_ == "PROPN":
        return True
    return False


# ---------------------------------------------------------------------------
# Scoring (interpretable; replaceable)
# ---------------------------------------------------------------------------

def score_lemma(count: int, zipf: float,
                weight: float = RECURRENCE_WEIGHT) -> tuple[float, float, float]:
    """Compute ``(rarity, recurrence, score)`` for one lemma.

    rarity     = max(0, 8 - zipf)        # how uncommon in general English
    recurrence = log1p(count)           # how often it recurs in THIS video
    score      = rarity * (1 + w * recurrence)

    Kept as a standalone function so it can be swapped out later
    without touching parsing / export code.
    """
    rarity = max(0.0, 8.0 - zipf)
    recurrence = math.log1p(max(0, count))
    score = rarity * (1.0 + weight * recurrence)
    return rarity, recurrence, score


def is_eligible(cefr_level: str | None, zipf: float,
                learner_order: int, min_zipf: float) -> bool:
    """Learner-level gate (inclusive: "B2 and above" means B1 excluded).

    - CEFR known: eligible iff ``CEFR(word) >= learner level``.
    - CEFR unknown: eligible iff ``zipf <= min_zipf`` (Zipf fallback;
      NOT a CEFR prediction -- labelled UNKNOWN in outputs).
      ``min_zipf <= 0`` disables UNKNOWN words entirely, so combining
      e.g. ``--level B2 --min-zipf 0`` selects exactly B2/C1/C2 words.
    """
    if cefr_level:
        return CEFR_ORDER[cefr_level] >= learner_order
    return min_zipf > 0 and zipf <= min_zipf


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_subtitles(subtitles, nlp) -> tuple[
        Counter, dict[str, Counter], dict[str, list[tuple[str, str]]],
        dict[str, Counter]]:
    """Tokenize subtitles -> lemma counts, surface forms, contexts, POS.

    Returns ``(lemma_counts, surface_forms, contexts, pos_counts)`` where
    ``surface_forms[lemma]`` is a Counter of spellings seen,
    ``contexts[lemma]`` is a list of ``(sentence, "HH:MM:SS")``, and
    ``pos_counts[lemma]`` is a Counter of spaCy UPOS tags (used for
    POS-aware CEFR lookup via the dominant tag).
    Consecutive duplicate subtitle lines are counted once (retimed
    duplicates should not inflate recurrence). Identical example
    sentences are stored once per lemma.
    """
    lemma_counts: Counter = Counter()
    surface_forms: dict[str, Counter] = defaultdict(Counter)
    contexts: dict[str, list[tuple[str, str]]] = defaultdict(list)
    pos_counts: dict[str, Counter] = defaultdict(Counter)
    seen_sentence: dict[str, set[str]] = defaultdict(set)

    prev_clean = None
    for sub in subtitles:
        content = getattr(sub, "content", "")
        start = getattr(sub, "start", None)
        cleaned = clean_subtitle(content)
        if not cleaned:
            continue
        if cleaned == prev_clean:
            continue  # retimed duplicate of the previous card
        prev_clean = cleaned
        ts = format_timestamp(start) if start is not None else "??:??:??"

        doc = nlp(cleaned)
        for tok in doc:
            if should_skip_token(tok):
                continue
            lemma = tok.lemma_.lower().strip()
            if not lemma:
                continue
            lemma_counts[lemma] += 1
            surface_forms[lemma][tok.text.lower()] += 1
            pos_counts[lemma][tok.pos_] += 1
            if cleaned not in seen_sentence[lemma]:
                seen_sentence[lemma].add(cleaned)
                contexts[lemma].append((cleaned, ts))
    return lemma_counts, surface_forms, contexts, pos_counts


def dominant_upos(pos_counts: dict[str, Counter], lemma: str) -> str | None:
    """Most frequent spaCy UPOS tag observed for *lemma* (ties -> alpha)."""
    counts = pos_counts.get(lemma)
    if not counts:
        return None
    return sorted(counts, key=lambda p: (-counts[p], p))[0]


def parse_exclude(raw: str | None) -> set[str]:
    """Parse --exclude 'a,b,c' into normalized lowercase lemmas."""
    if not raw:
        return set()
    return {w.strip().lower() for w in raw.split(",") if w.strip()}


def build_candidates(lemma_counts: Counter,
                     surface_forms: dict[str, Counter],
                     contexts: dict[str, list[tuple[str, str]]],
                     cefr,
                     learner_level: str,
                     min_zipf: float,
                     min_count: int,
                     pos_counts: dict[str, Counter] | None = None,
                     exclude: set[str] | None = None) -> list[dict]:
    """Apply exclude filter + level gate + scoring + ranking.

    ``cefr`` is a CefrProfile (or a legacy plain ``lemma -> level`` dict,
    looked up without POS). ``exclude`` holds normalized lemmas to drop
    AFTER analysis but BEFORE ranking -- the scorer itself is untouched.
    Returns ranked dicts.
    """
    lookup = cefr.lookup if isinstance(cefr, CefrProfile) else cefr.get
    excluded = {w.lower() for w in (exclude or set())}
    pos_counts = pos_counts or {}
    learner_order = CEFR_ORDER[learner_level]
    out: list[dict] = []
    for lemma, count in lemma_counts.items():
        if lemma in excluded:
            continue
        if count < min_count:
            continue
        zipf = float(zipf_frequency(lemma, "en"))
        upos = dominant_upos(pos_counts, lemma)
        level = lookup(lemma, upos) if isinstance(cefr, CefrProfile) \
            else lookup(lemma)
        if not is_eligible(level, zipf, learner_order, min_zipf):
            continue
        rarity, recurrence, score = score_lemma(count, zipf)
        forms = surface_forms.get(lemma, Counter())
        ordered_forms = sorted(forms, key=lambda w: (-forms[w], w))
        ctx = contexts.get(lemma, [])
        example = ctx[0][0] if ctx else ""
        timestamp = ctx[0][1] if ctx else ""
        out.append({
            "lemma": lemma,
            "surface_forms": ordered_forms,
            "count": count,
            "cefr": level or CEFR_UNKNOWN,
            "zipf": zipf,
            "rarity": rarity,
            "recurrence": recurrence,
            "score": score,
            "example": example,
            "timestamp": timestamp,
        })
    out.sort(key=lambda d: (-d["score"], -d["count"], d["lemma"]))
    return out


def suggest_excludes(ranked: list[dict], pos_counts: dict[str, Counter],
                     n: int = 30) -> list[str]:
    """Lemmas in the top-*n* that look like names or junk, for review.

    Flags UNKNOWN words whose dominant POS is PROPN (names the NER
    filter missed, e.g. ``roddy``) or whose Zipf is ~0 (typos, baby
    talk, joke compounds like ``henchfrog``). Returns lemmas only --
    the caller decides what to exclude.
    """
    out: list[str] = []
    for cand in ranked[:n]:
        if cand["cefr"] != CEFR_UNKNOWN:
            continue
        lemma = cand["lemma"]
        if dominant_upos(pos_counts, lemma) == "PROPN" or cand["zipf"] <= 0.5:
            out.append(lemma)
    return out


# ---------------------------------------------------------------------------
# Export: CSV
# ---------------------------------------------------------------------------

CSV_COLUMNS = ["rank", "lemma", "surface_forms", "count", "cefr", "zipf",
               "rarity", "recurrence", "score", "example", "timestamp"]


def write_csv(candidates: list[dict], path: Path) -> None:
    """Write ranked candidates with full score components for inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for i, c in enumerate(candidates, start=1):
            w.writerow({
                "rank": i,
                "lemma": c["lemma"],
                "surface_forms": ", ".join(c["surface_forms"]),
                "count": c["count"],
                "cefr": c["cefr"],
                "zipf": f"{c['zipf']:.2f}",
                "rarity": f"{c['rarity']:.2f}",
                "recurrence": f"{c['recurrence']:.2f}",
                "score": f"{c['score']:.2f}",
                "example": c["example"],
                "timestamp": c["timestamp"],
            })


# ---------------------------------------------------------------------------
# Export: Anki
# ---------------------------------------------------------------------------

def highlight_forms(sentence: str, forms: list[str]) -> str:
    """HTML-escape *sentence*, then <b>-highlight any of *forms* in it."""
    safe = html.escape(sentence)
    for form in sorted(set(forms), key=len, reverse=True):
        if not form:
            continue
        pattern = re.compile(rf"\b({re.escape(html.escape(form))})\b",
                             re.IGNORECASE)
        safe = pattern.sub(r"<b>\1</b>", safe)
    return safe


def deck_id_for(name: str) -> int:
    """Deterministic 31-bit deck id derived from the deck name."""
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % (2 ** 31 - 1000)) + 1000


def default_outdir(srt_path: Path) -> Path:
    """Base folder for per-input output folders.

    Normally ``<input-parent>/outputs``, but if the input already lives
    under a folder named ``outputs`` (e.g. a movie folder that was
    misplaced there), the nearest such ancestor is reused instead of
    nesting ``outputs/outputs``.
    """
    for parent in srt_path.resolve().parents:
        if parent.name == "outputs":
            return parent
    return srt_path.parent / "outputs"


def slugify_stem(name: str, max_len: int = 80) -> str:
    """Turn an SRT filename into a safe per-input folder name.

    Strips trailing .srt suffixes (``x.srt.srt``), keeps letters,
    digits, dots and dashes, maps everything else (spaces, brackets)
    to single underscores, truncates to *max_len*.
    """
    base = name
    while base.lower().endswith(".srt"):
        base = base[: -len(".srt")]
    slug = re.sub(r"[^A-Za-z0-9.\-]+", "_", base).strip("._")
    slug = re.sub(r"_+", "_", slug)
    slug = re.sub(r"_-", "-", slug)
    slug = re.sub(r"-_", "-", slug)
    slug = slug[:max_len].rstrip("_") or "subtitle"
    return slug


def build_deck(candidates: list[dict],
               contexts: dict[str, list[tuple[str, str]]],
               deck_name: str,
               definitions: dict[str, dict] | None = None) -> "genanki.Deck":
    """Build the Anki deck, keeping candidate order.

    ``definitions`` optionally maps lemma -> {"definition", "sense",
    "example"} (as produced by define.py). Cards without a definition
    render exactly as before.
    """
    model = genanki.Model(
        MODEL_ID,
        "Subtitle Vocabulary",
        fields=[
            {"name": "Word"},
            {"name": "Info"},
            {"name": "Context"},
        ],
        templates=[{
            "name": "Vocabulary",
            "qfmt": '<div class="word">{{Word}}</div>',
            "afmt": ("{{FrontSide}}<hr>"
                     '<div class="info">{{Info}}</div>'
                     '<div class="context">{{Context}}</div>'),
        }],
        css=(
            ".card{font-family:Arial,sans-serif;font-size:22px;text-align:center;"
            "color:#222;background:#fff;}"
            ".word{font-size:42px;font-weight:700;margin:40px 0;}"
            ".info{font-size:16px;margin-bottom:18px;color:#555;}"
            ".context{font-size:20px;line-height:1.5;text-align:left;}"
            ".defn{font-size:20px;margin:12px 0;text-align:left;}"
            ".sense{font-size:16px;color:#555;margin-bottom:12px;text-align:left;}"
        ),
    )
    deck = genanki.Deck(deck_id_for(deck_name), deck_name)
    definitions = definitions or {}
    for item in candidates:
        lemma = item["lemma"]
        lines = contexts.get(lemma, [])[:MAX_CONTEXTS]
        forms = item.get("surface_forms") or [lemma]
        parts = []
        defn = definitions.get(lemma) or {}
        if defn.get("definition"):
            parts.append(
                f'<div class="defn">{html.escape(defn["definition"])}</div>'
            )
            parts.append(
                f'<div class="sense">Used here as: '
                f'{html.escape(defn.get("sense") or "—")}<br>e.g. '
                f'{html.escape(defn.get("example") or "—")}</div>'
            )
            syns = defn.get("synonyms") or {}
            easier = syns.get("easier") if isinstance(syns, dict) else None
            harder = syns.get("harder") if isinstance(syns, dict) else None
            if easier or harder:
                parts.append(
                    f'<div class="sense">Synonyms: easier — '
                    f'{html.escape(easier or "—")} · harder — '
                    f'{html.escape(harder or "—")}</div>'
                )
            if defn.get("antonym") and defn.get("antonym") != "-":
                parts.append(
                    f'<div class="sense">Antonym: '
                    f'{html.escape(defn["antonym"])}</div>'
                )
        for sentence, ts in lines:
            parts.append(
                f"<div><small>{html.escape(ts)}</small> — "
                f"{highlight_forms(sentence, forms)}</div>"
            )
        context_html = "<br><br>".join(parts) if parts else "<i>No context.</i>"
        info = (f"CEFR: {html.escape(item['cefr'])} · "
                f"Zipf: {item['zipf']:.2f} · "
                f"Seen in this show: {item['count']}×")
        note = genanki.Note(
            model=model,
            fields=[html.escape(lemma), info, context_html],
        )
        deck.add_note(note)
    return deck


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build an Anki deck of the most useful vocabulary to learn "
            "BEFORE watching a specific movie/show.\n\n"
            "Baseline ranker (no ML): rarity (wordfreq Zipf) x "
            "recurrence (log1p count in this file).\n\n"
            "--level means 'show words AT OR ABOVE this level' "
            "(B2 keeps B2+C1+C2, C1 keeps C1+C2). "
            "Words with a known CEFR at or below your level are excluded. "
            "CEFR may come from a simple TSV or from CEFR-J/Octanove "
            "profile CSVs (POS-aware). Words missing from all CEFR files "
            "are labelled UNKNOWN (Zipf frequency is NOT a CEFR "
            "prediction) and gated by --min-zipf."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("srt_file", type=Path, help="Input .srt subtitle file")
    p.add_argument("--level", choices=sorted(CEFR_ORDER, key=CEFR_ORDER.get),
                   default="C1",
                   help="Learner level; keep words AT OR ABOVE this "
                        "(B2 keeps B2+C1+C2; C1 keeps C1+C2). Default: C1.")
    p.add_argument("--top", type=int, default=50,
                   help="Max cards/rows to keep; 0 = no limit, keep all "
                        "candidates (default: 50).")
    p.add_argument("--output", type=Path, default=None,
                   help="Anki .apkg path "
                        "(default: <outdir>/<slug>/vocab.apkg).")
    p.add_argument("--csv", type=Path, default=None,
                   help="CSV path (default: <outdir>/<slug>/vocab.csv).")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Base folder for per-input output folders "
                        "(default: outputs/ next to the input).")
    p.add_argument("-o", dest="output_alias", type=Path, default=None,
                   help="Alias for --output.")
    p.add_argument("--cefr", type=Path, action="append", default=None,
                   help="CEFR file, repeatable. Simple TSV "
                        "(lemma<TAB>cefr) or CEFR-J/Octanove profile CSV "
                        "(headword,pos,CEFR). Later files win on conflicts. "
                        "E.g. --cefr ../olp-en-cefrj/cefrj-vocabulary-profile-1.5.csv "
                        "--cefr ../olp-en-cefrj/octanove-vocabulary-profile-c1c2-1.0.csv")
    p.add_argument("--exclude", type=str, default=None,
                   help="Comma-separated lemmas to drop after analysis, "
                        "before ranking (case-insensitive, scorer untouched). "
                        'E.g. --exclude "mulan,hua,qiang,tung,xianniang"')
    p.add_argument("--min-count", type=int, default=1,
                   help="Minimum occurrences in this file (default: 1).")
    p.add_argument("--min-zipf", type=float, default=4.5,
                   help="Zipf fallback gate for CEFR-UNKNOWN words: keep "
                        "only words with zipf <= this (lower = rarer). "
                        "<=0 excludes all UNKNOWN words (default: 4.5).")
    return p.parse_args(argv)


def resolve_defaults(args: argparse.Namespace) -> tuple[Path, Path]:
    """Defaults live in outputs/<slug>/: short stable names, no clutter.

    Explicit --output/--csv always win. Otherwise
    ``<outdir or <input-parent>/outputs>/<slug>/vocab.{apkg,csv}``.
    """
    explicit_output = args.output or getattr(args, "output_alias", None)
    if explicit_output or args.csv:
        # Partial overrides: fill the missing half next to the given one.
        if explicit_output and args.csv:
            return explicit_output, args.csv
        anchor = explicit_output or args.csv
        sibling = (anchor.with_suffix(".csv") if explicit_output
                   else anchor.with_suffix(".apkg"))
        return (explicit_output or sibling), (args.csv or sibling)
    outdir = args.outdir or default_outdir(args.srt_file)
    folder = outdir / slugify_stem(args.srt_file.name)
    return folder / "vocab.apkg", folder / "vocab.csv"


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.srt_file.exists():
        raise SystemExit(f"Input not found: {args.srt_file}")
    if args.top < 0:
        raise SystemExit("--top must be >= 0 (0 = no limit)")

    output_path, csv_path = resolve_defaults(args)
    cefr = load_cefr_profiles(args.cefr)
    excluded = parse_exclude(args.exclude)

    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser"])
    except OSError:
        raise SystemExit(
            "spaCy model 'en_core_web_sm' not found. Run:\n"
            "  python -m spacy download en_core_web_sm"
        )

    subtitles = parse_srt_file(args.srt_file)
    lemma_counts, surface_forms, contexts, pos_counts = analyze_subtitles(
        subtitles, nlp)
    ranked_all = build_candidates(lemma_counts, surface_forms, contexts,
                                  cefr, args.level, args.min_zipf,
                                  args.min_count,
                                  pos_counts=pos_counts, exclude=excluded)
    selected = ranked_all[:args.top] if args.top > 0 else ranked_all

    write_csv(selected, csv_path)
    deck_name = output_path.stem.replace("_", " ").replace("-", " ").title()
    deck = build_deck(selected, contexts, deck_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    genanki.Package(deck).write_to_file(str(output_path))

    print(f"Subtitle file: {args.srt_file}")
    print(f"Subtitles:     {len(subtitles)}")
    print(f"Unique lemmas: {len(lemma_counts)}")
    if excluded:
        print(f"Excluded:      {len(excluded)} ({', '.join(sorted(excluded))})")
    if cefr:
        print(f"CEFR entries:  {len(cefr)} lemmas")
    print(f"Candidates:    {len(ranked_all)}")
    print(f"Selected:      {len(selected)}")
    print(f"Anki deck:     {output_path}")
    print(f"CSV:           {csv_path}")
    if selected:
        print("\nTop vocabulary:")
        for i, c in enumerate(selected[:min(10, len(selected))], start=1):
            print(f"{i}. {c['lemma']} "
                  f"(CEFR {c['cefr']}, count {c['count']}, "
                  f"zipf {c['zipf']:.2f}, score {c['score']:.2f})")
    if not args.cefr:
        print("\nNote: no --cefr file supplied; CEFR-UNKNOWN words gated "
              f"by --min-zipf={args.min_zipf} (Zipf is not a CEFR prediction).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
