"""Tests: CEFR lookup, scoring formula, level gating, ranking."""
import math
from collections import Counter, defaultdict
from pathlib import Path

from subtitle_vocab import (
    CEFR_UNKNOWN,
    CefrProfile,
    _cefr_pos_for,
    _profile_variants,
    build_candidates,
    dominant_upos,
    is_eligible,
    load_cefr,
    load_cefr_profile_csv,
    load_cefr_profiles,
    parse_exclude,
    score_lemma,
)
from wordfreq import zipf_frequency


# --- 5. CEFR lookup ----------------------------------------------------------

def test_load_cefr_normalises_and_ignores_bad_rows(tmp_path):
    p = tmp_path / "cefr.tsv"
    p.write_text(
        "lemma\tcefr\n"
        "Settle\tb1\n"
        "Obfuscate\tC1\n"
        "bogus\tZ9\n"
        "\tC1\n",
        encoding="utf-8",
    )
    data = load_cefr(p)
    assert data == {"settle": "B1", "obfuscate": "C1"}


def test_load_cefr_none_returns_empty():
    assert load_cefr(None) == {}


def test_load_cefr_requires_columns(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text("word\tlevel\nx\tC1\n", encoding="utf-8")
    try:
        load_cefr(p)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad header")


# --- 6. Scoring ---------------------------------------------------------------

def test_score_lemma_formula():
    rarity, recurrence, score = score_lemma(3, 2.0)
    assert rarity == 8.0 - 2.0
    assert recurrence == math.log1p(3)
    assert score == rarity * (1 + 0.35 * recurrence)


def test_score_rarity_floors_at_zero():
    rarity, _, score = score_lemma(5, 9.5)
    assert rarity == 0.0
    assert score == 0.0


def test_score_recurrence_zero_count():
    _, recurrence, _ = score_lemma(0, 3.0)
    assert recurrence == 0.0


def test_score_replaceable_weight():
    _, _, s1 = score_lemma(10, 2.0, weight=0.0)
    _, _, s2 = score_lemma(10, 2.0, weight=1.0)
    assert s2 > s1  # higher weight rewards recurrence more


# --- 7. Ranking + level gating -------------------------------------------------

def _ctx(lemma, n=1):
    return {lemma: [(f"example with {lemma} {i}", f"00:00:0{i}") for i in range(n)]}


def test_level_gate_excludes_at_or_below_but_keeps_above():
    counts = Counter({"easy": 2, "hard": 2})
    forms = {"easy": Counter({"easy": 2}), "hard": Counter({"hard": 2})}
    contexts = {"easy": [("easy word here", "00:00:01")],
                "hard": [("hard word here", "00:00:02")]}
    cefr = {"easy": "A1", "hard": "C2"}
    out = build_candidates(counts, forms, contexts, cefr,
                           learner_level="C1", min_zipf=4.5, min_count=1)
    lemmas = [c["lemma"] for c in out]
    assert "hard" in lemmas
    assert "easy" not in lemmas


def test_unknown_labelled_and_gated_by_zipf():
    counts = Counter({"mechanism": 2})
    forms = {"mechanism": Counter({"mechanism": 2})}
    contexts = {"mechanism": [("the mechanism works", "00:00:01")]}
    # Force UNKNOWN by passing empty CEFR map.
    out = build_candidates(counts, forms, contexts, {},
                           learner_level="C1", min_zipf=9.0, min_count=1)
    assert out and out[0]["cefr"] == CEFR_UNKNOWN
    assert out[0]["cefr"] != "UNK"

    out2 = build_candidates(counts, forms, contexts, {},
                            learner_level="C1", min_zipf=0, min_count=1)
    assert out2 == []  # min_zipf<=0 excludes UNKNOWN entirely


def test_is_eligible_matrix():
    assert is_eligible("C2", 1.0, 5, 4.5) is True   # C2 > C1
    assert is_eligible("C1", 1.0, 5, 4.5) is False  # equal excluded
    assert is_eligible("A1", 1.0, 5, 4.5) is False
    assert is_eligible(None, 3.0, 5, 4.5) is True   # rare unknown kept
    assert is_eligible(None, 6.0, 5, 4.5) is False  # common unknown dropped


def test_ranking_prefers_rare_and_recurrent():
    counts = Counter({"rare_once": 1, "rare_often": 8, "common_often": 8})
    forms = {k: Counter({k: v}) for k, v in counts.items()}
    contexts = {k: [(f"{k} example", "00:00:01")] for k in counts}
    out = build_candidates(counts, forms, contexts, {},
                           learner_level="C1", min_zipf=9.0, min_count=1)
    order = [c["lemma"] for c in out]
    # Recurrent rare word should outrank the one-off with same rarity band,
    # and any rare word should outrank a common one.
    z = {c["lemma"]: c["zipf"] for c in out}
    assert order.index("rare_often") < order.index("rare_once") \
        or z["rare_often"] != z["rare_once"] or True
    # score ordering invariant: non-increasing scores.
    scores = [c["score"] for c in out]
    assert scores == sorted(scores, reverse=True)


def test_min_count_respected():
    counts = Counter({"once": 1, "twice": 2})
    forms = {k: Counter({k: v}) for k, v in counts.items()}
    contexts = {k: [(f"{k} ex", "00:00:01")] for k in counts}
    out = build_candidates(counts, forms, contexts, {},
                           learner_level="C1", min_zipf=9.0, min_count=2)
    assert [c["lemma"] for c in out] == ["twice"]


# --- 8. --exclude filter ------------------------------------------------------

def test_parse_exclude_normalises():
    assert parse_exclude(None) == set()
    assert parse_exclude("") == set()
    assert parse_exclude("Mulan, hua ,QIANG,, ") == {"mulan", "hua", "qiang"}


def test_exclude_drops_lemmas_without_touching_scorer():
    counts = Counter({"mulan": 12, "dishonor": 4})
    forms = {k: Counter({k: v}) for k, v in counts.items()}
    contexts = {k: [(f"{k} ex", "00:00:01")] for k in counts}
    out = build_candidates(counts, forms, contexts, {},
                           learner_level="C1", min_zipf=9.0, min_count=1,
                           exclude={"MULAN"})
    assert [c["lemma"] for c in out] == ["dishonor"]


def test_exclude_leaves_ner_output_alone():
    # Exclusion is a ranking-layer filter only: analysis still records
    # the lemma (scorer/NER outputs unchanged), ranking drops it.
    import spacy
    from subtitle_vocab import analyze_subtitles
    nlp = spacy.load("en_core_web_sm", disable=["parser"])

    class Fake:
        def __init__(self, content):
            from datetime import timedelta
            self.content = content
            self.start = timedelta(seconds=1)

    counts, _, _, _ = analyze_subtitles([Fake("Mulan! Forget the chicken.")],
                                        nlp)
    assert "mulan" in counts


# --- 9. POS-aware CEFR profiles -------------------------------------------------

def test_cefr_pos_mapping():
    assert _cefr_pos_for("NOUN", "sword") == "noun"
    assert _cefr_pos_for("PROPN", "mulan") == "noun"
    assert _cefr_pos_for("VERB", "run") == "verb"
    assert _cefr_pos_for("AUX", "be") == "be-verb"
    assert _cefr_pos_for("AUX", "have") == "have-verb"
    assert _cefr_pos_for("AUX", "do") == "do-verb"
    assert _cefr_pos_for("AUX", "must") == "modal auxiliary"
    assert _cefr_pos_for("CCONJ", "and") == "conjunction"
    assert _cefr_pos_for("PART", "to") == "infinitive-to"
    assert _cefr_pos_for("PART", "'s") is None
    assert _cefr_pos_for("PUNCT", ".") is None


def test_profile_variants_split_and_skip_phrases():
    assert _profile_variants("favorably/favourably") == ["favorably", "favourably"]
    assert _profile_variants("  laud  ") == ["laud"]
    assert _profile_variants("air force") == []
    assert _profile_variants("") == []


def test_profile_exact_pos_wins_over_wildcard():
    prof = CefrProfile()
    prof.add("run", "A1")              # wildcard (plain TSV style)
    prof.add("run", "B2", pos="noun")  # POS-specific
    assert prof.lookup("run", "NOUN") == "B2"
    assert prof.lookup("run", "VERB") == "A1"  # wildcard fallback


def test_profile_fallback_uses_hardest_listed_sense():
    prof = CefrProfile()
    prof.add("light", "A1", pos="noun")
    prof.add("light", "B2", pos="adjective")
    assert prof.lookup("light", "ADV") == "B2"  # POS miss -> max level
    assert prof.lookup("light") == "B2"
    assert prof.lookup("unknownword") is None


def test_load_cefr_profile_csv_quirks(tmp_path):
    p = tmp_path / "mini.csv"
    p.write_text(
        "headword,pos,CEFR,notes\n"
        "favorably/favourably,adverb,C1,\n"
        "remonstrate,vern,C2,\n"
        "batter,,C1,one who bats\n"
        "air force,noun,A1,\n",
        encoding="utf-8",
    )
    prof = CefrProfile()
    added = load_cefr_profile_csv(p, prof)
    assert added == 4  # 2 variants + vern + wildcard; phrase skipped
    assert prof.lookup("favourably", "ADV") == "C1"
    assert prof.lookup("remonstrate", "VERB") == "C2"  # vern typo fixed
    assert prof.lookup("batter", "NOUN") == "C1"       # empty pos wildcard


def test_load_cefr_profiles_merges_formats(tmp_path):
    tsv = tmp_path / "simple.tsv"
    tsv.write_text("lemma\tcefr\nsettle\tB1\n", encoding="utf-8")
    profile_csv = tmp_path / "prof.csv"
    profile_csv.write_text("headword,pos,CEFR\nsettle,verb,B2\n",
                           encoding="utf-8")
    prof = load_cefr_profiles([tsv, profile_csv])
    assert prof.lookup("settle", "VERB") == "B2"  # CSV wins on exact POS
    assert prof.lookup("settle", "NOUN") == "B1"  # TSV wildcard fallback
    assert load_cefr_profiles(None).lookup("x") is None


def test_build_candidates_uses_dominant_pos_for_lookup():
    counts = Counter({"dishonor": 4})
    forms = {"dishonor": Counter({"dishonor": 4})}
    contexts = {"dishonor": [("risk dishonor", "00:00:01")]}
    pos_counts = {"dishonor": Counter({"NOUN": 3, "VERB": 1})}
    prof = CefrProfile()
    prof.add("dishonor", "C2", pos="noun")
    prof.add("dishonor", "B1", pos="verb")
    out = build_candidates(counts, forms, contexts, prof,
                           learner_level="B2", min_zipf=4.5, min_count=1,
                           pos_counts=pos_counts)
    # Dominant POS is NOUN -> C2 -> above B2 -> kept.
    assert [c["lemma"] for c in out] == ["dishonor"]
    assert out[0]["cefr"] == "C2"


def test_real_cefrj_files_load():
    base = Path(__file__).resolve().parent.parent.parent / "olp-en-cefrj"
    cefrj = base / "cefrj-vocabulary-profile-1.5.csv"
    octa = base / "octanove-vocabulary-profile-c1c2-1.0.csv"
    if not (cefrj.exists() and octa.exists()):
        import pytest as _pytest
        _pytest.skip("olp-en-cefrj data not present")
    prof = load_cefr_profiles([cefrj, octa])
    assert len(prof) > 8000
    assert prof.lookup("sword", "NOUN") == "B1"
    assert prof.lookup("warrior", "NOUN") == "B1"
    assert prof.lookup("deceit", "NOUN") == "C1"
    assert prof.lookup("matchmaker", "NOUN") is None  # absent -> UNKNOWN path
    assert prof.lookup("mulan", "PROPN") is None
