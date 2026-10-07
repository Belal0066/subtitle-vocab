"""Tests: TUI discovery helpers (interactive steps are not tested)."""
from pathlib import Path

from tui import detect_cefr_files, find_srt_files


def test_find_srt_files_sorted_deduped(tmp_path):
    (tmp_path / "b.srt").write_text("x")
    (tmp_path / "a.srt").write_text("x")
    (tmp_path / "note.txt").write_text("x")
    out = find_srt_files([tmp_path])
    assert [p.name for p in out] == ["a.srt", "b.srt"]


def test_find_srt_files_missing_root_ignored(tmp_path):
    assert find_srt_files([tmp_path / "nope"]) == []


def test_detect_cefr_files(tmp_path):
    data = tmp_path / "olp-en-cefrj"
    data.mkdir()
    (data / "cefrj-vocabulary-profile-1.5.csv").write_text("h")
    (data / "octanove-vocabulary-profile-c1c2-1.0.csv").write_text("h")
    found = detect_cefr_files([tmp_path])
    assert len(found) == 2
    assert found[0].name.startswith("cefrj")
    # A tree with no olp-en-cefrj anywhere above it finds nothing.
    assert detect_cefr_files([Path("/nonexistent-dir-xyz")]) == []


def test_detect_cefr_files_partial_pair_rejected(tmp_path):
    data = tmp_path / "olp-en-cefrj"
    data.mkdir()
    (data / "cefrj-vocabulary-profile-1.5.csv").write_text("h")
    assert detect_cefr_files([tmp_path]) == []
