"""Tests: output paths (outputs/<slug>/ structure)."""
from pathlib import Path

from define import define_default_paths
from subtitle_vocab import (
    default_outdir,
    parse_args,
    resolve_defaults,
    slugify_stem,
)


def test_slugify_strips_srt_and_spaces():
    assert slugify_stem("movie.srt") == "movie"
    assert slugify_stem("x.srt.srt") == "x"
    assert " " not in slugify_stem("Mulan (1998)-en.srt")
    assert slugify_stem("Mulan (1998)-en.srt") == "Mulan_1998-en"
    slug = slugify_stem("Mulan.2020.1080p.BluRay.x265-YAWNTiC_eng SDH.srt.srt")
    assert slug.endswith("SDH") and " " not in slug and ".srt" not in slug


def test_resolve_defaults_new_structure(tmp_path):
    args = parse_args([str(tmp_path / "movie.srt")])
    apkg, csv = resolve_defaults(args)
    assert apkg == tmp_path / "outputs" / "movie" / "vocab.apkg"
    assert csv == tmp_path / "outputs" / "movie" / "vocab.csv"


def test_resolve_defaults_explicit_wins(tmp_path):
    args = parse_args([str(tmp_path / "movie.srt"), "--output",
                       str(tmp_path / "d.apkg"), "--csv",
                       str(tmp_path / "d.csv")])
    apkg, csv = resolve_defaults(args)
    assert apkg == tmp_path / "d.apkg"
    assert csv == tmp_path / "d.csv"


def test_resolve_defaults_outdir_override(tmp_path):
    args = parse_args([str(tmp_path / "movie.srt"), "--outdir",
                       str(tmp_path / "base")])
    apkg, _ = resolve_defaults(args)
    assert apkg == tmp_path / "base" / "movie" / "vocab.apkg"


def test_define_default_paths():
    c, a, m = define_default_paths(Path("outputs/Foo/vocab.csv"))
    assert (c.name, a.name, m.name) == ("definitions.json",
                                        "vocab_defined.apkg", "vocab.md")
    c, a, m = define_default_paths(Path("x/my.csv"))
    assert (c.name, a.name, m.name) == ("my_definitions.json",
                                        "my_defined.apkg", "my.md")


def test_default_outdir_normal_case(tmp_path):
    from subtitle_vocab import default_outdir
    assert default_outdir(tmp_path / "movie.srt") == tmp_path / "outputs"


def test_default_outdir_reuses_existing_outputs(tmp_path):
    from subtitle_vocab import default_outdir
    nested = tmp_path / "outputs" / "Some.Movie.srt" / "Some.Movie.srt"
    assert default_outdir(nested) == tmp_path / "outputs"
