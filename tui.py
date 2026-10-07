#!/usr/bin/env python3
"""Interactive Rich TUI for the subtitle-vocab project.

Flow: pick an .srt -> tune a few settings -> rank (table + summary) ->
optionally fetch LLM definitions (progress bar) -> outputs folder.

Everything here calls subtitle_vocab / define functions directly (no
subprocesses). Run from the project folder:

    python tui.py [optional/path/to/movie.srt]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

import define as define_mod
import subtitle_vocab as sv

console = Console()

CEFRJ_NAME = "cefrj-vocabulary-profile-1.5.csv"
OCTANOVE_NAME = "octanove-vocabulary-profile-c1c2-1.0.csv"


# ---------------------------------------------------------------------------
# Discovery helpers (pure; tested)
# ---------------------------------------------------------------------------

def find_srt_files(roots: list[Path]) -> list[Path]:
    """All *.srt files under *roots* (non-recursive), deduplicated, sorted."""
    found: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.glob("*.srt")):
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                found.append(p)
    return found


def detect_cefr_files(hints: list[Path]) -> list[Path]:
    """CEFR-J + Octanove CSVs if an olp-en-cefrj dir sits near *hints*."""
    for hint in hints:
        base = hint if hint.is_dir() else hint.parent
        for folder in (base, base.parent):
            pair = [folder / "olp-en-cefrj" / CEFRJ_NAME,
                    folder / "olp-en-cefrj" / OCTANOVE_NAME]
            if all(p.exists() for p in pair):
                return pair
    return []


def default_exclude_suggestion() -> str:
    return ""


# ---------------------------------------------------------------------------
# Interactive steps
# ---------------------------------------------------------------------------

def pick_srt(cli_arg: str | None) -> Path:
    if cli_arg:
        p = Path(cli_arg).expanduser()
        if not p.exists():
            console.print(f"[red]Not found: {p}[/red]")
            raise SystemExit(1)
        return p
    here = Path.cwd()
    options = find_srt_files([here, here.parent])
    if options:
        table = Table(title="Subtitle files found", show_header=True)
        table.add_column("#", justify="right")
        table.add_column("File")
        for i, p in enumerate(options, start=1):
            table.add_row(str(i), str(p))
        table.add_row("0", "[italic]type a different path…[/italic]")
        console.print(table)
        choice = IntPrompt.ask("Pick a file", default=1)
        if choice == 0:
            return Path(Prompt.ask("Path to .srt")).expanduser()
        if 1 <= choice <= len(options):
            return options[choice - 1]
        console.print("[red]Out of range.[/red]")
        raise SystemExit(1)
    return Path(Prompt.ask("Path to .srt")).expanduser()


def ask_settings(srt: Path) -> dict:
    level = Prompt.ask("Learner level (words ABOVE this are kept)",
                       choices=list(sv.CEFR_ORDER), default="B2")
    top = IntPrompt.ask("Max words", default=50)
    min_count = IntPrompt.ask("Minimum occurrences", default=1)
    min_zipf = float(Prompt.ask("Zipf fallback for UNKNOWN words "
                                "(0 disables)", default="4.5"))
    exclude = Prompt.ask("Exclude lemmas (comma-separated, empty for none)",
                         default=default_exclude_suggestion())
    detected = detect_cefr_files([Path.cwd(), srt])
    if detected:
        console.print(f"[green]CEFR data detected:[/green] "
                      + ", ".join(p.name for p in detected))
        use = Confirm.ask("Use it?", default=True)
        cefr = detected if use else []
    else:
        console.print("[yellow]No CEFR-J/Octanove data detected nearby.[/yellow]")
        custom = Prompt.ask("Comma-separated --cefr paths (empty for none)",
                            default="")
        cefr = [Path(p.strip()) for p in custom.split(",") if p.strip()]
    return {"level": level, "top": top, "min_count": min_count,
            "min_zipf": min_zipf, "exclude": exclude, "cefr": cefr}


def load_nlp():
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["parser"])
    except OSError:
        console.print("[red]spaCy model 'en_core_web_sm' not found. Run:[/red]")
        console.print("  python -m spacy download en_core_web_sm")
        raise SystemExit(1)


def show_top_table(selected: list[dict], n: int = 10) -> None:
    table = Table(title=f"Top {min(n, len(selected))} words")
    for col in ("#", "lemma", "CEFR", "×", "zipf", "score"):
        table.add_column(col, justify="right" if col != "lemma" else "left")
    for i, c in enumerate(selected[:n], start=1):
        table.add_row(str(i), c["lemma"], c["cefr"], str(c["count"]),
                      f"{c['zipf']:.2f}", f"{c['score']:.2f}")
    console.print(table)


def review_excludes(ranked: list[dict], pos_counts: dict,
                    excluded: set[str]) -> set[str]:
    """Let the user drop names/junk, re-rank until they're happy."""
    while True:
        suggestions = [w for w in
                       sv.suggest_excludes(ranked, pos_counts)
                       if w not in excluded]
        if suggestions:
            console.print("[yellow]Likely names/junk:[/yellow] "
                          + ", ".join(suggestions))
            if Confirm.ask("Exclude all of them?", default=True):
                excluded.update(suggestions)
                return excluded
        extra = Prompt.ask(
            "Exclude more (comma-separated, ranks like 3 or 1,4, "
            "empty when happy)", default="").strip()
        if not extra:
            return excluded
        ranked_lemmas = [c["lemma"] for c in ranked]
        for token in extra.split(","):
            token = token.strip().lower()
            if not token:
                continue
            if token.isdigit() and 1 <= int(token) <= len(ranked_lemmas):
                excluded.add(ranked_lemmas[int(token) - 1])
            else:
                excluded.add(token)


def run_ranking(srt: Path, cfg: dict) -> dict:
    import genanki
    with console.status("[bold green]Analysing subtitles…[/bold green]"):
        nlp = load_nlp()
        profile = sv.load_cefr_profiles(cfg["cefr"])
        subtitles = sv.parse_srt_file(srt)
        counts, forms, contexts, pos = sv.analyze_subtitles(subtitles, nlp)

    def _build(excluded: set[str]) -> list[dict]:
        return sv.build_candidates(
            counts, forms, contexts, profile, cfg["level"], cfg["min_zipf"],
            cfg["min_count"], pos_counts=pos, exclude=excluded)

    excluded = sv.parse_exclude(cfg["exclude"])
    while True:
        ranked = _build(excluded)
        selected = ranked[: cfg["top"]]
        console.print(Panel(
            f"Subtitles {len(subtitles)} · lemmas {len(counts)} · "
            f"candidates {len(ranked)} · selected {len(selected)}"
            + (f" · excluded {len(excluded)}" if excluded else ""),
            title="Ranking"))
        if not selected:
            console.print("[red]Nothing selected. Loosen excludes/filters.[/red]")
        else:
            show_top_table(selected)
        before = set(excluded)
        excluded = review_excludes(ranked, pos, excluded)
        if set(excluded) == before:
            break
        ranked = _build(excluded)
        selected = ranked[: cfg["top"]]
        show_top_table(selected)

    outdir = sv.default_outdir(srt) / sv.slugify_stem(srt.name)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path, apkg_path = outdir / "vocab.csv", outdir / "vocab.apkg"
    sv.write_csv(selected, csv_path)
    deck_name = f"{srt.stem} Vocab"
    genanki.Package(sv.build_deck(selected, contexts, deck_name)) \
        .write_to_file(str(apkg_path))
    return {"selected": selected, "contexts_full": contexts,
            "csv_path": csv_path, "outdir": outdir, "deck_name": deck_name}


def run_definitions(csv_path: Path, srt: Path, cfg: dict,
                    outdir: Path) -> tuple[int, int, Path]:
    base_url, key_env, need_key = define_mod.PROVIDERS["groq"]
    model = Prompt.ask("Model", default=define_mod.DEFAULT_MODEL)
    api_key = os.environ.get(key_env)
    if not api_key:
        console.print(f"[yellow]${key_env} is not set.[/yellow] "
                      f"Tip: save it in a [bold].env[/bold] file "
                      f"(GROQ_API_KEY=...) to skip this step.")
        pasted = Prompt.ask("Paste a Groq key (session-only, never stored, "
                            "empty for context-only cards)",
                            password=True, default="")
        api_key = pasted.strip() or None
        if api_key:
            os.environ[key_env] = api_key
    rows = define_mod.load_csv_rows(csv_path)
    if not rows:
        console.print("[red]CSV is empty.[/red]")
        return 0, 0, outdir / "vocab.md"
    cache_path = outdir / "definitions.json"
    cache = define_mod.load_cache(cache_path)
    full_contexts = define_mod.analyse_srt_for_contexts(srt)
    failed: list[str] = []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                   BarColumn(), TextColumn("{task.completed}/{task.total}"),
                   console=console) as progress:
        task = progress.add_task("Defining…", total=len(rows))

        def _tick(lemma: str, ok: bool):
            if not ok:
                failed.append(lemma)
            progress.update(task, advance=1, description=f"Defining {lemma}")

        definitions, cache = define_mod.define_all(
            rows, cfg["level"], full_contexts, cache, base_url, api_key,
            model, "groq", verbose=False, on_each=_tick,
            rate_limiter=define_mod.RateLimiter())
    define_mod.save_cache(cache_path, cache)
    # Rebuild deck + study sheet with definitions.
    import genanki
    candidates = define_mod.rows_to_candidates(rows)
    contexts = {c["lemma"]: full_contexts.get(c["lemma"], [])[:3]
                for c in candidates}
    for c in candidates:  # keep CSV single-example fallback
        if not contexts[c["lemma"]] and c["example"]:
            contexts[c["lemma"]] = [(c["example"], c["timestamp"])]
    deck_path = outdir / "vocab_defined.apkg"
    deck = sv.build_deck(candidates, contexts, f"{srt.stem} Vocab",
                         definitions=definitions)
    genanki.Package(deck).write_to_file(str(deck_path))
    md_path = outdir / "vocab.md"
    define_mod.write_markdown(candidates, contexts, definitions,
                              cfg["level"], f"{srt.stem} Vocab", md_path)
    if failed:
        console.print(f"[yellow]No definition for: "
                      f"{', '.join(failed)}[/yellow]")
    return len(definitions), len(rows), md_path


def show_outputs(outdir: Path) -> None:
    table = Table(title="Outputs")
    table.add_column("File")
    table.add_column("Size", justify="right")
    for p in sorted(outdir.iterdir()):
        if p.is_file():
            table.add_row(str(p), f"{p.stat().st_size / 1024:.1f} KB")
    console.print(table)


def main(argv=None) -> int:
    args = (argv if argv is not None else sys.argv[1:])
    if "--help" in args or "-h" in args:
        console.print("Usage: python tui.py [movie.srt]")
        return 0
    if not sys.stdin.isatty():
        console.print("[red]tui.py needs an interactive terminal.[/red]")
        return 1
    console.rule("[bold]Subtitle vocab → Anki[/bold]")
    srt = pick_srt(args[0] if args else None)
    cfg = ask_settings(srt)
    ranked = run_ranking(srt, cfg)
    if ranked["selected"] and Confirm.ask(
            "Fetch LLM definitions for these words?", default=True):
        n, total, md = run_definitions(ranked["csv_path"], srt, cfg,
                                       ranked["outdir"])
        console.print(f"[green]Defined {n}/{total}.[/green] "
                      f"Study sheet: {md}")
    show_outputs(ranked["outdir"])
    console.print("[bold green]Done.[/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
