#!/usr/bin/env bash
# Zero-friction launcher: ./run.sh [movie.srt]
# Uses the project .venv (auto-runs setup.sh on first launch),
# picks up GROQ_API_KEY from .env if present, then starts the TUI.
# Your shell's working directory is left untouched, so relative
# subtitle paths keep working from wherever you call this.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional convenience: GROQ_API_KEY=... in a .env file next to this script.
# Only that single variable is read (the file is never executed).
if [ -f "$HERE/.env" ]; then
    KEY="$(grep -E '^[[:space:]]*GROQ_API_KEY=' "$HERE/.env" | tail -n 1 \
        | cut -d '=' -f 2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
        -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
    if [ -n "$KEY" ] && [ -z "${GROQ_API_KEY:-}" ]; then
        export GROQ_API_KEY="$KEY"
    fi
fi

if [ ! -x "$HERE/.venv/bin/python" ]; then
    echo "[run] first launch: setting up the environment (one time, a few minutes) ..."
    bash "$HERE/setup.sh"
fi

exec "$HERE/.venv/bin/python" "$HERE/tui.py" "$@"
