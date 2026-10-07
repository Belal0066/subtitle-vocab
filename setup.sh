#!/usr/bin/env bash
# One-time (or repeat-safe) environment setup for subtitle-vocab.
# Creates .venv/, installs requirements, fetches the spaCy model.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"

if [ ! -x "$VENV/bin/python" ]; then
    echo "[setup] creating virtualenv at $VENV ..."
    python3 -m venv "$VENV"
fi

echo "[setup] installing requirements ..."
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$HERE/requirements.txt"

echo "[setup] fetching spaCy English model ..."
"$VENV/bin/python" -m spacy download en_core_web_sm

echo "[setup] done. Launch the TUI with: ./run.sh"
