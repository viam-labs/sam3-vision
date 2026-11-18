#!/bin/sh
set -e
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
if [ ! -d .venv ]; then
    "$PYTHON" -m venv .venv
fi
.venv/bin/pip install -q -r requirements.txt
