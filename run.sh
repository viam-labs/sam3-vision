#!/bin/sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Reinstall when requirements.txt changes, since reloads keep the existing .venv.
if ! cmp -s "$SCRIPT_DIR/requirements.txt" "$SCRIPT_DIR/.venv/.requirements.txt"; then
    "$SCRIPT_DIR/setup.sh"
fi
cd "$SCRIPT_DIR/src"
exec "$SCRIPT_DIR/.venv/bin/python" main.py "$@"
