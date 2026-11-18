#!/bin/sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    "$SCRIPT_DIR/setup.sh"
fi
cd "$SCRIPT_DIR/src"
exec "$SCRIPT_DIR/.venv/bin/python" main.py "$@"
