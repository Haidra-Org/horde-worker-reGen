#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    "$SCRIPT_DIR/update-runtime.sh"
fi
"$SCRIPT_DIR/bin/uv" run --no-sync "$@"
