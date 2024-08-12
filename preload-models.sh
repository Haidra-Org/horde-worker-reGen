#!/bin/bash
# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if "$SCRIPT_DIR/runtime.sh" python -s "$SCRIPT_DIR/download_models.py"; then
    echo "Model Download OK."
else
    echo "download_models.py exited with error code."
fi
