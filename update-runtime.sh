#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  AI Horde Worker - Install / Update"
echo "============================================"
echo ""

# Install uv if not present
if [ ! -f "$SCRIPT_DIR/bin/uv" ]; then
    echo "Downloading uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCRIPT_DIR/bin" sh
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Failed to download uv. Check your internet connection."
        exit 1
    fi
    echo "Done."
    echo ""
fi

echo "Installing dependencies for GPU backend: cu128"
echo "(This may take a few minutes on first run...)"
echo ""
"$SCRIPT_DIR/bin/uv" sync --locked --extra cu128
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Installation failed."
    echo "  - Try deleting the .venv folder and running this script again."
    echo "  - If the problem persists, ask for help in #local-workers on Discord."
    exit 1
fi

echo ""
echo "============================================"
echo "  Installation complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Edit bridgeData.yaml with your API key and worker name"
echo "  2. Run ./horde-bridge.sh to start the worker"
echo "     (or ./horde-worker.sh for the interactive launcher)"
echo ""
