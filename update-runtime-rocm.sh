#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  AI Horde Worker - Install / Update (ROCm)"
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

# Determine if the user has a flash attention supported card.
SUPPORTED_CARD=$(rocminfo | grep -c -e gfx1100 -e gfx1101 -e gfx1102)
if [ "$SUPPORTED_CARD" -gt 0 ]; then export FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:=TRUE}"; fi

echo "Installing dependencies for GPU backend: rocm"
echo "(This may take a few minutes on first run...)"
echo ""
"$SCRIPT_DIR/bin/uv" sync --locked --extra rocm
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Installation failed."
    echo "  - Try deleting the .venv folder and running this script again."
    echo "  - If the problem persists, ask for help in #local-workers on Discord."
    exit 1
fi

# Ensure no NVIDIA packages leaked in
"$SCRIPT_DIR/bin/uv" pip uninstall pynvml nvidia-ml-py 2>/dev/null

# Check if we are running in WSL2
WSL_KERNEL=$(uname -a | grep -c -e WSL2 )
if [ "$WSL_KERNEL" -gt 0 ]; then
    export IN_WSL="TRUE"
    echo "WSL environment detected. Patching ROCm libhsa-runtime64.so"
    for i in $(find ./ -iname libhsa-runtime64.so); do cp /opt/rocm/lib/libhsa-runtime64.so "$i"; done
fi

# Install AMD Go Fast optimizations
"$SCRIPT_DIR/bin/uv" run "$SCRIPT_DIR/horde_worker_regen/amd_go_fast/install_amd_go_fast.sh"
