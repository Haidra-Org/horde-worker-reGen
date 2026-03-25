#!/bin/bash
# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# List of directories to check for jemalloc
dirs=(
    "/usr/lib"
    "/usr/local/lib"
    "/lib"
    "/lib64"
    "/usr/lib/x86_64-linux-gnu"
)

# Check each directory
for dir in "${dirs[@]}"; do
    if [ -f "$dir/libjemalloc.so.2" ]; then
        export LD_PRELOAD="$dir/libjemalloc.so.2"
        printf "Using jemalloc from $dir\n"
        break
    fi
done

# If jemalloc was not found, print a warning
if [ -z "$LD_PRELOAD" ]; then
    printf "WARNING: jemalloc not found. You may run into memory issues! We recommend running 'sudo apt install libjemalloc2'\n"
    # Press q to quit or any other key to continue
    read -n 1 -s -r -p "Press q to quit or any other key to continue: " key
    if [ "$key" = "q" ]; then
        printf "\n"
        exit 1
    fi
fi

echo "============================================"
echo "  AI Horde Worker"
echo "============================================"
echo ""
if "$SCRIPT_DIR/runtime.sh" python -s "$SCRIPT_DIR/download_models.py"; then
    echo ""
    echo "Models ready. Starting worker..."
    echo "(Press Ctrl+C to stop the worker gracefully)"
    echo ""
    "$SCRIPT_DIR/runtime.sh" python -s "$SCRIPT_DIR/run_worker.py" $*
else
    echo ""
    echo "ERROR: Model download failed. Check the output above and try again."
    echo "       Common fix: check your internet connection and disk space."
fi
