#!/bin/bash
# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Determine if the user has a flash attention supported card.
SUPPORTED_CARD=$(rocminfo | grep -c -e gfx1100 -e gfx1101 -e gfx1102)
if [ "$SUPPORTED_CARD" -gt 0 ]; then export FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:=TRUE}"; fi
export MIOPEN_FIND_MODE="FAST"

# Check if we are running in WSL2
WSL_KERNEL=$(uname -a | grep -c -e WSL2 )
if [ "$WSL_KERNEL" -gt 0 ]; then
    export IN_WSL="TRUE"
    echo "WSL environment detected. Patching ROCm libhsa-runtime64.so"
    for i in $(find ./ -iname libhsa-runtime64.so); do cp /opt/rocm/lib/libhsa-runtime64.so "$i"; done
fi

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


if "$SCRIPT_DIR/runtime-rocm.sh" python -s "$SCRIPT_DIR/download_models.py"; then
    echo "Model Download OK. Starting worker..."
    "$SCRIPT_DIR/runtime-rocm.sh" python -s "$SCRIPT_DIR/run_worker.py" $*
else
    echo "download_models.py exited with error code. Aborting"
fi
