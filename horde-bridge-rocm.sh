#!/bin/bash
# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Build the absolute path to the Conda environment
CONDA_ENV_PATH="$SCRIPT_DIR/conda/envs/linux/lib"

# Determine if the user has a flash attention supported card.
SUPPORTED_CARD=$(rocminfo | grep -c -e gfx1100 -e gfx1101 -e gfx1102)
if [ "$SUPPORTED_CARD" -gt 0 ]; then export FLASH_ATTENTION_USE_TRITON_ROCM="${FLASH_ATTENTION_USE_TRITON_ROCM:=TRUE}"; fi
export MIOPEN_FIND_MODE="FAST"

# Check if we are running in WSL2
WSL_KERNEL=$(uname -a | grep -c -e WSL2 )
if [ "$WSL_KERNEL" -gt 0 ]; then
    export "${IN_WSL:=TRUE}"
    for i in $(find ./ -iname libhsa-runtime64.so); do cp /opt/rocm/lib/libhsa-runtime64.so $i; done
fi

# Add the Conda environment to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_ENV_PATH:$LD_LIBRARY_PATH"

# Set torch garbage cleanup. Amd defaults cause problems. //this was less stable than the torch 2.5.0 defaults in my testing
#export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:2048

# List of directories to check
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
    printf "WARNING: jemalloc not found. You may run into memory issues! We recommend running `sudo apt install libjemalloc2`\n"
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
