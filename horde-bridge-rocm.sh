#!/bin/bash
# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Build the absolute path to the Conda environment
CONDA_ENV_PATH="$SCRIPT_DIR/conda/envs/linux/lib"

# Add the Conda environment to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_ENV_PATH:$LD_LIBRARY_PATH"

# Set torch garbage cleanup. Amd defaults cause problems.
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:2048

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


if ./runtime-rocm.sh python -s download_models.py; then
    echo "Model Download OK. Starting worker..."
    ./runtime-rocm.sh python -s run_worker.py --amd $*
else
    echo "download_models.py exited with error code. Aborting"
fi
