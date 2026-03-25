#!/bin/bash
set -e

# Source environment variables from /env_vars file
if [ -f "/env_vars" ]; then
    . /env_vars
else
    echo "/env_vars file not found. Exiting."
    exit 1
fi

cd ${APP_HOME}
git fetch
git reset --hard origin/${GIT_BRANCH:-main}

# Determine GPU type and set environment variables
if [ ! -z "${GPU_TYPE}" ] && [ "${GPU_TYPE}" = "rocm" ]; then
    export GPU_EXTRA="rocm"

    # Determine if the user has a flash attention supported card.
    SUPPORTED_CARD=$(rocminfo | grep -c -e gfx1100 -e gfx1101 -e gfx1102)
    if [ "$SUPPORTED_CARD" -gt 0 ]; then export FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:=TRUE}"; fi

    export MIOPEN_FIND_MODE="FAST"
else
    export GPU_EXTRA="cu128"
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
fi

uv sync --locked --extra ${GPU_EXTRA}

# Run GPU-specific setup scripts if they exist
if [ -f "${APP_HOME}/setup_${GPU_TYPE:-cuda}.sh" ]; then
    bash "${APP_HOME}/setup_${GPU_TYPE:-cuda}.sh"
fi

# Run the worker
if [ -e bridgeData.yaml ]; then
    uv run --no-sync python download_models.py
    exec uv run --no-sync python run_worker.py
else
    uv run --no-sync python download_models.py -e
    exec uv run --no-sync python run_worker.py -e
fi
