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

. venv/bin/activate

# Determine the GPU type and set environment variables accordingly
if [ ! -z "${ROCM_VERSION_SHORT}" ]; then
    # ROCm environment
    export GPU_TYPE="rocm"
    export PYTORCH_EXTRA_INDEX="https://download.pytorch.org/whl/rocm${ROCM_VERSION_SHORT}"
    export REQUIREMENTS_FILE="requirements.rocm.txt"

    # Determine if the user has a flash attention supported card.
    SUPPORTED_CARD=$(rocminfo | grep -c -e gfx1100 -e gfx1101 -e gfx1102)
    if [ "$SUPPORTED_CARD" -gt 0 ]; then export FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:=TRUE}"; fi

    #export PYTORCH_TUNABLEOP_ENABLED=1
    export MIOPEN_FIND_MODE="FAST"
    #export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512,expandable_segments:True"
elif [ ! -z "${CUDA_VERSION_SHORT}" ]; then
    # CUDA environment
    export GPU_TYPE="cuda"
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export PYTORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cu${CUDA_VERSION_SHORT}"
    export REQUIREMENTS_FILE="requirements.txt"
else
    echo "Neither ROCm nor CUDA environment variables found in /env_vars. Exiting."
    exit 1
fi

python -m pip install -r ${REQUIREMENTS_FILE} -U --extra-index-url ${PYTORCH_EXTRA_INDEX}

# Run GPU-specific setup scripts if they exist
if [ -f "${APP_HOME}/setup_${GPU_TYPE}.sh" ]; then
    bash "${APP_HOME}/setup_${GPU_TYPE}.sh"
fi

# Run the worker
if [ -e bridgeData.yaml ]; then
    # There is a bridgeData.yaml file, we'll load from that
    python download_models.py
    exec python run_worker.py
else
    # No bridgeData.yaml file, we'll use environment variables
    python download_models.py -e
    exec python run_worker.py -e
fi
