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
elif [ ! -z "${CUDA_VERSION_SHORT}" ]; then
    # CUDA environment
    export GPU_TYPE="cuda"
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export PYTORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cu${CUDA_VERSION_SHORT}"
else
    echo "Neither ROCm nor CUDA environment variables found in /env_vars. Exiting."
    exit 1
fi

python -m pip install -r requirements.txt -U --extra-index-url ${PYTORCH_EXTRA_INDEX}

# Run GPU-specific setup scripts if they exist
if [ -f "${APP_HOME}/setup_${GPU_TYPE}.sh" ]; then
    bash "${APP_HOME}/setup_${GPU_TYPE}.sh"
fi

# Run the worker
if [ -e bridgeData.yaml ]; then
    # There is a bridgeData.yaml file, we'll load from that
    python download_models.py
    python run_worker.py
else
    # No bridgeData.yaml file, we'll use environment variables
    python download_models.py -e
    python run_worker.py -e
fi
