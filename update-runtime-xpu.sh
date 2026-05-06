#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

hordelib=false

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --hordelib)
    hordelib=true
    shift
    ;;
    *)
    echo "Unknown option: $key"
    exit 1
    ;;
esac
done

CONDA_ENVIRONMENT_FILE=environment.xpu.yaml
PYTORCH_XPU_INDEX=https://download.pytorch.org/whl/xpu
PYPI_INDEX=https://pypi.org/simple

wget -qO- https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64.tar.bz2 | tar -xvj -C "${SCRIPT_DIR}"
if [ ! -f "$SCRIPT_DIR/conda/envs/linux/bin/python" ]; then
    ${SCRIPT_DIR}/bin/micromamba create --no-shortcuts -r "$SCRIPT_DIR/conda" -n linux -f ${CONDA_ENVIRONMENT_FILE} -y
fi
${SCRIPT_DIR}/bin/micromamba update --no-shortcuts -r "$SCRIPT_DIR/conda" -n linux -f ${CONDA_ENVIRONMENT_FILE} -y

${SCRIPT_DIR}/bin/micromamba run -r "$SCRIPT_DIR/conda" -n linux python -s -m pip uninstall -y \
    torch torchvision torchaudio triton xformers \
    intel-extension-for-pytorch intel-cmplr-lib-rt intel-cmplr-lib-ur intel-cmplr-lic-rt intel-sycl-rt \
    pytorch-triton-xpu tcmlib umf intel-pti \
    pynvml nvidia-ml-py

if [ "$hordelib" = true ]; then
    ${SCRIPT_DIR}/bin/micromamba run -r "$SCRIPT_DIR/conda" -n linux python -s -m pip uninstall -y \
        hordelib horde_engine horde_sdk horde_model_reference
    ${SCRIPT_DIR}/bin/micromamba run -r "$SCRIPT_DIR/conda" -n linux python -s -m pip install \
        horde_engine horde_model_reference \
        --index-url "${PYTORCH_XPU_INDEX}" \
        --extra-index-url "${PYPI_INDEX}"
else
    ${SCRIPT_DIR}/bin/micromamba run -r "$SCRIPT_DIR/conda" -n linux python -s -m pip install \
        -r "$SCRIPT_DIR/requirements.txt" -U \
        --index-url "${PYTORCH_XPU_INDEX}" \
        --extra-index-url "${PYPI_INDEX}"
fi

echo "Intel XPU runtime installed."
echo "Make sure the Intel GPU driver and Level Zero runtime are available on the host OS."
