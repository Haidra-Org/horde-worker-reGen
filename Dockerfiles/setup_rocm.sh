# Uninstall NVIDIA-specific packages in ROCm environment, just in case
. venv/bin/activate
python -m pip uninstall -y pynvml nvidia-ml-py

if [[ "${FLASH_ATTENTION_USE_TRITON_ROCM^^}" == "TRUE" ]]; then #this requires bash, not sh to work
  ./horde_worker_regen/amd_go_fast/install_amd_go_fast.sh
fi
