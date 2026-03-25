# Uninstall NVIDIA-specific packages in ROCm environment, just in case
uv pip uninstall pynvml nvidia-ml-py 2>/dev/null || true

./horde_worker_regen/amd_go_fast/install_amd_go_fast.sh
