"""Helpers for choosing and configuring the runtime backend used by the worker."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HordeRuntimeBackend:
    """Represents the GPU backend selected for the worker runtime."""

    amd_gpu: bool = False
    directml: int | None = None
    xpu: bool = False
    oneapi_device_selector: str | None = None

    def __post_init__(self) -> None:
        selected_backends = []

        if self.amd_gpu:
            selected_backends.append("--amd")
        if self.directml is not None:
            selected_backends.append("--directml")
        if self.xpu:
            selected_backends.append("--xpu")

        if len(selected_backends) > 1:
            raise ValueError(
                "Backend flags are mutually exclusive. "
                f"Choose only one of: {', '.join(selected_backends)}.",
            )

        if self.oneapi_device_selector is not None and not self.xpu:
            raise ValueError("--oneapi-device-selector requires --xpu.")

    @property
    def name(self) -> str:
        """Return a short name for the active backend."""
        if self.directml is not None:
            return "directml"
        if self.xpu:
            return "xpu"
        if self.amd_gpu:
            return "rocm"
        return "cuda"

    @property
    def supports_gpu_safety(self) -> bool:
        """Return whether GPU-backed safety is reliable on this backend."""
        return self.directml is None and not self.xpu

    def apply_environment(self) -> None:
        """Apply any backend-specific environment variables for the current process."""
        if self.oneapi_device_selector is not None:
            os.environ["ONEAPI_DEVICE_SELECTOR"] = self.oneapi_device_selector

    def append_comfyui_args(self, extra_comfyui_args: list[str]) -> None:
        """Append backend-specific ComfyUI arguments in-place."""
        if self.amd_gpu or self.xpu:
            extra_comfyui_args.append("--use-pytorch-cross-attention")

        if self.directml is not None:
            extra_comfyui_args.append(f"--directml={self.directml}")


def get_torch_device_infos(backend: HordeRuntimeBackend) -> list[tuple[str, int, int]]:
    """Return the visible torch devices for the chosen backend."""
    import torch

    device_infos: list[tuple[str, int, int]] = []

    if backend.directml is not None:
        return device_infos

    if backend.xpu:
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            return device_infos

        for index in range(torch.xpu.device_count()):
            properties = torch.xpu.get_device_properties(index)
            device_infos.append(
                (
                    getattr(properties, "name", torch.xpu.get_device_name(index)),
                    index,
                    properties.total_memory,
                ),
            )
        return device_infos

    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        device_infos.append((properties.name, index, properties.total_memory))

    return device_infos


def clear_torch_cache(backend: HordeRuntimeBackend) -> None:
    """Clear the PyTorch cache for the active backend."""
    import torch

    if backend.directml is not None:
        return

    if backend.xpu:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
