import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "horde_worker_regen" / "runtime_backend.py"
MODULE_SPEC = importlib.util.spec_from_file_location("runtime_backend", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
runtime_backend = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = runtime_backend
MODULE_SPEC.loader.exec_module(runtime_backend)

HordeRuntimeBackend = runtime_backend.HordeRuntimeBackend
clear_torch_cache = runtime_backend.clear_torch_cache
get_torch_device_infos = runtime_backend.get_torch_device_infos


class RuntimeBackendTests(unittest.TestCase):
    def test_conflicting_backends_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            HordeRuntimeBackend(amd_gpu=True, xpu=True)

    def test_oneapi_selector_requires_xpu(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --xpu"):
            HordeRuntimeBackend(oneapi_device_selector="level_zero:gpu:0")

    def test_append_comfyui_args_for_xpu(self) -> None:
        backend = HordeRuntimeBackend(xpu=True)
        extra_args = ["--disable-smart-memory"]

        backend.append_comfyui_args(extra_args)

        self.assertEqual(extra_args, ["--disable-smart-memory", "--use-pytorch-cross-attention"])
        self.assertFalse(backend.supports_gpu_safety)

    def test_append_comfyui_args_for_directml(self) -> None:
        backend = HordeRuntimeBackend(directml=0)
        extra_args: list[str] = []

        backend.append_comfyui_args(extra_args)

        self.assertEqual(extra_args, ["--directml=0"])
        self.assertFalse(backend.supports_gpu_safety)

    def test_get_torch_device_infos_uses_xpu(self) -> None:
        backend = HordeRuntimeBackend(xpu=True)
        fake_torch = types.SimpleNamespace(
            xpu=types.SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_properties=lambda index: types.SimpleNamespace(total_memory=16 * 1024**3),
                get_device_name=lambda index: "Intel Arc A770",
            ),
            cuda=types.SimpleNamespace(device_count=lambda: 0),
        )

        with patch.dict("sys.modules", {"torch": fake_torch}):
            infos = get_torch_device_infos(backend)

        self.assertEqual(infos, [("Intel Arc A770", 0, 16 * 1024**3)])

    def test_get_torch_device_infos_ignores_directml(self) -> None:
        backend = HordeRuntimeBackend(directml=0)
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                device_count=lambda: 1,
                get_device_properties=lambda index: types.SimpleNamespace(name="Unexpected CUDA", total_memory=1),
            ),
        )

        with patch.dict("sys.modules", {"torch": fake_torch}):
            infos = get_torch_device_infos(backend)

        self.assertEqual(infos, [])

    def test_clear_torch_cache_uses_xpu(self) -> None:
        backend = HordeRuntimeBackend(xpu=True)
        fake_xpu = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: setattr(fake_xpu, "called", True))
        fake_xpu.called = False
        fake_torch = types.SimpleNamespace(xpu=fake_xpu, cuda=types.SimpleNamespace(is_available=lambda: False))

        with patch.dict("sys.modules", {"torch": fake_torch}):
            clear_torch_cache(backend)

        self.assertTrue(fake_xpu.called)

    def test_clear_torch_cache_uses_cuda(self) -> None:
        backend = HordeRuntimeBackend()
        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: setattr(fake_cuda, "called", True),
        )
        fake_cuda.called = False
        fake_torch = types.SimpleNamespace(cuda=fake_cuda)

        with patch.dict("sys.modules", {"torch": fake_torch}):
            clear_torch_cache(backend)

        self.assertTrue(fake_cuda.called)

    def test_clear_torch_cache_ignores_directml(self) -> None:
        backend = HordeRuntimeBackend(directml=0)
        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: setattr(fake_cuda, "called", True),
        )
        fake_cuda.called = False
        fake_torch = types.SimpleNamespace(cuda=fake_cuda)

        with patch.dict("sys.modules", {"torch": fake_torch}):
            clear_torch_cache(backend)

        self.assertFalse(fake_cuda.called)


if __name__ == "__main__":
    unittest.main()
