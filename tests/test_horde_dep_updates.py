from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

PYPROJECT_FILE_PATH = Path(__file__).parent.parent / "pyproject.toml"
UV_LOCK_FILE_PATH = Path(__file__).parent.parent / "uv.lock"


def test_uv_lock_exists() -> None:
    """Check that uv.lock exists and is not empty."""
    assert UV_LOCK_FILE_PATH.exists(), "uv.lock not found — run 'uv lock' to generate it"
    assert UV_LOCK_FILE_PATH.stat().st_size > 0, "uv.lock is empty"


def test_gpu_extras_defined() -> None:
    """Check that all GPU extras are defined in pyproject.toml."""
    with open(PYPROJECT_FILE_PATH, "rb") as f:
        data = tomllib.load(f)

    extras = data["project"]["optional-dependencies"]
    for extra_name in ("cu128", "rocm", "directml", "cpu"):
        assert extra_name in extras, f"GPU extra '{extra_name}' not found in pyproject.toml"
        assert len(extras[extra_name]) > 0, f"GPU extra '{extra_name}' has no dependencies"
