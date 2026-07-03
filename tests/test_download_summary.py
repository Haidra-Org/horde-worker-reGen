"""Tests for the pre-download summary (issues #400 and #267)."""

from pathlib import Path

import pytest

from horde_worker_regen.download_summary import (
    PendingDownload,
    collect_pending_downloads,
    format_bytes,
    log_download_summary,
    populate_download_sizes,
)


class FakeModelManager:
    """Duck-typed stand-in for a hordelib model manager."""

    def __init__(self, available_models: list[str], downloads: dict[str, list[dict] | dict]) -> None:
        """Initialise with the models considered on-disk and the download entries per model."""
        self._available_models = available_models
        self._downloads = downloads

    def is_model_available(self, model_name: str) -> bool:
        """Return True if the model was declared as already on disk."""
        return model_name in self._available_models

    def get_model_download(self, model_name: str) -> list[dict] | dict:
        """Return the declared download entries for the model."""
        return self._downloads[model_name]


def _capture_loguru_to_stdout() -> None:
    from loguru import logger

    logger.remove()
    logger.add(lambda message: print(message, end=""), level="INFO", format="{message}")


def test_format_bytes() -> None:
    """Sizes are rendered with a sensible unit and two decimals."""
    assert format_bytes(0) == "0 B"
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048) == "2.00 KB"
    assert format_bytes(5 * 1024**2) == "5.00 MB"
    assert format_bytes(1.5 * 1024**3) == "1.50 GB"
    assert format_bytes(2.25 * 1024**4) == "2.25 TB"


def test_collect_pending_downloads_skips_available_models() -> None:
    """Models already on disk are not counted as pending downloads."""
    manager = FakeModelManager(
        available_models=["model_on_disk"],
        downloads={
            "model_on_disk": [{"file_url": "https://example.com/on_disk.safetensors"}],
            "model_missing": [{"file_url": "https://example.com/missing.safetensors"}],
        },
    )

    pending = collect_pending_downloads(manager, ["model_on_disk", "model_missing"])

    assert len(pending) == 1
    assert pending[0].model_name == "model_missing"
    assert pending[0].file_url == "https://example.com/missing.safetensors"


def test_collect_pending_downloads_handles_multi_file_models_and_missing_urls() -> None:
    """Every download entry with a URL is counted; entries without a URL are skipped."""
    manager = FakeModelManager(
        available_models=[],
        downloads={
            "multi_file_model": [
                {"file_url": "https://example.com/part1.safetensors"},
                {"file_url": "https://example.com/part2.yaml"},
                {"file_url": None},
                {},
            ],
        },
    )

    pending = collect_pending_downloads(manager, ["multi_file_model"])

    assert [p.file_url for p in pending] == [
        "https://example.com/part1.safetensors",
        "https://example.com/part2.yaml",
    ]


def test_collect_pending_downloads_accepts_a_bare_dict_entry() -> None:
    """Older hordelib releases may hand back a single dict instead of a list of entries."""
    manager = FakeModelManager(
        available_models=[],
        downloads={
            "legacy_model": {"file_url": "https://example.com/legacy.safetensors"},
        },
    )

    pending = collect_pending_downloads(manager, ["legacy_model"])

    assert [p.file_url for p in pending] == ["https://example.com/legacy.safetensors"]


def test_collect_pending_downloads_survives_manager_errors() -> None:
    """A model manager raising for one model must not break the whole summary."""

    class ExplodingManager:
        def is_model_available(self, model_name: str) -> bool:
            raise RuntimeError("boom")

        def get_model_download(self, model_name: str) -> list[dict]:
            return []

    pending = collect_pending_downloads(ExplodingManager(), ["some_model"])

    assert pending == []


def test_populate_download_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sizes resolved via HEAD requests are written back onto the pending downloads."""
    sizes: dict[str, int | None] = {
        "https://example.com/a.safetensors": 100,
        "https://example.com/b.safetensors": None,
    }
    monkeypatch.setattr(
        "horde_worker_regen.download_summary.fetch_remote_file_size",
        lambda file_url, **kwargs: sizes[file_url],
    )

    pending = [
        PendingDownload(model_name="a", file_url="https://example.com/a.safetensors"),
        PendingDownload(model_name="b", file_url="https://example.com/b.safetensors"),
    ]
    populate_download_sizes(pending)

    assert pending[0].size_bytes == 100
    assert pending[1].size_bytes is None


def test_log_download_summary_reports_totals(capsys: pytest.CaptureFixture[str]) -> None:
    """The summary reports per-category and overall counts, sizes and unknown-size files."""
    _capture_loguru_to_stdout()

    pending_by_category = {
        "Stable Diffusion": [
            PendingDownload("model_a", "https://example.com/a", size_bytes=2 * 1024**3),
            PendingDownload("model_b", "https://example.com/b", size_bytes=None),
        ],
        "ESRGAN": [],
    }
    log_download_summary(pending_by_category)

    output = capsys.readouterr().out
    assert "About to download 2 model file(s)" in output
    assert "Stable Diffusion: 2 file(s), 2.00 GB + 1 file(s) of unknown size" in output
    assert "Total download size: 2.00 GB + 1 file(s) of unknown size" in output


def test_log_download_summary_nothing_to_download(capsys: pytest.CaptureFixture[str]) -> None:
    """When everything is already on disk, the summary says so instead of listing categories."""
    _capture_loguru_to_stdout()

    log_download_summary({"Stable Diffusion": []})

    output = capsys.readouterr().out
    assert "All models are already downloaded" in output


def test_log_download_summary_warns_on_low_disk_space(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warning is logged when the pending downloads exceed the free disk space."""
    import shutil as shutil_module

    _capture_loguru_to_stdout()

    fake_usage = shutil_module.disk_usage(tmp_path)._replace(free=1024)
    monkeypatch.setattr("horde_worker_regen.download_summary.shutil.disk_usage", lambda path: fake_usage)

    pending_by_category = {
        "Stable Diffusion": [PendingDownload("model_a", "https://example.com/a", size_bytes=10 * 1024**3)],
    }
    log_download_summary(pending_by_category, download_target_dir=tmp_path)

    output = capsys.readouterr().out
    assert "Low disk space" in output
