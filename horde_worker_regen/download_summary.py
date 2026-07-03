"""Pre-download reporting: how many model files are about to be downloaded and how much space they need.

Model reference records do not include file sizes, so sizes are resolved with HTTP HEAD requests
(`Content-Length`) against the download URLs. Files whose size cannot be determined are still
counted, but reported separately as "unknown size".
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import requests
from loguru import logger

HEAD_REQUEST_TIMEOUT_SECONDS = 15
MAX_PARALLEL_SIZE_CHECKS = 8

DISK_SPACE_SAFETY_MARGIN = 1.05
"""Warn when free disk space is below the total download size times this margin."""


@dataclass
class PendingDownload:
    """A single model file which is about to be downloaded."""

    model_name: str
    file_url: str
    size_bytes: int | None = None


def format_bytes(num_bytes: float) -> str:
    """Return a human readable size such as `1.24 GB`."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.2f} {unit}" if unit != "B" else f"{int(num_bytes)} B"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def fetch_remote_file_size(file_url: str, *, timeout: float = HEAD_REQUEST_TIMEOUT_SECONDS) -> int | None:
    """Return the size of a remote file in bytes using a HEAD request, or None if it cannot be determined."""
    try:
        response = requests.head(file_url, allow_redirects=True, timeout=timeout)
        content_length = response.headers.get("Content-Length")
        if content_length is not None and content_length.isdigit():
            return int(content_length)
    except requests.RequestException:
        logger.debug(f"Could not determine remote file size for {file_url}")
    return None


class SupportsModelDownloads(Protocol):
    """The parts of a hordelib model manager needed to inspect pending downloads."""

    def is_model_available(self, model_name: str) -> bool:
        """Return True if the model is already downloaded and valid."""
        ...

    def get_model_download(self, model_name: str) -> list[dict]:
        """Return the download entries (with `file_url`) for the model."""
        ...


def collect_pending_downloads(
    model_manager: SupportsModelDownloads,
    model_names: Iterable[str],
) -> list[PendingDownload]:
    """Return the files which downloading the given models would fetch (models already on disk are skipped)."""
    pending_downloads: list[PendingDownload] = []
    for model_name in model_names:
        try:
            if model_manager.is_model_available(model_name):
                continue
            download_entries = model_manager.get_model_download(model_name)
        except Exception as e:
            logger.debug(f"Could not inspect download info for model {model_name}: {e}")
            continue

        for download_entry in download_entries:
            file_url = download_entry.get("file_url")
            if file_url:
                pending_downloads.append(PendingDownload(model_name=model_name, file_url=file_url))

    return pending_downloads


def populate_download_sizes(pending_downloads: list[PendingDownload]) -> None:
    """Fill in `size_bytes` for each pending download using parallel HEAD requests."""
    if not pending_downloads:
        return

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SIZE_CHECKS) as executor:
        sizes = executor.map(lambda pending: fetch_remote_file_size(pending.file_url), pending_downloads)
        for pending_download, size_bytes in zip(pending_downloads, sizes, strict=True):
            pending_download.size_bytes = size_bytes


def log_download_summary(
    pending_by_category: dict[str, list[PendingDownload]],
    *,
    download_target_dir: str | Path | None = None,
) -> None:
    """Log a summary of the pending downloads (file counts and sizes, per category and overall).

    If `download_target_dir` is given, also compare the total against the free disk space
    and warn when it looks like the downloads will not fit (see issue #267).
    """
    all_pending = [pending for category_pending in pending_by_category.values() for pending in category_pending]

    if not all_pending:
        logger.info("All models are already downloaded; nothing new to fetch.")
        return

    total_known_bytes = 0
    total_unknown_count = 0

    logger.info(f"About to download {len(all_pending)} model file(s):")
    for category, category_pending in pending_by_category.items():
        if not category_pending:
            continue

        known_bytes = sum(pending.size_bytes for pending in category_pending if pending.size_bytes is not None)
        unknown_count = sum(1 for pending in category_pending if pending.size_bytes is None)
        total_known_bytes += known_bytes
        total_unknown_count += unknown_count

        size_note = format_bytes(known_bytes)
        if unknown_count:
            size_note += f" + {unknown_count} file(s) of unknown size"
        logger.info(f"  {category}: {len(category_pending)} file(s), {size_note}")

    total_note = format_bytes(total_known_bytes)
    if total_unknown_count:
        total_note += f" + {total_unknown_count} file(s) of unknown size"
    logger.info(f"Total download size: {total_note}")

    if download_target_dir is not None and total_known_bytes > 0:
        _warn_if_low_disk_space(download_target_dir, total_known_bytes)


def _warn_if_low_disk_space(download_target_dir: str | Path, total_download_bytes: int) -> None:
    """Warn when the free space at the download location is unlikely to fit the pending downloads."""
    try:
        free_bytes = shutil.disk_usage(download_target_dir).free
    except OSError as e:
        logger.debug(f"Could not check free disk space for {download_target_dir}: {e}")
        return

    required_bytes = total_download_bytes * DISK_SPACE_SAFETY_MARGIN
    if free_bytes < required_bytes:
        logger.warning(
            f"Low disk space: the pending downloads need about {format_bytes(total_download_bytes)} "
            f"but only {format_bytes(free_bytes)} is free at {download_target_dir}. "
            "The worker may fail or crash-loop if it runs out of disk space mid-download.",
        )
    else:
        logger.info(f"Free disk space at {download_target_dir}: {format_bytes(free_bytes)} (sufficient)")
