"""The primary package for the reGen worker."""

from dotenv import load_dotenv

load_dotenv()

from pathlib import Path  # noqa: E402

ASSETS_FOLDER_PATH = Path(__file__).parent / "assets"

from horde_worker_regen.process_management.main_entry_point import start_working  # noqa: E402

__all__ = [
    "start_working",
]
