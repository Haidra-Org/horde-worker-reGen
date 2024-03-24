"""The primary package for the reGen worker."""

from dotenv import load_dotenv

load_dotenv()

from pathlib import Path  # noqa: E402

ASSETS_FOLDER_PATH = Path(__file__).parent / "assets"

__version__ = "4.3.9"
