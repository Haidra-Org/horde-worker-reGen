"""The primary package for the reGen worker."""

from dotenv import load_dotenv

load_dotenv()

from pathlib import Path  # noqa: E402

ASSETS_FOLDER_PATH = Path(__file__).parent / "assets"

__version__ = "9.6.17"


import pkg_resources  # noqa: E402


def check_hordelib_not_installed() -> None:
    """Check that hordelib is not installed."""
    try:
        pkg_resources.get_distribution("hordelib")
        raise RuntimeError(
            "hordelib is installed. Please uninstall it before running this package. "
            "`hordelib` has been renamed to `horde_engine`.",
        )
    except pkg_resources.DistributionNotFound:
        pass


check_hordelib_not_installed()
