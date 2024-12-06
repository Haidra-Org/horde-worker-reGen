"""Configures pytest and creates fixtures."""

# import hordelib
from pathlib import Path

import pytest
from loguru import logger


@pytest.fixture(scope="session", autouse=True)
def init_hordelib() -> None:
    """Initialise hordelib for the tests."""
    # hordelib.initialise() # FIXME
    logger.warning("hordelib.initialise() not called")


PRECOMMIT_FILE_PATH = Path(__file__).parent.parent / ".pre-commit-config.yaml"
REQUIREMENTS_FILE_PATH = Path(__file__).parent.parent / "requirements.txt"
ROCM_REQUIREMENTS_FILE_PATH = Path(__file__).parent.parent / "requirements.rocm.txt"
DIRECTML_REQUIREMENTS_FILE_PATH = Path(__file__).parent.parent / "requirements.directml.txt"

TRACKED_DEPENDENCIES = [
    "horde_sdk",
    "horde_engine",
    "horde_model_reference",
    "horde_safety",
    "torch",
    "pydantic",
]


@pytest.fixture(scope="session")
def tracked_dependencies() -> list[str]:
    """Get the tracked dependencies."""
    return TRACKED_DEPENDENCIES


def get_dependency_versions(requirements_file_path: str) -> dict[str, str]:
    """Get the versions of horde dependencies from the given requirements file."""
    with open(requirements_file_path) as f:
        requirements = f.readlines()

    dependencies = {}
    for req in requirements:
        for dep in TRACKED_DEPENDENCIES:
            if req.startswith(dep):
                if "==" in req:
                    version = req.split("==")[1].strip()
                elif "~=" in req:
                    version = req.split("~=")[1].strip()
                elif ">=" in req:
                    version = req.split(">=")[1].strip()
                else:
                    raise ValueError(f"Unsupported version pin: {req}")

                # Strip any info starting from the `+` character
                version = version.split("+")[0]
                dependencies[dep] = version

    return dependencies


@pytest.fixture(scope="session")
def horde_dependency_versions() -> dict[str, str]:
    """Get the versions of horde dependencies from the requirements file."""
    return get_dependency_versions(REQUIREMENTS_FILE_PATH)


@pytest.fixture(scope="session")
def rocm_horde_dependency_versions() -> dict[str, str]:
    """Get the versions of horde dependencies from the ROCm requirements file."""
    return get_dependency_versions(ROCM_REQUIREMENTS_FILE_PATH)


@pytest.fixture(scope="session")
def directml_horde_dependency_versions() -> dict[str, str]:
    """Get the versions of horde dependencies from the DirectML requirements file."""
    return get_dependency_versions(DIRECTML_REQUIREMENTS_FILE_PATH)
