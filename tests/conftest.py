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


@pytest.fixture(scope="session")
def horde_dependency_versions() -> list[tuple[str, str]]:
    """Get the versions of horde dependencies from the requirements file."""
    with open(REQUIREMENTS_FILE_PATH) as f:
        requirements = f.readlines()

    dependencies = []
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
                dependencies.append((dep, version))

    return dependencies


@pytest.fixture(scope="session")
def rocm_horde_dependency_versions() -> list[tuple[str, str]]:
    """Get the versions of horde dependencies from the ROCm requirements file."""
    with open(ROCM_REQUIREMENTS_FILE_PATH) as f:
        requirements = f.readlines()

    dependencies = []
    for req in requirements:
        for dep in TRACKED_DEPENDENCIES:
            if req.startswith(dep):
                if "==" in req:
                    version = req.split("==")[1].strip()
                elif "~=" in req:
                    version = req.split("~=")[1].strip()
                else:
                    raise ValueError(f"Unsupported version pin: {req}")

                # Strip any info starting from the `+` character
                version = version.split("+")[0]
                dependencies.append((dep, version))

    return dependencies
