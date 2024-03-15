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

TRACKED_DEPENDENCIES = [
    "horde_sdk",
    "hordelib",
    "horde_model_reference",
    # "horde_safety"
]


@pytest.fixture(scope="session")
def horde_dependency_versions() -> list[tuple[str, str]]:
    """Get the versions of horde dependencies from the requirements file."""
    with open(REQUIREMENTS_FILE_PATH) as f:
        requirements = f.readlines()

    dependencies = []
    for req in requirements:
        for dep in TRACKED_DEPENDENCIES:
            if req.startswith(dep):
                dependencies.append((dep, req.split("~=")[1].strip()))

    return dependencies
