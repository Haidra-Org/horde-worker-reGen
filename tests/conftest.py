# import hordelib
import pytest
from loguru import logger


@pytest.fixture(scope="session", autouse=True)
def init_hordelib() -> None:
    # hordelib.initialise()
    logger.warning("hordelib.initialise() not called")
