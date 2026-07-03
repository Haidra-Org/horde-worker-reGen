"""Tests for the bridge config file location override (issue #17)."""

import importlib
from collections.abc import Generator

import pytest

import horde_worker_regen.consts


@pytest.fixture(autouse=True)
def _restore_consts_module() -> Generator[None, None, None]:
    """Reload the consts module after each test so the reloads here don't leak into other tests."""
    yield
    importlib.reload(horde_worker_regen.consts)


def test_bridge_config_filename_defaults_to_bridgedata_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the env var, the config location is the historical `bridgeData.yaml`."""
    monkeypatch.delenv("AIWORKER_BRIDGE_DATA_LOCATION", raising=False)

    consts = importlib.reload(horde_worker_regen.consts)

    assert consts.BRIDGE_CONFIG_FILENAME == "bridgeData.yaml"
    assert consts.DEFAULT_BRIDGE_CONFIG_FILENAME == "bridgeData.yaml"


def test_bridge_config_filename_overridden_by_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """`AIWORKER_BRIDGE_DATA_LOCATION` overrides the config location, including full paths."""
    monkeypatch.setenv("AIWORKER_BRIDGE_DATA_LOCATION", "/configs/worker1/bridgeData.yaml")

    consts = importlib.reload(horde_worker_regen.consts)

    assert consts.BRIDGE_CONFIG_FILENAME == "/configs/worker1/bridgeData.yaml"
    assert consts.DEFAULT_BRIDGE_CONFIG_FILENAME == "bridgeData.yaml"
