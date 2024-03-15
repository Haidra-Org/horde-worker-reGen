# import yaml
import pathlib

import pytest
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_sdk.generic_api.consts import ANON_API_KEY
from ruamel.yaml import YAML

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.bridge_data.load_config import BridgeDataLoader, ConfigFormat


def test_bridge_data_yaml() -> None:
    # bridge_data_filename = "bridgeData.yaml"
    bridge_data_filename = "bridgeData_template.yaml"
    bridge_data_raw: dict | None = None

    yaml = YAML(typ="safe")

    with open(bridge_data_filename, encoding="utf-8") as f:
        bridge_data_raw = yaml.load(f)

    assert bridge_data_raw is not None

    parsed_bridge_data = reGenBridgeData.model_validate(bridge_data_raw)

    assert parsed_bridge_data is not None
    assert parsed_bridge_data.disable_terminal_ui is False
    assert parsed_bridge_data.api_key == ANON_API_KEY

    assert parsed_bridge_data.meta_load_instructions is not None
    assert len(parsed_bridge_data.meta_load_instructions) == 1


def test_bridge_data_loader_yaml_template() -> None:
    bridge_data_loader = BridgeDataLoader()

    horde_model_reference_manager = ModelReferenceManager(
        download_and_convert_legacy_dbs=True,
        override_existing=True,
    )
    bridge_data = bridge_data_loader.load(
        file_path="bridgeData_template.yaml",
        file_format=ConfigFormat.yaml,
        horde_model_reference_manager=horde_model_reference_manager,
    )

    assert bridge_data is not None
    assert bridge_data.disable_terminal_ui is False
    assert bridge_data.api_key == ANON_API_KEY


def test_bridge_data_loader_yaml_local_if_present() -> None:
    bridge_data_loader = BridgeDataLoader()

    horde_model_reference_manager = ModelReferenceManager(
        download_and_convert_legacy_dbs=True,
        override_existing=True,
    )

    if pathlib.Path("bridgeData.yaml").is_file():
        bridge_data = bridge_data_loader.load(
            file_path="bridgeData.yaml",
            file_format=ConfigFormat.yaml,
            horde_model_reference_manager=horde_model_reference_manager,
        )

        assert bridge_data is not None
        assert bridge_data.api_key != ANON_API_KEY
        assert len(bridge_data.image_models_to_load) > 0

    else:
        pytest.skip("bridgeData.yaml not found")


def test_bridge_data_load_from_env_vars() -> None:
    import os

    os.environ["AIWORKER_REGEN_HORDE_URL"] = "https://localhost:8080"
    os.environ["AIWORKER_REGEN_MODELS_TO_LOAD"] = "['model1', 'model2']"

    horde_model_reference_manager = ModelReferenceManager(
        download_and_convert_legacy_dbs=True,
        override_existing=True,
    )

    bridge_data = BridgeDataLoader.load_from_env_vars(
        horde_model_reference_manager=horde_model_reference_manager,
    )
    assert bridge_data is not None


def test_bridge_data_to_dot_env_file() -> None:
    bridge_data = reGenBridgeData.model_validate({})

    bridge_data.horde_url = "https://localhost:8080"
    bridge_data.image_models_to_load = ["model1", "model2"]

    BridgeDataLoader.write_bridge_data_as_dot_env_file(bridge_data, "bridgeData.env")
