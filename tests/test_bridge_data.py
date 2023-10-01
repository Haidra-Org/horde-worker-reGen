import yaml
from horde_sdk.generic_api.consts import ANON_API_KEY

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.bridge_data.load_config import BridgeDataLoader, ConfigFormat


def test_bridge_data_yaml() -> None:
    # bridge_data_filename = "bridgeData.yaml"
    bridge_data_filename = "bridgeData_template.yaml"
    bridge_data_raw: dict | None = None

    with open(bridge_data_filename) as f:
        bridge_data_raw = yaml.safe_load(f)

    assert bridge_data_raw is not None

    parsed_bridge_data = reGenBridgeData.model_validate(bridge_data_raw)

    assert parsed_bridge_data is not None
    assert parsed_bridge_data.disable_terminal_ui is False
    assert parsed_bridge_data.api_key == ANON_API_KEY

    assert parsed_bridge_data.meta_load_instructions is not None
    assert len(parsed_bridge_data.meta_load_instructions) == 1


def test_bridge_data_loader_yaml() -> None:
    bridge_data_loader = BridgeDataLoader()
    bridge_data = bridge_data_loader.load(
        file_path="bridgeData_template.yaml",
        file_format=ConfigFormat.yaml,
    )

    assert bridge_data is not None
    assert bridge_data.disable_terminal_ui is False
    assert bridge_data.api_key == ANON_API_KEY
