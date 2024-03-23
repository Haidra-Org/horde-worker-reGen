import os
import pathlib

os.environ["HORDE_SDK_DISABLE_CUSTOM_SINKS"] = "1"
# isort: off
from horde_worker_regen.load_env_vars import load_env_vars_from_config

load_env_vars_from_config()

# isort: on

import argparse

from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_worker_regen.bridge_data.load_config import BridgeDataLoader, ConfigFormat


def convert_config_to_env(config_filename: str = "bridgeData.yaml", dot_env_filename: str = "bridgeData.env") -> None:
    """Convert the config file to an env file (suitable for use in a container or similar)."""
    bridge_data_loader = BridgeDataLoader()
    horde_model_reference_manager = ModelReferenceManager(
        download_and_convert_legacy_dbs=True,
        override_existing=True,
    )

    if pathlib.Path(config_filename).is_file():
        bridge_data = bridge_data_loader.load(
            file_path=config_filename,
            file_format=ConfigFormat.yaml,
            horde_model_reference_manager=horde_model_reference_manager,
        )
        if bridge_data is not None:
            try:
                BridgeDataLoader.write_bridge_data_as_dot_env_file(bridge_data, dot_env_filename)
            except Exception as e:
                print(f"Failed to write config to {dot_env_filename} ({type(e)}): {e}")

        else:
            print("Failed to convert config to env")

    else:
        print(f"File {config_filename} not found")


if __name__ == "__main__":
    # Add argparse arguments
    parser = argparse.ArgumentParser(description="Convert config to env")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="The file to convert",
        default="bridgeData.yaml",
    )
    args = parser.parse_args()

    convert_config_to_env(args.file)
