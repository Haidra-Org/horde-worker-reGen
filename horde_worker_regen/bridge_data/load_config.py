import json
from enum import auto
from pathlib import Path

import yaml
from strenum import StrEnum

from horde_worker_regen.bridge_data.data_model import reGenBridgeData


class UnsupportedConfigFormat(Exception):
    def __init__(self, file_path: str | Path, file_format: str) -> None:
        super().__init__(f"Unsupported config file format: {file_format} ({file_path})")


class ConfigFormat(StrEnum):
    yaml = auto()
    json = auto()


class BridgeDataLoader:
    @staticmethod
    def _infer_format(file_path: str | Path) -> ConfigFormat:
        """Infer the config file format from the file extension.

        Args:
            file_path (str | Path): The path to the config file.

        Returns:
            ConfigFormat: The config file format.

        Raises:
            UnsupportedConfigFormat: If the config file format is not supported.
        """
        file_path = Path(file_path)

        if file_path.suffix == ".yaml" or file_path.suffix == ".yml":
            return ConfigFormat.yaml

        if file_path.suffix == ".json":
            return ConfigFormat.json

        raise UnsupportedConfigFormat(file_path, file_path.suffix)

    @staticmethod
    def load(
        file_path: str | Path,
        *,
        file_format: ConfigFormat | None = None,
    ) -> reGenBridgeData:
        """Load the config file and validate it.

        Args:
            file_path (str | Path): The path to the config file.
            file_format (ConfigFormat | None, optional): The config file format. Defaults to None.
            The file format will be inferred from the file extension if not provided.

        Returns:
            reGenBridgeData: The validated config file.

        Raises:
            ValidationError: If the config file is invalid.
            UnsupportedConfigFormat: If the config file format is not supported.

        """
        # Infer the file format if not provided
        if not file_format:
            file_format = BridgeDataLoader._infer_format(file_path)

        if file_format == ConfigFormat.yaml:
            with open(file_path) as f:
                config = yaml.safe_load(f)

            return reGenBridgeData.model_validate(config)

        if file_format == ConfigFormat.json:
            with open(file_path) as f:
                config = json.load(f)

            return reGenBridgeData.model_validate(config)

        raise UnsupportedConfigFormat(file_path, file_format)
