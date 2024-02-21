"""Contains methods for loading the config file."""

import json
from enum import auto
from pathlib import Path

from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPIManualClient
from horde_sdk.ai_horde_worker.model_meta import ImageModelLoadResolver
from loguru import logger
from ruamel.yaml import YAML
from strenum import StrEnum

from horde_worker_regen.bridge_data.data_model import reGenBridgeData


class UnsupportedConfigFormat(Exception):
    """The config file format is not supported."""

    def __init__(self, file_path: str | Path, file_format: str) -> None:
        """Initialise the exception."""
        super().__init__(f"Unsupported config file format: {file_format} ({file_path})")


class ConfigFormat(StrEnum):
    """The format of the config file."""

    yaml = auto()
    json = auto()


class BridgeDataLoader:
    """Contains methods for loading the config file."""

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
        horde_model_reference_manager: ModelReferenceManager | None = None,
    ) -> reGenBridgeData:
        """Load the config file and validate it.

        Args:
            file_path (str | Path): The path to the config file.
            file_format (ConfigFormat | None, optional): The config file format. Defaults to None. \
            The file format will be inferred from the file extension if not provided.
            horde_model_reference_manager (ModelReferenceManager | None, optional): The model reference manager. \
            Used to resolve meta instructions. Defaults to None.

        Returns:
            reGenBridgeData: The validated config file.

        Raises:
            ValidationError: If the config file is invalid.
            UnsupportedConfigFormat: If the config file format is not supported.

        """
        file_path = Path(file_path)
        # Infer the file format if not provided
        if not file_format:
            file_format = BridgeDataLoader._infer_format(file_path)

        bridge_data: reGenBridgeData | None = None

        if file_format == ConfigFormat.yaml:
            yaml = YAML()
            with open(file_path, encoding="utf-8") as f:
                config = yaml.load(f)

            bridge_data = reGenBridgeData.model_validate(config)
            if bridge_data is not None:
                bridge_data._yaml_loader = yaml

        if file_format == ConfigFormat.json:
            with open(file_path, encoding="utf-8") as f:
                config = json.load(f)

            bridge_data = reGenBridgeData.model_validate(config)

        if not bridge_data:
            raise UnsupportedConfigFormat(file_path, file_format)

        if not horde_model_reference_manager:
            logger.warning(
                "No model reference manager provided. The config file will not be able to resolve meta instructions.",
            )
            return bridge_data

        bridge_data.image_models_to_load = BridgeDataLoader._resolve_meta_instructions(
            bridge_data,
            horde_model_reference_manager,
        )

        return bridge_data

    @staticmethod
    def _resolve_meta_instructions(  # FIXME: This should be moved into the SDK
        bridge_data: reGenBridgeData,
        horde_model_reference_manager: ModelReferenceManager,
    ) -> list[str]:
        load_resolver = ImageModelLoadResolver(horde_model_reference_manager)

        resolved_models = None
        if bridge_data.meta_load_instructions is not None:
            resolved_models = load_resolver.resolve_meta_instructions(
                list(bridge_data.meta_load_instructions),
                AIHordeAPIManualClient(),
            )

        if resolved_models is not None:
            bridge_data.image_models_to_load = list(set(bridge_data.image_models_to_load + list(resolved_models)))

        if bridge_data.image_models_to_skip is not None and len(bridge_data.image_models_to_skip) > 0:
            bridge_data.image_models_to_load = list(
                set(bridge_data.image_models_to_load) - set(bridge_data.image_models_to_skip),
            )

        return bridge_data.image_models_to_load
