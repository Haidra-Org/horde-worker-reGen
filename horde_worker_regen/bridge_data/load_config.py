"""Contains methods for loading the config file."""

import json
import os
import re
from enum import auto
from pathlib import Path

from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPIManualClient
from horde_sdk.ai_horde_worker.model_meta import ImageModelLoadResolver
from loguru import logger
from ruamel.yaml import YAML
from strenum import StrEnum

from horde_worker_regen.bridge_data import AIWORKER_REGEN_PREFIX
from horde_worker_regen.bridge_data.data_model import reGenBridgeData


class UnsupportedConfigFormat(Exception):
    """The config file format is not supported."""

    def __init__(self, file_path: str | Path, file_format: str) -> None:
        """Initialise the exception."""
        super().__init__(f"Unsupported config file format: {file_format} ({file_path})")

    @staticmethod
    def load_from_env_vars(
        *,
        horde_model_reference_manager: ModelReferenceManager,
    ) -> reGenBridgeData:
        """Checks for AIWORKER_REGEN_* format environment variables and loads the config from them."""
        raw_config: dict[str, str] = {}
        for key, value in os.environ.items():
            if key.startswith(AIWORKER_REGEN_PREFIX):
                # Coverts the env var name to the attr name found in the reGenBridgeData model
                raw_config[key[len(AIWORKER_REGEN_PREFIX) :].lower()] = value

        config: dict[str, object] = {}

        for key, value in raw_config.items():
            attr_name = key[len(AIWORKER_REGEN_PREFIX) :].lower()
            if value.lower() in {"true", "false"}:
                config[attr_name] = value.lower() == "true"
            elif any(delimiter in value for delimiter in ["[", ",", ";"]):
                if "[" in value and "]" not in value:
                    raise ValueError(f"Invalid list format for {attr_name}. Missing closing bracket.")
                value_as_list = re.split(r"[\[\],;]", value.strip("[]"))
                config[attr_name] = [item.strip().strip("'").strip('"') for item in value_as_list]
                logger.debug(f"Converted {attr_name} to list: {config[attr_name]} from {value}")
            else:
                config[attr_name] = value

        bridge_data = reGenBridgeData.model_validate(config)

        for set_field in bridge_data.model_fields_set:
            logger.warning(f"AIWORKER_REGEN_{set_field} environment variable set.")

        bridge_data.image_models_to_load = BridgeDataLoader._resolve_meta_instructions(
            bridge_data,
            horde_model_reference_manager,
        )

        return bridge_data

    @staticmethod
    def write_bridge_data_as_dot_env_file(bridge_data: reGenBridgeData, file_path: str | Path) -> None:
        """Write the bridge data to a .env file.

        Args:
            bridge_data (reGenBridgeData): The bridge data to write to the .env file.
            file_path (str | Path): The path to the .env file to write the bridge data to.
        """
        file_path = Path(file_path)

        with open(file_path, "w", encoding="utf-8") as f:
            for field_name, _ in reGenBridgeData.model_fields.items():
                if field_name in bridge_data.model_fields_set:
                    f.write(f"AIWORKER_REGEN_{field_name.upper()}={getattr(bridge_data, field_name)}\n")


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
    def load_from_env_vars(
        *,
        horde_model_reference_manager: ModelReferenceManager | None = None,
    ) -> reGenBridgeData:
        """Checks for AIWORKER_REGEN_* format environment variables and loads the config from them."""
        config: dict[str, object] = {}

        for key, value in os.environ.items():
            if key.startswith(AIWORKER_REGEN_PREFIX):
                # Coverts the env var name to the attr name found in the reGenBridgeData model
                attr_name = key[len(AIWORKER_REGEN_PREFIX) :].lower()
                if value.lower() in ("true", "false"):
                    config[attr_name] = value.lower() == "true"
                elif any(delimiter in value for delimiter in ["[", ",", ";"]):
                    if "[" in value and "]" not in value:
                        raise ValueError(f"Invalid list format for {attr_name}. Missing closing bracket.")
                    value_as_list = re.split(r"[\[\],;]", value.strip("[]"))
                    config[attr_name] = [item.strip().strip("'").strip('"') for item in value_as_list]
                    logger.debug(f"Converted {attr_name} to list: {config[attr_name]} from {value}")
                else:
                    config[attr_name] = value

        for field_name, field_info in reGenBridgeData.model_fields.items():
            if field_info.alias is not None and field_name in config:
                config[field_info.alias] = config.pop(field_name)
                logger.warning(
                    f"Config `{field_name}` was set by an environment variable. "
                    f"However, it is an aliased field in the config file. "
                    f"Renaming to `{field_info.alias}`.",
                )

        # Load the config
        bridge_data = reGenBridgeData.model_validate(config)

        for set_field in bridge_data.model_fields_set:
            if bridge_data.model_extra is not None and set_field in bridge_data.model_extra:
                logger.warning(
                    f"Config `{set_field}` was set by an environment variable. "
                    f"However, it is not a valid field in the config file.",
                )
            logger.info(f"Config `{set_field}` was set by an environment variable.")

        if horde_model_reference_manager is not None:
            bridge_data.image_models_to_load = BridgeDataLoader._resolve_meta_instructions(
                bridge_data,
                horde_model_reference_manager,
            )

        bridge_data._loaded_from_env_vars = True
        return bridge_data

    @staticmethod
    def write_bridge_data_as_dot_env_file(bridge_data: reGenBridgeData, file_path: str | Path) -> None:
        """Write the bridge data to a .env file.

        Args:
            bridge_data (reGenBridgeData): The bridge data to write to the .env file.
            file_path (str | Path): The path to the .env file to write the bridge data to.
        """
        file_path = Path(file_path)

        with open(file_path, "w", encoding="utf-8") as f:
            for field_name, _ in reGenBridgeData.model_fields.items():
                if field_name in bridge_data.model_fields_set:
                    field_info = reGenBridgeData.model_fields[field_name]
                    config_field_value = getattr(bridge_data, field_name)
                    if config_field_value == field_info.default:
                        continue

                    field_alias = field_info.alias

                    f.write(
                        f"{AIWORKER_REGEN_PREFIX}{field_name.upper() if field_alias is None else field_alias.upper()}"
                        f"={config_field_value}\n",
                    )

    @staticmethod
    def _resolve_meta_instructions(  # FIXME: This should be moved into the SDK
        bridge_data: reGenBridgeData,
        horde_model_reference_manager: ModelReferenceManager,
    ) -> list[str]:
        """Resolve the meta instructions in the bridge data. Note that this modifies the bridge data in place.

        Args:
            bridge_data (reGenBridgeData): The bridge data.
            horde_model_reference_manager (ModelReferenceManager): The model reference manager.

        Returns:
            list[str]: The image models that will be loaded.
        """
        load_resolver = ImageModelLoadResolver(horde_model_reference_manager)

        resolved_models = None
        if bridge_data.meta_load_instructions is not None:
            resolved_models = load_resolver.resolve_meta_instructions(
                list(bridge_data.meta_load_instructions),
                AIHordeAPIManualClient(),
            )

        if bridge_data.meta_skip_instructions is not None:
            skip_models: set[str] = load_resolver.resolve_meta_instructions(
                list(bridge_data.meta_skip_instructions),
                AIHordeAPIManualClient(),
            )
            existing_skip_models = set(bridge_data.image_models_to_skip)
            bridge_data.image_models_to_skip = list(existing_skip_models.union(skip_models))

        if resolved_models is not None:
            bridge_data.image_models_to_load = list(set(bridge_data.image_models_to_load + list(resolved_models)))

        if bridge_data.image_models_to_skip is not None and len(bridge_data.image_models_to_skip) > 0:
            bridge_data.image_models_to_load = list(
                set(bridge_data.image_models_to_load) - set(bridge_data.image_models_to_skip),
            )

        return bridge_data.image_models_to_load
