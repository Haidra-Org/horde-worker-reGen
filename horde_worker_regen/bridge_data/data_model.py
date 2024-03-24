"""The config model and initializers for the reGen configuration model."""

from __future__ import annotations

import os

from horde_sdk.ai_horde_worker.bridge_data import CombinedHordeBridgeData
from loguru import logger
from pydantic import Field, model_validator
from ruamel.yaml import YAML

from horde_worker_regen.consts import TOTAL_LORA_DOWNLOAD_TIMEOUT
from horde_worker_regen.locale_info.regen_bridge_data_fields import BRIDGE_DATA_FIELD_DESCRIPTIONS


class reGenBridgeData(CombinedHordeBridgeData):
    """The config model for reGen. Extra fields added here are specific to this worker implementation.

    See `CombinedHordeBridgeData` from the SDK for more information..
    """

    _loaded_from_env_vars: bool = False

    disable_terminal_ui: bool = Field(
        default=True,
    )

    safety_on_gpu: bool = Field(
        default=False,
    )

    _yaml_loader: YAML | None = None

    cycle_process_on_model_change: bool = Field(
        default=False,
    )

    CIVIT_API_TOKEN: str | None = Field(
        default=None,
        alias="civitai_api_token",
    )
    unload_models_from_vram: bool = Field(default=True)

    process_timeout: int = Field(default=900)
    """The maximum amount of time to allow a job to run before it is killed"""

    post_process_timeout: int = Field(default=60, ge=15)

    download_timeout: int = Field(default=TOTAL_LORA_DOWNLOAD_TIMEOUT + 1)
    preload_timeout: int = Field(default=60, ge=15)

    horde_model_stickiness: float = Field(default=0.0, le=1.0, ge=0.0, alias="model_stickiness")
    """
    A percent chance (expressed as a decimal between 0 and 1) that the currently loaded models will
    be favored when popping a job.
    """

    high_memory_mode: bool = Field(default=False)

    very_high_memory_mode: bool = Field(default=False)

    high_performance_mode: bool = Field(default=False)
    """If you have a 4090 or better, set this to true to enable high performance mode."""

    moderate_performance_mode: bool = Field(default=False)
    """If you have a 3080 or better, set this to true to enable moderate performance mode."""

    capture_kudos_training_data: bool = Field(default=False)
    kudos_training_data_file: str | None = Field(default=None)

    exit_on_unhandled_faults: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_performance_modes(self) -> reGenBridgeData:
        """Validate the performance modes and set the appropriate values.

        Returns:
            reGenBridgeData: The config model with the performance modes set appropriately.
        """
        if self.max_threads == 2 and self.queue_size > 3:
            self.queue_size = 3
            logger.warning(
                "The queue_size value has been set to 3 because the max_threads value is 2.",
            )

        if self.max_threads > 2 and self.queue_size > 2:
            self.queue_size = 2
            logger.warning(
                "The queue_size value has been set to 2 because the max_threads value is greater than 2.",
            )

        if self.very_high_memory_mode and not self.high_memory_mode:
            self.high_memory_mode = True
            logger.warning(
                "Very high memory mode is enabled, so the high_memory_mode value has been set to True.",
            )

        if self.high_memory_mode and not self.very_high_memory_mode:
            if self.max_threads != 1:
                self.max_threads = 1
                logger.warning(
                    "High memory mode is enabled, so the max_threads value has been set to 1.",
                )

            if self.queue_size == 0:
                logger.warning(
                    "High memory mode is enabled and works best with a queue_size of 1.",
                )

            if self.queue_size > 1:
                self.queue_size = 1
                logger.warning(
                    "High memory mode is enabled, so the queue_size value has been set to 1.",
                )

            if self.cycle_process_on_model_change:
                self.cycle_process_on_model_change = False
                logger.warning(
                    "High memory mode is enabled, so the cycle_process_on_model_change value has been set to False.",
                )

        return self

    def load_env_vars(self) -> None:
        """Load the environment variables into the config model."""
        # See load_env_vars.py's `def load_env_vars(self) -> None:`
        if self.models_folder_parent and os.getenv("AIWORKER_CACHE_HOME") is None:
            os.environ["AIWORKER_CACHE_HOME"] = self.models_folder_parent
        if self.horde_url:
            if os.environ.get("AI_HORDE_URL"):
                logger.warning(
                    "AI_HORDE_URL environment variable already set. This will override the value for `horde_url` in "
                    "the config file.",
                )
            else:
                if os.environ.get("AI_HORDE_DEV_URL"):
                    logger.warning(
                        "AI_HORDE_DEV_URL environment variable already set. This will override the value for "
                        "`horde_url` in the config file.",
                    )
                if os.environ.get("AI_HORDE_URL") is None:
                    os.environ["AI_HORDE_URL"] = self.horde_url
                else:
                    logger.warning(
                        "AI_HORDE_URL environment variable already set. This will override the value for `horde_url` "
                        "in the config file.",
                    )

        if self.CIVIT_API_TOKEN is not None:
            os.environ["CIVIT_API_TOKEN"] = self.CIVIT_API_TOKEN

        if self.max_lora_cache_size and os.getenv("AIWORKER_LORA_CACHE_SIZE") is None:
            os.environ["AIWORKER_LORA_CACHE_SIZE"] = str(self.max_lora_cache_size * 1024)

    def save(self, file_path: str) -> None:
        """Save the config model to a file.

        Args:
            file_path (str): The path to the file to save the config model to.
        """
        if self._yaml_loader is None:
            self._yaml_loader = YAML()

        with open(file_path, "w", encoding="utf-8") as f:
            self._yaml_loader.dump(self.model_dump(), f)


# Dynamically add descriptions to the fields of the model
for field_name, field in reGenBridgeData.model_fields.items():
    if field_name in BRIDGE_DATA_FIELD_DESCRIPTIONS:
        field.description = BRIDGE_DATA_FIELD_DESCRIPTIONS[field_name]
