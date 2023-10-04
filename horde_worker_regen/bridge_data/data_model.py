import os

from horde_sdk.ai_horde_worker.bridge_data import CombinedHordeBridgeData
from loguru import logger
from pydantic import Field
from ruamel.yaml import YAML

from horde_worker_regen.locale_info.regen_bridge_data_fields import BRIDGE_DATA_FIELD_DESCRIPTIONS


class reGenBridgeData(CombinedHordeBridgeData):
    disable_terminal_ui: bool = Field(
        default=True,
    )

    safety_on_gpu: bool = Field(
        default=False,
    )

    _yaml_loader: YAML | None = None

    def load_env_vars(self) -> None:
        """Load the environment variables into the config model."""
        if self.models_folder_parent:
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
                else:
                    os.environ["AI_HORDE_URL"] = self.horde_url

        if self.max_lora_cache_size:
            os.environ["AIWORKER_LORA_CACHE_SIZE"] = str(self.max_lora_cache_size * 1024)

    def save(self, file_path: str) -> None:
        """Save the config model to a file.

        Args:
            file_path (str): The path to the file to save the config model to.
        """

        if not self._yaml_loader:
            self._yaml_loader = YAML()

        with open(file_path, "w") as f:
            self._yaml_loader.dump(self.model_dump(), f)


# Dynamically add descriptions to the fields of the model
for field_name, field in reGenBridgeData.model_fields.items():
    if field_name in BRIDGE_DATA_FIELD_DESCRIPTIONS:
        field.description = BRIDGE_DATA_FIELD_DESCRIPTIONS[field_name]
