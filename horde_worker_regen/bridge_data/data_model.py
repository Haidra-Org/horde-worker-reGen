import os

from horde_sdk.ai_horde_worker.bridge_data import CombinedHordeBridgeData
from loguru import logger
from pydantic import Field

from horde_worker_regen.locale_info.regen_bridge_data_fields import BRIDGE_DATA_FIELD_DESCRIPTIONS


class reGenBridgeData(CombinedHordeBridgeData):
    disable_terminal_ui: bool = Field(
        value=True,
    )

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


# Dynamically add descriptions to the fields of the model
for field_name, field in reGenBridgeData.model_fields.items():
    if field_name in BRIDGE_DATA_FIELD_DESCRIPTIONS:
        field.description = BRIDGE_DATA_FIELD_DESCRIPTIONS[field_name]
