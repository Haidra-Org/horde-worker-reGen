from __future__ import annotations

import enum
from enum import auto

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateJobPopResponse,
)
from horde_sdk.ai_horde_api.fields import JobID
from loguru import logger
from pydantic import BaseModel, model_validator


class ModelLoadState(enum.Enum):
    DOWNLOADING = auto()
    ON_DISK = auto()
    LOADING = auto()
    LOADED_IN_RAM = auto()
    LOADED_IN_VRAM = auto()
    IN_USE = auto()

    def is_loaded(self) -> bool:
        return (
            self == ModelLoadState.LOADED_IN_RAM
            or self == ModelLoadState.LOADED_IN_VRAM
            or self == ModelLoadState.IN_USE
        )


class ModelInfo(BaseModel):
    horde_model_name: str
    horde_model_load_state: ModelLoadState
    process_id: int


class HordeProcessState(enum.Enum):
    PROCESS_STARTING = auto()

    PROCESS_ENDING = auto()
    PROCESS_ENDED = auto()

    WAITING_FOR_JOB = auto()
    JOB_RECEIVED = auto()

    DOWNLOADING_MODEL = auto()
    DOWNLOAD_COMPLETE = auto()

    PRELOADING_MODEL = auto()
    PRELOADED_MODEL = auto()

    UNLOADED_MODEL_FROM_VRAM = auto()
    UNLOADED_MODEL_FROM_RAM = auto()

    INFERENCE_STARTING = auto()
    INFERENCE_COMPLETE = auto()
    INFERENCE_FAILED = auto()

    ALCHEMY_STARTING = auto()
    ALCHEMY_COMPLETE = auto()
    ALCHEMY_FAILED = auto()

    EVALUATING_SAFETY = auto()
    SAFETY_FAILED = auto()


class HordeProcessMessage(BaseModel):
    """Process messages are sent from the child processes to the main process."""

    process_id: int
    info: str
    time_elapsed: float | None = None


class HordeProcessMemoryMessage(HordeProcessMessage):
    ram_usage_bytes: int

    vram_usage_bytes: int | None = None
    vram_total_bytes: int | None = None


class HordeProcessHeartbeatMessage(HordeProcessMessage):
    pass


class HordeProcessStateChangeMessage(HordeProcessMessage):
    process_state: HordeProcessState


class HordeModelStateChangeMessage(HordeProcessStateChangeMessage):
    horde_model_name: str
    horde_model_state: ModelLoadState


class HordeDownloadProgressMessage(HordeModelStateChangeMessage):
    total_downloaded_bytes: int
    total_bytes: int

    @property
    def progress_percent(self) -> float:
        return self.total_downloaded_bytes / self.total_bytes * 100


class HordeDownloadCompleteMessage(HordeModelStateChangeMessage):
    pass


class HordeInferenceResultMessage(HordeProcessMessage):
    job_result_images_base64: list[str] | None = None
    state: GENERATION_STATE
    job_info: ImageGenerateJobPopResponse


class HordeSafetyEvaluation(BaseModel):
    is_nsfw: bool
    is_csam: bool
    replacement_image_base64: str | None
    failed: bool = False


class HordeSafetyResultMessage(HordeProcessMessage):
    job_id: JobID
    safety_evaluations: list[HordeSafetyEvaluation]


class HordeControlFlag(enum.Enum):
    DOWNLOAD_MODEL = auto()
    PRELOAD_MODEL = auto()
    START_INFERENCE = auto()
    EVALUATE_SAFETY = auto()
    UNLOAD_MODELS_FROM_VRAM = auto()
    UNLOAD_MODELS_FROM_RAM = auto()
    END_PROCESS = auto()


class HordeControlMessage(BaseModel):
    """Control messages are sent from the main process to the child processes."""

    control_flag: HordeControlFlag


class HordeControlModelMessage(HordeControlMessage):
    horde_model_name: str


class HordePreloadInferenceModelMessage(HordeControlModelMessage):
    will_load_loras: bool
    seamless_tiling_enabled: bool


class HordeInferenceControlMessage(HordeControlModelMessage):
    job_info: ImageGenerateJobPopResponse


class HordeSafetyControlMessage(HordeControlMessage):
    job_id: JobID
    prompt: str
    censor_nsfw: bool
    sfw_worker: bool
    images_base64: list[str]
    horde_model_info: dict

    @model_validator(mode="after")
    def validate_censor_flags_logical(self) -> HordeSafetyControlMessage:
        if not self.censor_nsfw and self.sfw_worker:
            logger.warning("HordeSafetyControlMessage: sfw_worker is True but censor_nsfw is False")
            self.censor_nsfw = True

        return self
