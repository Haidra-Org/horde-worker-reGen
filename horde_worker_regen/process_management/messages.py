"""Contains messages (and associated helper types) used for communication between the main process and the child processes."""  # noqa: E501
from __future__ import annotations

import enum
from enum import auto

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import (
    GenMetadataEntry,
    ImageGenerateJobPopResponse,
)
from horde_sdk.ai_horde_api.fields import JobID
from loguru import logger
from pydantic import BaseModel, Field, model_validator


class ModelLoadState(enum.Enum):
    """The state of a model.

    e.g., if a model is `IN_USE` or `LOADED_IN_VRAM`
    """

    DOWNLOADING = auto()
    """The model is being downloaded."""
    ON_DISK = auto()
    """The model is on disk. It may or may not be loaded in RAM."""  # TODO: this caveat is subject to change
    LOADING = auto()
    """The model is being loaded into RAM."""
    LOADED_IN_RAM = auto()
    """The model is loaded in RAM."""
    LOADED_IN_VRAM = auto()
    """The model is loaded in VRAM."""
    IN_USE = auto()
    """The model is in use by a process."""

    def is_loaded(self) -> bool:
        """Return if the model is loaded in RAM, VRAM, or in use."""
        return (
            self == ModelLoadState.LOADED_IN_RAM
            or self == ModelLoadState.LOADED_IN_VRAM
            or self == ModelLoadState.IN_USE
        )


class ModelInfo(BaseModel):
    """Information about a model loaded or used by a process."""

    horde_model_name: str
    """The name of the model as defined in the horde model reference."""
    horde_model_load_state: ModelLoadState
    """The state of the model."""
    process_id: int
    """The ID of the process that is using the model."""


class HordeProcessState(enum.Enum):
    """The state of a process.

    e.g., if a process is `INFERENCE_STARTING` or `WAITING_FOR_JOB`
    """

    PROCESS_STARTING = auto()
    """The process is starting."""

    PROCESS_ENDING = auto()
    """The process is ending."""
    PROCESS_ENDED = auto()
    """The process has ended."""

    WAITING_FOR_JOB = auto()
    """The process is waiting for a job."""
    JOB_RECEIVED = auto()
    """The process has received a job."""

    DOWNLOADING_MODEL = auto()
    """The process is downloading a model."""
    DOWNLOAD_COMPLETE = auto()
    """The process has finished downloading a model."""

    DOWNLOADING_AUX_MODEL = auto()
    """The process is downloading an auxiliary model. (e.g., LORA)"""
    DOWNLOAD_AUX_COMPLETE = auto()
    """The process has finished downloading an auxiliary model. (e.g., LORA)"""

    PRELOADING_MODEL = auto()
    """The process is preloading a model."""
    PRELOADED_MODEL = auto()
    """The process has finished preloading a model."""

    UNLOADED_MODEL_FROM_VRAM = auto()
    """The process has unloaded a model from VRAM."""
    UNLOADED_MODEL_FROM_RAM = auto()
    """The process has unloaded a model from RAM."""

    INFERENCE_STARTING = auto()
    """The process is starting inference."""
    INFERENCE_COMPLETE = auto()
    """The process has finished inference."""
    INFERENCE_FAILED = auto()
    """The process has failed inference."""

    ALCHEMY_STARTING = auto()
    """The process is starting performing alchemy jobs."""
    ALCHEMY_COMPLETE = auto()
    """The process has finished performing alchemy jobs."""
    ALCHEMY_FAILED = auto()
    """The process has failed performing alchemy jobs."""

    EVALUATING_SAFETY = auto()
    """The process is evaluating safety."""
    SAFETY_FAILED = auto()
    """The process has failed evaluating safety."""


class HordeProcessMessage(BaseModel):
    """Process messages are sent from the child processes to the main process."""

    process_id: int
    """The ID of the process that sent the message."""
    info: str
    """Information about the process."""
    time_elapsed: float | None = None
    """The time elapsed since the process started."""


class HordeProcessMemoryMessage(HordeProcessMessage):
    """Memory messages that are sent from the child processes to the main process."""

    ram_usage_bytes: int
    """The number of bytes of RAM used by the process."""

    vram_usage_bytes: int | None = None
    """The number of bytes of VRAM used by the GPU."""
    vram_total_bytes: int | None = None
    """The total number of bytes of VRAM available on the GPU."""


class HordeProcessHeartbeatMessage(HordeProcessMessage):
    """Heartbeat messages that are sent from the child processes to the main process."""


class HordeProcessStateChangeMessage(HordeProcessMessage):
    """State change messages that are sent from the child processes to the main process."""

    process_state: HordeProcessState
    """The state of the process."""


class HordeModelStateChangeMessage(HordeProcessStateChangeMessage):
    """Model state change messages that are sent from the child processes to the main process.

    See also `ModelLoadState`.
    """

    horde_model_name: str
    """The name of the model as defined in the horde model reference."""
    horde_model_state: ModelLoadState
    """The state of the model."""


class HordeAuxModelStateChangeMessage(HordeProcessStateChangeMessage):
    """Auxiliary model state change messages that are sent from the child processes to the main process.

    See also `ModelLoadState`.
    """

    sdk_api_job_info: ImageGenerateJobPopResponse | None = None
    """If the model state change is related to a job, the job as sent by the API."""


class HordeDownloadProgressMessage(HordeModelStateChangeMessage):
    """Download progress messages that are sent from the child processes to the main process."""

    total_downloaded_bytes: int
    """The total number of bytes downloaded so far."""
    total_bytes: int
    """The total number of bytes that will be downloaded."""

    @property
    def progress_percent(self) -> float:
        """The progress of the download as a percentage."""
        return self.total_downloaded_bytes / self.total_bytes * 100


class HordeDownloadCompleteMessage(HordeModelStateChangeMessage):
    """Download complete messages that are sent from the child processes to the main process."""


class HordeImageResult(BaseModel):
    """Contains information about a single image that has been generated in a job."""

    image_base64: str
    """The base64 strings of one image generated by the job."""
    generation_faults: list[GenMetadataEntry] = Field(default_factory=list)
    """The generation faults recorded for that image."""


class HordeInferenceResultMessage(HordeProcessMessage):
    """Inference result messages that are sent from the child processes to the main process."""

    job_image_results: list[HordeImageResult] | None = None
    """The base64 strings of the images generated by the job."""
    state: GENERATION_STATE
    """The state of the job to be sent to the API."""
    sdk_api_job_info: ImageGenerateJobPopResponse

    @property
    def faults_count(self) -> int:
        """Return a count of all generation faults."""

        if self.job_image_results is None:
            return 0
        total = 0
        for f in self.job_image_results:
            if f.generation_faults is not None:
                total += len(f.generation_faults)
        return total


class HordeSafetyEvaluation(BaseModel):
    """The result of a safety evaluation."""

    is_nsfw: bool
    """If the image is NSFW."""
    is_csam: bool
    """If the image is CSAM."""
    replacement_image_base64: str | None
    """The base64 string of the replacement image if it was censored."""
    failed: bool = False
    """If the safety evaluation failed."""


class HordeSafetyResultMessage(HordeProcessMessage):
    """Safety result messages that are sent from the child processes to the main process."""

    job_id: JobID
    """The ID of the job that was evaluated."""
    safety_evaluations: list[HordeSafetyEvaluation]
    """A list of safety evaluations for each image in the job."""


class HordeControlFlag(enum.Enum):
    """Control flags are sent from the main process to the child processes."""

    DOWNLOAD_MODEL = auto()
    """Signal the child process to download a model."""
    PRELOAD_MODEL = auto()
    """Signal the child process to preload a model."""
    START_INFERENCE = auto()
    """Signal the child process to start inference."""
    EVALUATE_SAFETY = auto()
    """Signal the child process to evaluate safety of images from inference."""
    UNLOAD_MODELS_FROM_VRAM = auto()
    """Signal the child process to unload models from VRAM."""
    UNLOAD_MODELS_FROM_RAM = auto()
    """Signal the child process to unload models from RAM."""
    END_PROCESS = auto()
    """Signal the child process to end."""


class HordeControlMessage(BaseModel):
    """Control messages are sent from the main process to the child processes."""

    control_flag: HordeControlFlag
    """The control flag signaling the child process to perform an action."""


class HordeControlModelMessage(HordeControlMessage):
    """Control messages that are sent from the main process to the child processes that involve models."""

    horde_model_name: str
    """The name of the model as defined in the horde model reference."""


class HordePreloadInferenceModelMessage(HordeControlModelMessage):
    """Preload model (for image generation) messages that are sent from the main process to the child processes."""

    will_load_loras: bool
    """If the model will be patched with LoRa(s)."""
    seamless_tiling_enabled: bool
    """If seamless tiling will be enabled."""

    sdk_api_job_info: ImageGenerateJobPopResponse


class HordeInferenceControlMessage(HordeControlModelMessage):
    """Inference control messages that are sent from the main process to the child processes."""

    sdk_api_job_info: ImageGenerateJobPopResponse
    """The job as sent by the API."""


class HordeSafetyControlMessage(HordeControlMessage):
    """Message with images and other information to be evaluated for safety."""

    job_id: JobID
    """The ID of the job that was evaluated."""
    prompt: str
    """The prompt used to generate the images."""
    censor_nsfw: bool
    """If NSFW images should be censored."""
    sfw_worker: bool
    """If the worker is SFW."""
    images_base64: list[str]
    """The base64 strings of the images generated by the job."""
    horde_model_info: dict
    """The model info as defined in the horde model reference."""

    @model_validator(mode="after")
    def validate_censor_flags_logical(self) -> HordeSafetyControlMessage:
        """Validate that the censor flags are logical (reasonable)."""
        if not self.censor_nsfw and self.sfw_worker:
            logger.warning("HordeSafetyControlMessage: sfw_worker is True but censor_nsfw is False")
            self.censor_nsfw = True

        return self
