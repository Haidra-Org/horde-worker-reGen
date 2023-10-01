from __future__ import annotations

import base64
import contextlib
import enum
import io
import time
from enum import auto

try:
    from multiprocessing.connection import PipeConnection as Connection
except ImportError:
    from multiprocessing.connection import Connection  # type: ignore
from multiprocessing.synchronize import Lock, Semaphore
from typing import TYPE_CHECKING

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateJobPopResponse,
)
from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager
from loguru import logger
from PIL.Image import Image
from typing_extensions import override

from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.horde_process import HordeProcess
from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeControlModelMessage,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
    HordeModelStateChangeMessage,
    HordePreloadInferenceModelMessage,
    HordeProcessState,
    ModelLoadState,
)

if TYPE_CHECKING:
    from hordelib.nodes.node_model_loader import HordeCheckpointLoader
else:
    # Create a dummy class to prevent type errors
    class HordeCheckpointLoader:
        pass


class HordeProcessKind(enum.Enum):
    INFERENCE = auto()
    SAFETY = auto()


class HordeInferenceProcess(HordeProcess):
    _inference_semaphore: Semaphore
    """A semaphore used to limit the number of concurrent inference jobs."""

    _horde: HordeLib
    """The HordeLib instance used by this process. It is not shared between processes."""
    _shared_model_manager: SharedModelManager
    """The SharedModelManager instance used by this process. It is not shared between processes (despite the name)."""
    _checkpoint_loader: HordeCheckpointLoader

    _active_model_name: str | None = None

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        inference_semaphore: Semaphore,
        disk_lock: Lock,
    ) -> None:
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
        )

        import hordelib

        hordelib.initialise(
            setup_logging=None,
            process_id=self.process_id,
            logging_verbosity=0,
        )

        self._inference_semaphore = inference_semaphore

        from hordelib.nodes.node_model_loader import HordeCheckpointLoader

        self._horde = HordeLib()
        self._shared_model_manager = SharedModelManager()
        self._checkpoint_loader = HordeCheckpointLoader()

        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    def on_horde_model_state_change(
        self,
        horde_model_name: str,
        process_state: HordeProcessState,
        horde_model_state: ModelLoadState,
    ) -> None:
        """Update the main process with the current process state and model state."""
        self.send_memory_report_message(include_vram=True)

        model_update_message = HordeModelStateChangeMessage(
            process_state=process_state,
            process_id=self.process_id,
            info=f"Model {horde_model_name} {horde_model_state.name}",
            horde_model_name=horde_model_name,
            horde_model_state=horde_model_state,
            time_elapsed=self._last_message_time - time.time(),
        )
        self.process_message_queue.put(model_update_message)

        self.send_process_state_change_message(
            process_state=process_state,
            info=f"Model {horde_model_name} {horde_model_state.name}",
        )

    def download_callback(self, downloaded_bytes: int, total_bytes: int) -> None:
        if downloaded_bytes % (total_bytes / 20) == 0:
            self.send_process_state_change_message(
                process_state=HordeProcessState.DOWNLOADING_MODEL,
                info=f"Downloading model ({downloaded_bytes} / {total_bytes})",
            )

    def download_model(self, horde_model_name: str) -> None:
        self.send_process_state_change_message(
            process_state=HordeProcessState.DOWNLOADING_MODEL,
            info=f"Downloading model {horde_model_name}",
        )

        if self._shared_model_manager.manager.is_model_available(horde_model_name):
            logger.info(f"Model {horde_model_name} already downloaded")

        success = self._shared_model_manager.manager.download_model(horde_model_name, self.download_callback)

        if success:
            self.send_process_state_change_message(
                process_state=HordeProcessState.DOWNLOAD_COMPLETE,
                info=f"Downloaded model {horde_model_name}",
            )

        self.on_horde_model_state_change(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            horde_model_name=horde_model_name,
            horde_model_state=ModelLoadState.ON_DISK,
        )

    def preload_model(
        self,
        horde_model_name: str,
        will_load_loras: bool,
        seamless_tiling_enabled: bool,
    ) -> None:
        if self._active_model_name == horde_model_name:
            return

        logger.debug(f"Currently active model is {self._active_model_name}")
        logger.debug(f"Preloading model {horde_model_name}")

        if self._active_model_name is not None:
            self.on_horde_model_state_change(
                process_state=HordeProcessState.UNLOADED_MODEL_FROM_RAM,
                horde_model_name=self._active_model_name,
                horde_model_state=ModelLoadState.ON_DISK,
            )

        self.on_horde_model_state_change(
            process_state=HordeProcessState.PRELOADING_MODEL,
            horde_model_name=horde_model_name,
            horde_model_state=ModelLoadState.LOADING,
        )

        with contextlib.nullcontext():  # self.disk_lock:
            self._checkpoint_loader.load_checkpoint(
                will_load_loras=will_load_loras,
                seamless_tiling_enabled=seamless_tiling_enabled,
                horde_model_name=horde_model_name,
                preloading=True,
            )

        logger.info(f"Preloaded model {horde_model_name}")
        self._active_model_name = horde_model_name
        self.on_horde_model_state_change(
            process_state=HordeProcessState.PRELOADED_MODEL,
            horde_model_name=horde_model_name,
            horde_model_state=ModelLoadState.LOADED_IN_RAM,
        )

        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info=f"Preloaded model {horde_model_name}",
        )

    def start_inference(self, job_info: ImageGenerateJobPopResponse) -> list[Image] | None:
        with self._inference_semaphore:
            return self._horde.basic_inference(job_info)

    def unload_models_from_vram(self) -> None:
        from hordelib.comfy_horde import unload_all_models_vram

        unload_all_models_vram()
        if self._active_model_name is not None:
            self.on_horde_model_state_change(
                process_state=HordeProcessState.UNLOADED_MODEL_FROM_VRAM,
                horde_model_name=self._active_model_name,
                horde_model_state=ModelLoadState.LOADED_IN_RAM,
            )

            self.send_process_state_change_message(
                process_state=HordeProcessState.WAITING_FOR_JOB,
                info="Unloaded models from VRAM",
            )
        else:
            self.send_process_state_change_message(
                process_state=HordeProcessState.WAITING_FOR_JOB,
                info="No models to unload from VRAM",
            )

    def unload_models_from_ram(self) -> None:
        from hordelib.comfy_horde import unload_all_models_ram

        unload_all_models_ram()
        self.send_memory_report_message(include_vram=True)
        if self._active_model_name is not None:
            self.on_horde_model_state_change(
                process_state=HordeProcessState.UNLOADED_MODEL_FROM_RAM,
                horde_model_name=self._active_model_name,
                horde_model_state=ModelLoadState.ON_DISK,
            )

            self.send_process_state_change_message(
                process_state=HordeProcessState.WAITING_FOR_JOB,
                info="Unloaded models from RAM",
            )
        else:
            self.send_process_state_change_message(
                process_state=HordeProcessState.WAITING_FOR_JOB,
                info="No models to unload from RAM",
            )
        logger.info("Unloaded all models from RAM")
        self._active_model_name = None

    def cleanup_and_exit(self) -> None:
        self.unload_models_from_ram()
        self.send_process_state_change_message(
            process_state=HordeProcessState.PROCESS_ENDED,
            info="Process ended",
        )

    def send_inference_result_message(
        self,
        process_state: HordeProcessState,
        job_info: ImageGenerateJobPopResponse,
        images: list[Image] | None,
    ) -> None:
        images_as_base64 = []

        if images is not None:
            for image in images:
                buffered_image = io.BytesIO()
                image.save(buffered_image, format="PNG")
                image_base64 = base64.b64encode(buffered_image.getvalue()).decode("utf-8")
                images_as_base64.append(image_base64)

        message = HordeInferenceResultMessage(
            process_id=self.process_id,
            info="Inference result",
            state=GENERATION_STATE.ok if images is not None and len(images) > 0 else GENERATION_STATE.faulted,
            time_elapsed=self._last_message_time - time.time(),
            job_result_images_base64=images_as_base64,
            job_info=job_info,
        )
        self.process_message_queue.put(message)
        self._last_message_time = time.time()

        if self._active_model_name is None:
            logger.critical("No active model name, cannot update model state")
            return

        self.on_horde_model_state_change(
            process_state=process_state,
            horde_model_name=self._active_model_name,
            horde_model_state=ModelLoadState.LOADED_IN_VRAM,
        )

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        logger.debug(f"Received ({type(message)}): {message.control_flag}")

        if isinstance(message, HordePreloadInferenceModelMessage):
            self.preload_model(
                horde_model_name=message.horde_model_name,
                will_load_loras=message.will_load_loras,
                seamless_tiling_enabled=message.seamless_tiling_enabled,
            )
        elif isinstance(message, HordeInferenceControlMessage):
            if message.control_flag == HordeControlFlag.START_INFERENCE:
                if message.horde_model_name != self._active_model_name:
                    error_message = f"Received START_INFERENCE control message for model {message.horde_model_name} "
                    error_message += f"but currently active model is {self._active_model_name}"
                    logger.error(error_message)

                    self.send_process_state_change_message(
                        process_state=HordeProcessState.INFERENCE_FAILED,
                        info=error_message,
                    )

                self.send_process_state_change_message(
                    process_state=HordeProcessState.INFERENCE_STARTING,
                    info=f"Starting inference for {message.job_info.id_} with model {message.horde_model_name}",
                )

                images = self.start_inference(message.job_info)
                process_state = HordeProcessState.INFERENCE_COMPLETE if images else HordeProcessState.INFERENCE_FAILED
                logger.debug(f"Finished inference with process state {process_state}")
                self.send_inference_result_message(
                    process_state=process_state,
                    job_info=message.job_info,
                    images=images,
                )
            else:
                logger.critical(f"Received unexpected message: {message}")
                return
        elif isinstance(message, HordeControlModelMessage):
            if message.control_flag == HordeControlFlag.DOWNLOAD_MODEL:
                self.download_model(message.horde_model_name)
            elif message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                self.unload_models_from_vram()
            elif message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
                self.unload_models_from_ram()
            else:
                logger.critical(f"Received unexpected message: {message}")
                return

        elif message.control_flag == HordeControlFlag.END_PROCESS:
            self.send_process_state_change_message(
                process_state=HordeProcessState.PROCESS_ENDING,
                info="Process stopping",
            )

            self._end_process = True
