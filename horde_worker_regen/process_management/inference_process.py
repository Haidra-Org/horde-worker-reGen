"""Contains the classes to form an inference process, which generate images."""

from __future__ import annotations

import base64
import contextlib
import io
import sys
import time

from horde_worker_regen.consts import BASE_LORA_DOWNLOAD_TIMEOUT, EXTRA_LORA_DOWNLOAD_TIMEOUT

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore
from multiprocessing.synchronize import Lock, Semaphore
from typing import TYPE_CHECKING

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateJobPopResponse,
)
from loguru import logger
from typing_extensions import override

from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.horde_process import HordeProcess
from horde_worker_regen.process_management.messages import (
    HordeAuxModelStateChangeMessage,
    HordeControlFlag,
    HordeControlMessage,
    HordeControlModelMessage,
    HordeHeartbeatType,
    HordeImageResult,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
    HordeModelStateChangeMessage,
    HordePreloadInferenceModelMessage,
    HordeProcessState,
    ModelLoadState,
)

if TYPE_CHECKING:
    from hordelib.horde import HordeLib, ProgressReport, ResultingImageReturn
    from hordelib.nodes.node_model_loader import HordeCheckpointLoader
    from hordelib.shared_model_manager import SharedModelManager
else:
    # Create a dummy class to prevent type errors at runtime
    # This is so we can defer the import of these classes until runtime
    class HordeCheckpointLoader:  # noqa
        pass

    class HordeLib:  # noqa
        pass

    class SharedModelManager:  # noqa
        pass

    class ProgressReport:  # noqa
        pass


class HordeInferenceProcess(HordeProcess):
    """Represents an inference process, which generates images."""

    _inference_semaphore: Semaphore
    """A semaphore used to limit the number of concurrent inference jobs."""

    _horde: HordeLib
    """The HordeLib instance used by this process. It is not shared between processes."""
    _shared_model_manager: SharedModelManager
    """The SharedModelManager instance used by this process. It is not shared between processes (despite the name)."""
    _checkpoint_loader: HordeCheckpointLoader
    """The HordeCheckpointLoader instance used by this process. This is horde hordelib signals comfyui \
        to load a model. It is not shared between processes."""

    _active_model_name: str | None = None
    """The name of the currently active model. Note that other models may be loaded in RAM or VRAM."""
    _aux_model_lock: Lock

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        inference_semaphore: Semaphore,
        aux_model_lock: Lock,
        disk_lock: Lock,
    ) -> None:
        """Initialise the HordeInferenceProcess.

        Args:
            process_id (int): The ID of the process. This is not the same as the process PID.
            process_message_queue (ProcessQueue): The queue the main process uses to receive messages from all worker \
                processes.
            pipe_connection (Connection): Receives `HordeControlMessage`s from the main process.
            inference_semaphore (Semaphore): A semaphore used to limit the number of concurrent inference jobs.
            aux_model_lock (Lock): A lock used to prevent multiple processes from downloading auxiliary models at the \
            disk_lock (Lock): A lock used to prevent multiple processes from accessing disk at the same time.
        """
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
        )

        self._aux_model_lock = aux_model_lock

        # We import these here to guard against potentially importing them in the main process
        # which would create shared objects, potentially causing issues
        try:
            with logger.catch(reraise=True):
                from hordelib.horde import HordeLib
                from hordelib.shared_model_manager import SharedModelManager
        except Exception as e:
            logger.critical(f"Failed to import HordeLib or SharedModelManager: {type(e).__name__} {e}")
            sys.exit(1)

        self._inference_semaphore = inference_semaphore

        from hordelib.nodes.node_model_loader import HordeCheckpointLoader

        try:
            with logger.catch(reraise=True):
                self._horde = HordeLib(comfyui_callback=self._comfyui_callback)
                self._shared_model_manager = SharedModelManager(do_not_load_model_mangers=True)
        except Exception as e:
            logger.critical(f"Failed to initialise HordeLib: {type(e).__name__} {e}")
            sys.exit(1)

        try:
            with logger.catch(reraise=True):
                self._checkpoint_loader = HordeCheckpointLoader()
        except Exception as e:
            logger.critical(f"Failed to initialise HordeCheckpointLoader: {type(e).__name__} {e}")
            sys.exit(1)

        SharedModelManager.load_model_managers(multiprocessing_lock=self.disk_lock)

        if SharedModelManager.manager.compvis is None:
            logger.critical("Failed to initialise SharedModelManager")
            self.send_process_state_change_message(
                process_state=HordeProcessState.PROCESS_ENDED,
                info="Failed to initialise compvis in SharedModelManager",
            )
            sys.exit(1)

        if len(SharedModelManager.manager.compvis.available_models) == 0:
            logger.critical("No models available in SharedModelManager")
            self.send_process_state_change_message(
                process_state=HordeProcessState.PROCESS_ENDED,
                info="No models available in SharedModelManager",
            )
            sys.exit(1)

        logger.info("HordeInferenceProcess initialised")

        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    def _comfyui_callback(self, label: str, data: dict, _id: str) -> None:
        self.send_heartbeat_message(heartbeat_type=HordeHeartbeatType.PIPELINE_STATE_CHANGE)

    @logger.catch(reraise=True)
    def on_horde_model_state_change(
        self,
        horde_model_name: str,
        process_state: HordeProcessState,
        horde_model_state: ModelLoadState,
        time_elapsed: float | None = None,
    ) -> None:
        """Update the main process with the current process state and model state.

        Args:
            horde_model_name (str): The name of the model.
            process_state (HordeProcessState): The state of the process.
            horde_model_state (ModelLoadState): The state of the model.
            time_elapsed (float | None, optional): The time elapsed during the last operation, if applicable. \
                Defaults to None.
        """
        self.send_memory_report_message(include_vram=True)

        model_update_message = HordeModelStateChangeMessage(
            process_state=process_state,
            process_id=self.process_id,
            info=f"Model {horde_model_name} {horde_model_state.name}",
            horde_model_name=horde_model_name,
            horde_model_state=horde_model_state,
            time_elapsed=time_elapsed,
        )
        self.process_message_queue.put(model_update_message)

        self.send_process_state_change_message(
            process_state=process_state,
            info=f"Model {horde_model_name} {horde_model_state.name}",
        )
        self.send_memory_report_message(include_vram=True)

    def download_callback(
        self,
        downloaded_bytes: int,
        total_bytes: int,
    ) -> None:
        """Handle the callback for progress when a model is being downloaded.

        Args:
            downloaded_bytes (int): The number of bytes downloaded so far.
            total_bytes (int): The total number of bytes to download.
        """
        # TODO
        if downloaded_bytes % (total_bytes / 20) == 0:
            self.send_process_state_change_message(
                process_state=HordeProcessState.DOWNLOADING_MODEL,
                info=f"Downloading model ({downloaded_bytes} / {total_bytes})",
            )

    def download_model(self, horde_model_name: str) -> None:
        """Download a model as defined in the horde model reference.

        Args:
            horde_model_name (str): The name of the model to download.\
        """
        # TODO
        self.send_process_state_change_message(
            process_state=HordeProcessState.DOWNLOADING_MODEL,
            info=f"Downloading model {horde_model_name}",
        )

        if self._shared_model_manager.manager.is_model_available(horde_model_name):
            logger.info(f"Model {horde_model_name} already downloaded")

        time_start = time.time()

        success = self._shared_model_manager.manager.download_model(horde_model_name, self.download_callback)

        if success:
            self.send_process_state_change_message(
                process_state=HordeProcessState.DOWNLOAD_COMPLETE,
                info=f"Downloaded model {horde_model_name}",
                time_elapsed=time.time() - time_start,
            )

        self.on_horde_model_state_change(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            horde_model_name=horde_model_name,
            horde_model_state=ModelLoadState.ON_DISK,
        )

    @logger.catch(reraise=True)
    def download_aux_models(self, job_info: ImageGenerateJobPopResponse) -> float | None:
        """Download auxiliary models required for the job.

        Args:
            job_info (ImageGenerateJobPopResponse): The job to download auxiliary models for.

        Returns:
            float | None: The time elapsed during downloading, or None if no models were downloaded.
        """
        with self._aux_model_lock:
            time_start = time.time()

            lora_manager = self._shared_model_manager.manager.lora
            if lora_manager is None:
                raise RuntimeError("Failed to load LORA model manager")

            performed_a_download = False

            loras = job_info.payload.loras

            try:
                lora_manager.load_model_database()
                lora_manager.reset_adhoc_loras()
            except Exception as e:
                logger.error(f"Failed to reset adhoc loras: {type(e).__name__} {e}")

            time_to_wait_for_downloads = 0
            for lora_entry in loras:
                if not lora_manager.is_model_available(lora_entry.name):
                    if not performed_a_download:
                        self.send_aux_model_message(
                            process_state=HordeProcessState.DOWNLOADING_AUX_MODEL,
                            info="Downloading auxiliary models",
                            time_elapsed=0.0,
                            job_info=job_info,
                        )
                        performed_a_download = True
                    lora_manager.fetch_adhoc_lora(lora_entry.name, timeout=None, is_version=lora_entry.is_version)
                    if time_to_wait_for_downloads == 0:
                        time_to_wait_for_downloads = BASE_LORA_DOWNLOAD_TIMEOUT
                    else:
                        time_to_wait_for_downloads += EXTRA_LORA_DOWNLOAD_TIMEOUT
                try:
                    lora_manager.wait_for_downloads(time_to_wait_for_downloads)
                except Exception as e:
                    logger.error(f"Failed to wait for downloads: {type(e).__name__} {e}")

            time_elapsed = round(time.time() - time_start, 2)

            lora_manager.save_cached_reference_to_disk()

            if performed_a_download:
                logger.info(f"Downloaded auxiliary models in {time_elapsed} seconds")
                return time_elapsed

            return None

    @logger.catch(reraise=True)
    def preload_model(
        self,
        horde_model_name: str,
        will_load_loras: bool,
        seamless_tiling_enabled: bool,
        job_info: ImageGenerateJobPopResponse,
    ) -> None:
        """Preload a model into RAM.

        Args:
            horde_model_name (str): The name of the model to preload.
            will_load_loras (bool): Whether or not the model will be loaded into VRAM.
            seamless_tiling_enabled (bool): Whether or not seamless tiling is enabled.
            job_info (ImageGenerateJobPopResponse): The job to preload the model for.
        """
        if self._active_model_name == horde_model_name:
            return

        if self._is_busy:
            logger.warning("Cannot preload model while busy")

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

        time_start = time.time()

        with self.disk_lock:
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
            time_elapsed=time.time() - time_start,
        )

        download_time = self.download_aux_models(job_info)

        if download_time is not None:
            self.send_aux_model_message(
                process_state=HordeProcessState.DOWNLOAD_AUX_COMPLETE,
                info="Downloaded auxiliary models",
                time_elapsed=download_time,
                job_info=job_info,
            )

        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info=f"Preloaded model {horde_model_name}",
        )

    _is_busy: bool = False

    _start_inference_time: float = 0.0

    def progress_callback(
        self,
        progress_report: ProgressReport,
    ) -> None:
        """Handle progress updates from the HordeLib instance.

        Args:
            progress_report (ProgressReport): The progress report from the HordeLib instance.
        """
        from hordelib.horde import ProgressState

        if progress_report.hordelib_progress_state == ProgressState.post_processing:
            self.send_process_state_change_message(
                process_state=HordeProcessState.INFERENCE_POST_PROCESSING,
                info="Post Processing",
                time_elapsed=time.time() - self._start_inference_time,
            )
            try:
                self._inference_semaphore.release()
                logger.debug("Released inference semaphore")
            except Exception as e:
                logger.error(f"Failed to release inference semaphore: {type(e).__name__} {e}")

        self.send_heartbeat_message(heartbeat_type=HordeHeartbeatType.INFERENCE_STEP)

    def start_inference(self, job_info: ImageGenerateJobPopResponse) -> list[ResultingImageReturn] | None:
        """Start an inference job in the HordeLib instance.

        Args:
            job_info (ImageGenerateJobPopResponse): The job to start inference on.

        Returns:
            list[Image] | None: The generated images, or None if inference failed.
        """
        logger.info("Checking if too many inference jobs are already running...")
        self._inference_semaphore.acquire()
        logger.info("Acquired inference semaphore.")
        self._is_busy = True
        try:
            logger.info(f"Starting inference for job(s) {job_info.ids}")
            logger.debug(
                f"has source_image: {job_info.source_image is not None} "
                f"has source_mask: {job_info.source_mask is not None}",
            )
            logger.debug(f"{job_info.payload.model_dump(exclude={'prompt'})}")

            with logger.catch(reraise=True):
                self._start_inference_time = time.time()
                results = self._horde.basic_inference(job_info, progress_callback=self.progress_callback)
        except Exception as e:
            logger.critical(f"Inference failed: {type(e).__name__} {e}")
            return None
        finally:
            self._is_busy = False
            with contextlib.suppress(Exception):
                self._inference_semaphore.release()
        return results

    @logger.catch(reraise=True)
    def unload_models_from_vram(self) -> None:
        """Unload all models from VRAM."""
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

    @logger.catch(reraise=True)
    def unload_models_from_ram(self) -> None:
        """Unload all models from RAM."""
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

    @logger.catch(reraise=True)
    def cleanup_for_exit(self) -> None:
        """Cleanup the process pending a shutdown."""
        self.unload_models_from_ram()
        self.send_process_state_change_message(
            process_state=HordeProcessState.PROCESS_ENDED,
            info="Process ended",
        )

    def send_aux_model_message(
        self,
        job_info: ImageGenerateJobPopResponse,
        time_elapsed: float,
        process_state: HordeProcessState,
        info: str,
    ) -> None:
        """Send an auxiliary model download complete message to the main process.

        Args:
            job_info (ImageGenerateJobPopResponse): The job that was inferred.
            time_elapsed (float): The time elapsed during the last operation.
            process_state (HordeProcessState): The state of the process.
            info (str): Additional information about the message.
        """
        message = HordeAuxModelStateChangeMessage(
            process_state=process_state,
            process_id=self.process_id,
            info=info,
            time_elapsed=time_elapsed,
            sdk_api_job_info=job_info,
        )
        self.process_message_queue.put(message)

    def send_inference_result_message(
        self,
        process_state: HordeProcessState,
        job_info: ImageGenerateJobPopResponse,
        results: list[ResultingImageReturn] | None,
        time_elapsed: float,
    ) -> None:
        """Send an inference result message to the main process.

        Args:
            process_state (HordeProcessState): The state of the process.
            job_info (ImageGenerateJobPopResponse): The job that was inferred.
            images (list[Image] | None): The generated images, or None if inference failed.
            results (list[ResultingImageReturn] | None): The generated images, or None if inference failed.
            time_elapsed (float): The time elapsed during the last operation.
        """
        all_image_results = []

        if results is not None:
            for result in results:
                buffered_image = io.BytesIO()
                if result.image is None:
                    logger.critical("Result or result image is None")
                    continue

                result.image.save(buffered_image, format="PNG")
                image_base64 = base64.b64encode(buffered_image.getvalue()).decode("utf-8")
                all_image_results.append(
                    HordeImageResult(
                        image_base64=image_base64,
                        generation_faults=result.faults,
                    ),
                )

        message = HordeInferenceResultMessage(
            process_id=self.process_id,
            info="Inference result",
            state=GENERATION_STATE.ok if results is not None and len(results) > 0 else GENERATION_STATE.faulted,
            time_elapsed=time_elapsed,
            job_image_results=all_image_results,
            sdk_api_job_info=job_info,
        )
        self.process_message_queue.put(message)

        if self._active_model_name is None:
            logger.critical("No active model name, cannot update model state")
            return

        self.on_horde_model_state_change(
            process_state=process_state,
            horde_model_name=self._active_model_name,
            horde_model_state=ModelLoadState.LOADED_IN_VRAM,
        )

    @override
    @logger.catch(reraise=True)
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        """Receive and handle a control message from the main process.

        Args:
            message (HordeControlMessage): The message to handle.
        """
        logger.debug(f"Received ({type(message)}): {message.control_flag}")

        if isinstance(message, HordePreloadInferenceModelMessage):
            self.preload_model(
                horde_model_name=message.horde_model_name,
                will_load_loras=message.will_load_loras,
                seamless_tiling_enabled=message.seamless_tiling_enabled,
                job_info=message.sdk_api_job_info,
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

                self.on_horde_model_state_change(
                    horde_model_name=message.horde_model_name,
                    process_state=HordeProcessState.INFERENCE_STARTING,
                    horde_model_state=ModelLoadState.IN_USE,
                )

                time_start = time.time()

                results = self.start_inference(message.sdk_api_job_info)

                if results is None or len(results) == 0:
                    self.send_memory_report_message(include_vram=True)
                    self.send_inference_result_message(
                        process_state=HordeProcessState.INFERENCE_FAILED,
                        job_info=message.sdk_api_job_info,
                        results=None,
                        time_elapsed=time.time() - time_start,
                    )

                    active_model_name = self._active_model_name
                    logger.debug("Unloading models from RAM")
                    self.unload_models_from_ram()
                    logger.debug("Unloaded models from RAM")
                    self.send_memory_report_message(include_vram=True)

                    if active_model_name is None:
                        logger.critical("No active model name, cannot update model state")

                    else:
                        self.preload_model(
                            active_model_name,
                            will_load_loras=True,
                            seamless_tiling_enabled=False,
                            job_info=message.sdk_api_job_info,
                        )
                        logger.warning("A non-blocking LoRas/TIs preload didn't occur!. This is a bug.")

                    self.send_process_state_change_message(
                        process_state=HordeProcessState.WAITING_FOR_JOB,
                        info="Waiting for job",
                    )
                    return

                process_state = HordeProcessState.INFERENCE_COMPLETE if results else HordeProcessState.INFERENCE_FAILED
                logger.debug(f"Finished inference with process state {process_state}")
                self.send_inference_result_message(
                    process_state=process_state,
                    job_info=message.sdk_api_job_info,
                    results=results,
                    time_elapsed=time.time() - time_start,
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
