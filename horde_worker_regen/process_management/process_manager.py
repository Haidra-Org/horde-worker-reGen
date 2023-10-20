import asyncio
import base64
import multiprocessing
import os
import random
import sys
import time
from asyncio import CancelledError
from asyncio import Lock as Lock_Asyncio
from collections import deque
from collections.abc import Mapping
from io import BytesIO
from multiprocessing.context import BaseContext
from multiprocessing.synchronize import Lock as Lock_MultiProcessing
from multiprocessing.synchronize import Semaphore

import aiohttp
import PIL
import PIL.Image
import psutil
import torch
import yarl
from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientError
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY, STABLE_DIFFUSION_BASELINE_CATEGORY
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import StableDiffusion_ModelReference
from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPIAsyncClientSession, AIHordeAPIAsyncSimpleClient
from horde_sdk.ai_horde_api.apimodels import (
    FindUserRequest,
    FindUserResponse,
    ImageGenerateJobPopRequest,
    ImageGenerateJobPopResponse,
    JobSubmitResponse,
)
from loguru import logger
from pydantic import BaseModel, ConfigDict, RootModel, ValidationError

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.bridge_data.load_config import BridgeDataLoader
from horde_worker_regen.consts import BRIDGE_CONFIG_FILENAME
from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeControlModelMessage,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
    HordeModelStateChangeMessage,
    HordePreloadInferenceModelMessage,
    HordeProcessMemoryMessage,
    HordeProcessMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
    HordeSafetyControlMessage,
    HordeSafetyResultMessage,
    ModelInfo,
    ModelLoadState,
)
from horde_worker_regen.process_management.worker_entry_points import start_inference_process, start_safety_process

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore


class HordeProcessInfo:
    """Contains information about a horde child process."""

    mp_process: multiprocessing.Process
    """The multiprocessing.Process object for this process."""
    pipe_connection: Connection
    """The connection through which messages can be sent to this process."""
    process_id: int
    """The ID of this process. This is not an OS process ID."""
    process_type: HordeProcessType
    """The type of this process."""
    last_process_state: HordeProcessState
    """The last known state of this process."""
    loaded_horde_model_name: str | None = None
    """The name of the horde model that is (supposedly) currently loaded in this process."""

    cuda_visible_device_number: int | None = None
    """The CUDA_VISIBLE_DEVICE number that this process is using."""

    ram_usage_bytes: int = 0
    """The amount of RAM used by this process."""
    vram_usage_bytes: int = 0
    """The amount of VRAM used by this process."""
    total_vram_bytes: int = 0
    """The total amount of VRAM available to this process."""

    # TODO: VRAM usage

    def __init__(
        self,
        mp_process: multiprocessing.Process,
        pipe_connection: Connection,
        process_id: int,
        process_type: HordeProcessType,
        last_process_state: HordeProcessState,
        *,
        cuda_visible_device_number: int | None = None,
    ) -> None:
        """Initializes a new HordeProcessInfo object.

        Args:
            mp_process (multiprocessing.Process): The multiprocessing.Process object for this process.
            pipe_connection (Connection): The connection through which messages can be sent to this process.
            process_id (int): The ID of this process. This is not an OS process ID.
            process_type (HordeProcessType): The type of this process.
            last_process_state (HordeProcessState): The last known state of this process.
        """
        self.mp_process = mp_process
        self.pipe_connection = pipe_connection
        self.process_id = process_id
        self.process_type = process_type
        self.last_process_state = last_process_state
        self.cuda_visible_device_number = cuda_visible_device_number

    def is_process_busy(self) -> bool:
        """Return true if the process is actively engaged in a task.
        This does not include the process starting up or shutting down."""

        return (
            self.last_process_state == HordeProcessState.INFERENCE_STARTING
            or self.last_process_state == HordeProcessState.ALCHEMY_STARTING
            or self.last_process_state == HordeProcessState.DOWNLOADING_MODEL
            or self.last_process_state == HordeProcessState.PRELOADING_MODEL
            or self.last_process_state == HordeProcessState.JOB_RECEIVED
            or self.last_process_state == HordeProcessState.EVALUATING_SAFETY
            or self.last_process_state == HordeProcessState.PROCESS_STARTING
        )

    def __repr__(self) -> str:
        return str(
            f"HordeProcessInfo(process_id={self.process_id}, last_process_state={self.last_process_state}, "
            f"loaded_horde_model_name={self.loaded_horde_model_name})",
        )

    def can_accept_job(self) -> bool:
        """Return true if the process can accept a job."""
        return (
            self.last_process_state == HordeProcessState.WAITING_FOR_JOB
            or self.last_process_state == HordeProcessState.INFERENCE_COMPLETE
            or self.last_process_state == HordeProcessState.ALCHEMY_COMPLETE
        )


class HordeModelMap(RootModel[dict[str, ModelInfo]]):
    def update_entry(
        self,
        horde_model_name: str,
        *,
        load_state: ModelLoadState | None = None,
        process_id: int | None = None,
    ) -> None:
        """Update the entry for the given model name. If the model does not exist, it will be created.

        Args:
            horde_model_name (str): The (horde) name of the model to update.
            load_state (ModelLoadState | None, optional): The load state of the model. Defaults to None.
            process_id (int | None, optional): The process ID of the process that has this model loaded. \
                Defaults to None.

        Raises:
            ValueError: If the process_id is None and the model does not exist.
            ValueError: If the load_state is None and the model does not exist.
        """
        if horde_model_name not in self.root:
            if process_id is None:
                raise ValueError("process_id must be provided when adding a new model to the map")
            if load_state is None:
                raise ValueError("model_load_state must be provided when adding a new model to the map")

            self.root[horde_model_name] = ModelInfo(
                horde_model_name=horde_model_name,
                horde_model_load_state=load_state,
                process_id=process_id,
            )

        if load_state is not None:
            self.root[horde_model_name].horde_model_load_state = load_state

        if process_id is not None:
            self.root[horde_model_name].process_id = process_id

    def is_model_loaded(self, horde_model_name: str) -> bool:
        """Return true if the given model is loaded in any process."""
        if horde_model_name not in self.root:
            return False
        return self.root[horde_model_name].horde_model_load_state.is_loaded()

    def is_model_loading(self, horde_model_name: str) -> bool:
        """Return true if the given model is currently being loaded in any process."""
        if horde_model_name not in self.root:
            return False
        return self.root[horde_model_name].horde_model_load_state == ModelLoadState.LOADING


class ProcessMap(dict[int, HordeProcessInfo]):
    """A mapping of process IDs to HordeProcessInfo objects. Contains some helper methods."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def update_entry(
        self,
        process_id: int,
        *,
        last_process_state: HordeProcessState | None = None,
        loaded_horde_model_name: str | None = None,
        ram_usage_bytes: int | None = None,
        vram_usage_bytes: int | None = None,
        total_vram_bytes: int | None = None,
    ) -> None:
        """Update the entry for the given process ID. If the process does not exist, it will be created.

        Args:
            process_id (int): The ID of the process to update.
            last_process_state (HordeProcessState | None, optional): The last process state of the process. \
                Defaults to None.
            loaded_horde_model_name (str | None, optional): The name of the horde model that is (supposedly) \
                currently loaded in this process. Defaults to None.
            ram_usage_bytes (int | None, optional): The amount of RAM used by this process. Defaults to None.
            vram_usage_bytes (int | None, optional): The amount of VRAM used by this process. Defaults to None.
            total_vram_bytes (int | None, optional): The total amount of VRAM available to this process. \
                Defaults to None.
        """
        if last_process_state is not None:
            self[process_id].last_process_state = last_process_state

        if loaded_horde_model_name is not None:
            self[process_id].loaded_horde_model_name = loaded_horde_model_name

        if ram_usage_bytes is not None:
            self[process_id].ram_usage_bytes = ram_usage_bytes

        if vram_usage_bytes is not None:
            self[process_id].vram_usage_bytes = vram_usage_bytes

        if total_vram_bytes is not None:
            self[process_id].total_vram_bytes = total_vram_bytes

    def num_inference_processes(self) -> int:
        """Return the number of inference processes."""
        count = 0
        for p in self.values():
            if p.process_type == HordeProcessType.INFERENCE:
                count += 1
        return count

    def num_available_inference_processes(self) -> int:
        """Return the number of inference processes that are available to accept jobs."""
        count = 0
        for p in self.values():
            if p.process_type == HordeProcessType.INFERENCE and not p.is_process_busy():
                count += 1
        return count

    def get_first_available_inference_process(self) -> HordeProcessInfo | None:
        """Return the first available inference process, or None if there are none available."""

        multi_gpu = False
        if any(p.cuda_visible_device_number for p in self.values()):
            multi_gpu = True

        found_busy_gpus: list[int] = []
        found_gpu_and_process_info: list[tuple[int, HordeProcessInfo]] = []

        for process_info in self.values():
            if process_info.process_type == HordeProcessType.INFERENCE:
                if process_info.can_accept_job():
                    if process_info.last_process_state == HordeProcessState.PRELOADED_MODEL:
                        if multi_gpu:
                            found_busy_gpus.append(process_info.cuda_visible_device_number or 0)
                        continue

                    if not multi_gpu:
                        return process_info
                    found_gpu_and_process_info.append((process_info.cuda_visible_device_number or 0, process_info))
                else:
                    found_busy_gpus.append(process_info.cuda_visible_device_number or 0)

        if multi_gpu:
            for found_gpu, found_process_info in found_gpu_and_process_info:
                if found_gpu not in found_busy_gpus:
                    return found_process_info

        return None

    def _get_first_inference_process_to_kill(self) -> HordeProcessInfo | None:
        """Return the first inference process eligible to be killed, or None if there are none.
        Used during shutdown."""
        for p in self.values():
            if p.process_type != HordeProcessType.INFERENCE:
                continue

            if (
                p.last_process_state == HordeProcessState.WAITING_FOR_JOB
                or p.last_process_state == HordeProcessState.PROCESS_STARTING
                or p.last_process_state == HordeProcessState.DOWNLOADING_MODEL
                or p.last_process_state == HordeProcessState.INFERENCE_COMPLETE
            ):
                return p

            if p.is_process_busy():
                continue

            if ():
                return p
        return None

    def get_safety_process(self) -> HordeProcessInfo | None:
        """Return the safety process."""
        for p in self.values():
            if p.process_type == HordeProcessType.SAFETY:
                return p
        return None

    def num_safety_processes(self) -> int:
        """Return the number of safety processes."""
        count = 0
        for p in self.values():
            if p.process_type == HordeProcessType.SAFETY:
                count += 1
        return count

    def get_first_available_safety_process(self) -> HordeProcessInfo | None:
        """Return the first available safety process, or None if there are none available."""
        for p in self.values():
            if p.process_type == HordeProcessType.SAFETY and p.last_process_state == HordeProcessState.WAITING_FOR_JOB:
                return p
        return None

    def get_process_by_horde_model_name(self, horde_model_name: str) -> HordeProcessInfo | None:
        """Return the process that has the given horde model loaded, or None if there is none."""
        for p in self.values():
            if p.loaded_horde_model_name == horde_model_name:
                return p
        return None

    def num_busy_processes(self) -> int:
        """Return the number of processes that are actively engaged in a task. This does not include processes which
        are starting up or shutting down, or in a faulted state."""
        count = 0
        for p in self.values():
            if p.is_process_busy():
                count += 1
        return count

    def __repr__(self) -> str:
        base_string = "Processes: "
        for process_id, process_info in self.items():
            if process_info.process_type == HordeProcessType.INFERENCE:
                base_string += f"{process_id}: ({process_info.loaded_horde_model_name}) "
            else:
                base_string += f"{process_id}: ({process_info.process_type.name}) "
            base_string += f"{process_info.last_process_state.name}; "

        return base_string


class TorchDeviceInfo(BaseModel):
    device_name: str
    device_index: int
    total_memory: int


class TorchDeviceMap(RootModel[dict[int, TorchDeviceInfo]]):  # TODO
    pass


class CompletedJobInfo(BaseModel):
    """Contains information about a job that has been generated. It is used to track the state of the job
    as it goes through the safety process and then when it is returned to the requesting user."""

    job_info: ImageGenerateJobPopResponse
    """The API response which has all of the information about the job."""
    job_result_images_base64: list[str] | None = None
    """A list of base64 encoded images that are the result of the job."""
    state: GENERATION_STATE
    """The state of the job to send to the API."""
    censored: bool | None = None
    """Whether or not the job was censored. This is set by the safety process."""
    time_to_generate: float
    """The time it took to generate the job. This is set by the inference process."""

    @property
    def is_job_checked_for_safety(self) -> bool:
        return self.censored is not None


class HordeWorkerProcessManager:
    """Manages and controls processes to act as a horde worker."""

    bridge_data: reGenBridgeData
    """The bridge data for this worker."""

    horde_model_reference_manager: ModelReferenceManager
    """The model reference manager for this worker."""

    max_inference_processes: int
    """The maximum number of inference processes that can be active. This is not the number of jobs that
    can run at once. Use `max_concurrent_inference_processes` to control that behavior."""

    _max_concurrent_inference_processes: int

    @property
    def max_concurrent_inference_processes(self) -> int:
        """The maximum number of inference processes that can run jobs concurrently."""
        return self._max_concurrent_inference_processes

    max_safety_processes: int
    """The maximum number of safety processes that can run at once."""
    max_download_processes: int
    """The maximum number of download processes that can run at once."""

    total_ram_bytes: int
    """The total amount of RAM on the system."""
    target_ram_overhead_bytes: int
    """The target amount of RAM to keep free."""

    target_vram_overhead_bytes_map: Mapping[int, int] | None = None

    @property
    def max_queue_size(self) -> int:
        return self.bridge_data.queue_size

    @property
    def current_queue_size(self) -> int:
        return len(self.job_deque)

    @property
    def target_ram_bytes_used(self) -> int:
        return self.total_ram_bytes - self.target_ram_overhead_bytes

    def get_process_total_ram_usage(self) -> int:
        total = 0
        for process_info in self._process_map.values():
            total += process_info.ram_usage_bytes
        return total

    jobs_in_progress: list[ImageGenerateJobPopResponse]
    """A list of jobs that are currently in progress."""

    jobs_pending_safety_check: list[CompletedJobInfo]
    _jobs_safety_check_lock: Lock_Asyncio

    jobs_being_safety_checked: list[CompletedJobInfo]

    completed_jobs: list[CompletedJobInfo]
    """A list of 3 tuples containing the job, the state, and whether or not the job was censored."""

    _completed_jobs_lock: Lock_Asyncio

    kudos_generated_this_session: float = 0
    session_start_time: float = 0

    _aiohttp_session: aiohttp.ClientSession

    stable_diffusion_reference: StableDiffusion_ModelReference | None
    horde_client: AIHordeAPIAsyncSimpleClient
    horde_client_session: AIHordeAPIAsyncClientSession

    user_info: FindUserResponse | None = None
    """The user info for the user that this worker is logged in as."""
    _last_user_info_fetch_time: float = 0
    """The time at which the user info was last fetched."""
    _user_info_fetch_interval: float = 10
    """The number of seconds between each fetch of the user info."""

    _process_map: ProcessMap
    """A mapping (dict) of process IDs to HordeProcessInfo objects. Contains some helper methods."""
    _horde_model_map: HordeModelMap
    """A mapping (dict) of horde model names to ModelInfo objects. Contains some helper methods."""
    _device_map: TorchDeviceMap
    """A mapping (dict) of device IDs to TorchDeviceInfo objects. Contains some helper methods."""

    _loop_interval: float = 0.20
    """The number of seconds to wait between each loop of the main process (inter process management) loop."""
    _api_call_loop_interval = 0.05
    """The number of seconds to wait between each loop of the main API call loop."""

    _api_get_user_info_interval = 15
    """The number of seconds to wait between each fetch of the user info."""

    _last_get_user_info_time: float = 0
    """The time at which the user info was last fetched."""

    @property
    def num_total_processes(self) -> int:
        return self.max_inference_processes + self.max_safety_processes + self.max_download_processes

    _process_message_queue: ProcessQueue
    """A queue of messages sent from child processes."""

    job_deque: deque[ImageGenerateJobPopResponse]
    """A deque of jobs that are waiting to be processed."""
    _job_deque_lock: Lock_Asyncio

    job_pop_timestamps: dict[str, float]
    _job_pop_timestamps_lock: Lock_Asyncio

    _inference_semaphore: Semaphore
    """A semaphore that limits the number of inference processes that can run at once."""
    _disk_lock: Lock_MultiProcessing

    _shutting_down = False

    def __init__(
        self,
        *,
        ctx: BaseContext,
        bridge_data: reGenBridgeData,
        horde_model_reference_manager: ModelReferenceManager,
        target_ram_overhead_bytes: int = 9 * 1024 * 1024 * 1024,
        target_vram_overhead_bytes_map: Mapping[int, int] | None = None,  # FIXME
        max_safety_processes: int = 1,
        max_download_processes: int = 1,
    ) -> None:
        self.session_start_time = time.time()

        self.bridge_data = bridge_data
        self.horde_model_reference_manager = horde_model_reference_manager

        self._process_map = ProcessMap({})
        self._horde_model_map = HordeModelMap(root={})

        self.max_safety_processes = max_safety_processes
        self.max_download_processes = max_download_processes

        self.max_inference_processes = self.bridge_data.queue_size + self.bridge_data.max_threads

        # If there is only one model to load and only one inference process, then we can only run one job at a time
        # and there is no point in having more than one inference process
        if len(self.bridge_data.image_models_to_load) == 1 and self.max_concurrent_inference_processes == 1:
            self.max_inference_processes = 1

        self._disk_lock = Lock_MultiProcessing(ctx=ctx)

        self.completed_jobs = []
        self._completed_jobs_lock = Lock_Asyncio()

        self.jobs_pending_safety_check = []
        self.jobs_being_safety_checked = []

        self._jobs_safety_check_lock = Lock_Asyncio()

        # region Read and handle RAM/VRAM overhead settings
        self.target_vram_overhead_bytes_map = target_vram_overhead_bytes_map  # TODO

        self.total_ram_bytes = psutil.virtual_memory().total

        self.target_ram_overhead_bytes = target_ram_overhead_bytes
        self.target_ram_overhead_bytes = min(int(self.total_ram_bytes / 2), 9)

        if self.target_ram_overhead_bytes > self.total_ram_bytes:
            raise ValueError(
                f"target_ram_overhead_bytes ({self.target_ram_overhead_bytes}) is greater than "
                "total_ram_bytes ({self.total_ram_bytes})",
            )

        self._status_message_frequency = bridge_data.stats_output_frequency

        logger.debug(f"Total RAM: {self.total_ram_bytes / 1024 / 1024 / 1024} GB")
        logger.debug(f"Target RAM overhead: {self.target_ram_overhead_bytes / 1024 / 1024 / 1024} GB")

        # endregion

        # Get the total memory of each GPU
        self._device_map = TorchDeviceMap(root={})
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            self._device_map.root[i] = TorchDeviceInfo(
                device_name=device.name,
                device_index=i,
                total_memory=device.total_memory,
            )

        # region Read and handle the thread count and auto_dual_gpu settings
        if self.bridge_data.auto_dual_gpu:
            if len(self._device_map.root) < 2:
                logger.exception("auto_dual_gpu is enabled, but there are not enough GPUs to use it")
                raise ValueError("auto_dual_gpu is enabled, but there are not enough GPUs to use it")

            if self.bridge_data.max_threads % 2 != 0:
                logger.warning(
                    "auto_dual_gpu is enabled, but max_threads is not a multiple of 2. "
                    "This means one GPU will get more work than the other.",
                )

        self._max_concurrent_inference_processes = bridge_data.max_threads
        self._inference_semaphore = Semaphore(self._max_concurrent_inference_processes, ctx=ctx)
        # endregion

        self.jobs_in_progress = []

        self.job_deque = deque()
        self._job_deque_lock = Lock_Asyncio()

        self.job_pop_timestamps = {}
        self._job_pop_timestamps_lock = Lock_Asyncio()

        self._process_message_queue = multiprocessing.Queue()

        self.stable_diffusion_reference = None

        # Get the model reference for stable diffusion from the model reference manager
        # (we can't proceed without this)
        while self.stable_diffusion_reference is None:
            try:
                horde_model_reference_manager = ModelReferenceManager(
                    download_and_convert_legacy_dbs=False,
                    override_existing=False,
                )
                all_refs = horde_model_reference_manager.get_all_model_references(False)
                _sd_ref = all_refs[MODEL_REFERENCE_CATEGORY.stable_diffusion]

                if not isinstance(_sd_ref, StableDiffusion_ModelReference):
                    raise ValueError("Expected StableDiffusion_ModelReference")

                self.stable_diffusion_reference = _sd_ref
            except Exception as e:
                logger.error(e)
                time.sleep(5)

    def is_time_for_shutdown(self) -> bool:
        if all(
            inference_process.last_process_state == HordeProcessState.PROCESS_ENDING
            or inference_process.last_process_state == HordeProcessState.PROCESS_ENDED
            for inference_process in self._process_map.values()
        ):
            return True

        # If any job hasn't been submitted to the API yet, then we can't shut down
        if len(self.completed_jobs) > 0:
            return False

        # If there are any jobs in progress, then we can't shut down
        if len(self.jobs_being_safety_checked) > 0 or len(self.jobs_pending_safety_check) > 0:
            return False
        if len(self.jobs_in_progress) > 0:
            return False
        if len(self.job_deque) > 0:
            return False

        any_process_alive = False

        for process_info in self._process_map.values():
            # The safety process gets shut down last and is part of cleanup
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue

            if (
                process_info.last_process_state != HordeProcessState.PROCESS_ENDED
                and process_info.last_process_state != HordeProcessState.PROCESS_ENDING
            ):
                any_process_alive = True
                continue

        # If there are any inference processes still alive, then we can't shut down
        return not any_process_alive

    def is_free_inference_process_available(self) -> bool:
        """Return true if there is an inference process available which can accept a job."""
        return self._process_map.num_available_inference_processes() > 0

    def get_expected_ram_usage(self, horde_model_name: str) -> int:  # TODO: Use or rework this
        if self.stable_diffusion_reference is None:
            raise ValueError("stable_diffusion_reference is None")

        horde_model_record = self.stable_diffusion_reference.root[horde_model_name]

        if horde_model_record.baseline == STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_1:
            return int(3 * 1024 * 1024 * 1024)
        if horde_model_record.baseline == STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_2_512:
            return 4 * 1024 * 1024 * 1024
        if horde_model_record.baseline == STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_2_768:
            return 5 * 1024 * 1024 * 1024
        if horde_model_record.baseline == STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_xl:
            return int(5.75 * 1024 * 1024 * 1024)

        raise ValueError(f"Model {horde_model_name} has an unknown baseline {horde_model_record.baseline}")

    def start_safety_processes(self) -> None:
        """Start all the safety processes configured to be used. This can be used after a configuration
        change to get just the newly configured processes running."""

        num_processes_to_start = self.max_safety_processes - self._process_map.num_safety_processes()

        # If the number of processes to start is less than 0, log a critical error and raise a ValueError
        if num_processes_to_start < 0:
            logger.critical(
                f"There are already {self._process_map.num_safety_processes()} safety processes running, but "
                f"max_safety_processes is set to {self.max_safety_processes}",
            )
            raise ValueError("num_processes_to_start cannot be less than 0")

        # Start the required number of processes

        for _ in range(num_processes_to_start):
            # Create a two-way communication pipe for the parent and child processes
            pid = len(self._process_map)
            pipe_connection, child_pipe_connection = multiprocessing.Pipe(duplex=True)

            cpu_only = not self.bridge_data.safety_on_gpu

            # Create a new process that will run the start_safety_process function
            process = multiprocessing.Process(
                target=start_safety_process,
                args=(
                    pid,
                    self._process_message_queue,
                    child_pipe_connection,
                    self._disk_lock,
                    cpu_only,
                ),
            )

            process.start()

            # Add the process to the process map
            self._process_map[pid] = HordeProcessInfo(
                mp_process=process,
                pipe_connection=pipe_connection,
                process_id=pid,
                process_type=HordeProcessType.SAFETY,
                last_process_state=HordeProcessState.PROCESS_STARTING,
            )

            logger.info(f"Started safety process (id: {pid})")

    def start_inference_processes(self) -> None:
        """Start all the inference processes configured to be used. This can be used after a configuration
        change to get just the newly configured processes running."""

        num_processes_to_start = self.max_inference_processes - self._process_map.num_inference_processes()

        # If the number of processes to start is less than 0, log a critical error and raise a ValueError
        if num_processes_to_start < 0:
            logger.critical(
                f"There are already {self._process_map.num_inference_processes()} inference processes running, but "
                f"max_inference_processes is set to {self.max_inference_processes}",
            )
            raise ValueError("num_processes_to_start cannot be less than 0")

        if num_processes_to_start == 0:
            return

        if self._process_map.num_inference_processes() > 0 and self.bridge_data.auto_dual_gpu:
            raise NotImplementedError(
                "auto_dual_gpu is not supported when starting inference processes after startup",
            )

        # Start the required number of processes
        for i in range(num_processes_to_start):
            # Create a two-way communication pipe for the parent and child processes
            pid = len(self._process_map)
            pipe_connection, child_pipe_connection = multiprocessing.Pipe(duplex=True)

            kwargs = {}

            if self.bridge_data.auto_dual_gpu:
                kwargs["CUDA_VISIBLE_DEVICES"] = str(i % 2)

            # Create a new process that will run the start_inference_process function
            process = multiprocessing.Process(
                target=start_inference_process,
                args=(
                    pid,
                    self._process_message_queue,
                    child_pipe_connection,
                    self._inference_semaphore,
                    self._disk_lock,
                ),
                kwargs=kwargs,
            )

            process.start()

            # Add the process to the process map
            self._process_map[pid] = HordeProcessInfo(
                mp_process=process,
                pipe_connection=pipe_connection,
                process_id=pid,
                process_type=HordeProcessType.INFERENCE,
                last_process_state=HordeProcessState.PROCESS_STARTING,
                cuda_visible_device_number=i % 2 if self.bridge_data.auto_dual_gpu else None,
            )

            logger.info(f"Started inference process (id: {pid})")

    def end_inference_processes(self) -> None:
        """End any inference processes above the configured limit, or all of them if shutting down."""

        if len(self.job_deque) > 0 and len(self.job_deque) != len(self.jobs_in_progress):
            return

        # Get the process to end
        process_info = self._process_map._get_first_inference_process_to_kill()

        if process_info is None:
            return

        # Send the process a message to end
        process_info.pipe_connection.send(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))

        # Update the process map
        self._process_map.update_entry(process_id=process_info.process_id)

        logger.info(f"Ended inference process {process_info.process_id}")

        # Join the process with a timeout of 1 second
        process_info.mp_process.join(timeout=1)

    total_num_completed_jobs: int = 0

    def end_safety_processes(self) -> None:
        """End any safety processes above the configured limit, or all of them if shutting down."""

        process_info = self._process_map.get_first_available_safety_process()

        if process_info is None:
            return

        # Send the process a message to end
        process_info.pipe_connection.send(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))

        # Update the process map
        self._process_map.update_entry(process_id=process_info.process_id)

        logger.info(f"Ended safety process {process_info.process_id}")

    def receive_and_handle_process_messages(self) -> None:
        """Receive and handle any messages from the child processes. This is the backbone of the \
            inter-process communication system and is the main way that the parent process knows what is going on \
            in the child processes.

        **Note** also that this is a synchronous function and any interaction with objects that are shared between \
            coroutines should be done with care. Critically, this function should be called with locks already \
            acquired on any shared objects.

        See also `._process_map` and `._horde_model_map`, which are updated by this function, and `HordeProcessState` \
            and `ModelLoadState` for the possible states that the processes and models can be in.
        """

        # We want to completely flush the queue, to maximize the chances we get the most up to date information
        while not self._process_message_queue.empty():
            message: HordeProcessMessage = self._process_message_queue.get()

            logger.debug(
                f"Received {type(message).__name__} from process {message.process_id}:",
                # f"{message.model_dump(exclude={'job_result_images_base64', 'replacement_image_base64'})}",
            )

            # These events happening are program-breaking conditions that (hopefully) should never happen in production
            # and are mainly to make debugging easier when making changes to the code, but serve as a guard against
            # truly catastrophic failures
            if not isinstance(message, HordeProcessMessage):
                raise ValueError(f"Received a message that is not a HordeProcessMessage: {message}")
            if message.process_id not in self._process_map:
                raise ValueError(f"Received a message from an unknown process: {message}")

            # If the process is updating us on its memory usage, update the process map for those values only
            # and then continue to the next message
            if isinstance(message, HordeProcessMemoryMessage):
                self._process_map.update_entry(
                    process_id=message.process_id,
                    ram_usage_bytes=message.ram_usage_bytes,
                    vram_usage_bytes=message.vram_usage_bytes,
                    total_vram_bytes=message.vram_total_bytes,
                )
                continue

            # If the process state has changed, update the process map
            if isinstance(message, HordeProcessStateChangeMessage):
                self._process_map.update_entry(
                    process_id=message.process_id,
                    last_process_state=message.process_state,
                )

                logger.debug(f"Process {message.process_id} changed state to {message.process_state}")
                if message.process_state == HordeProcessState.INFERENCE_STARTING:
                    # logger.info(f"Process {message.process_id} is starting inference on model {message.info}")

                    loaded_model_name = self._process_map[message.process_id].loaded_horde_model_name
                    if loaded_model_name is None:
                        raise ValueError(
                            f"Process {message.process_id} has no model loaded, but is starting inference",
                        )
                    self._horde_model_map.update_entry(
                        horde_model_name=loaded_model_name,
                        load_state=ModelLoadState.IN_USE,
                        process_id=message.process_id,
                    )

            # If The model state has changed, update the model map
            if isinstance(message, HordeModelStateChangeMessage):
                self._horde_model_map.update_entry(
                    horde_model_name=message.horde_model_name,
                    load_state=message.horde_model_state,
                    process_id=message.process_id,
                )

                if message.horde_model_state == ModelLoadState.LOADING:
                    logger.debug(f"Process {message.process_id} is loading model {message.horde_model_name}")
                    self._process_map.update_entry(
                        process_id=message.process_id,
                        loaded_horde_model_name=message.horde_model_name,
                    )

                # If the model was just loaded, so update the process map and log a message with the time it took
                if (
                    message.horde_model_state == ModelLoadState.LOADED_IN_VRAM
                    or message.horde_model_state == ModelLoadState.LOADED_IN_RAM
                ):
                    if (
                        message.process_id in self._process_map
                        and message.horde_model_state != self._process_map[message.process_id].loaded_horde_model_name
                    ):
                        loaded_message = f"Process {message.process_id} has model {message.horde_model_name} loaded. "

                        if message.time_elapsed is not None:
                            # round to 2 decimal places
                            loaded_message += f"Loading took {message.time_elapsed:.2f} seconds"

                        logger.info(loaded_message)

                    self._process_map.update_entry(
                        process_id=message.process_id,
                        loaded_horde_model_name=message.horde_model_name,
                    )

                elif message.horde_model_state == ModelLoadState.ON_DISK:
                    self._process_map.update_entry(
                        process_id=message.process_id,
                        loaded_horde_model_name=None,
                    )
                    # FIXME this message is wrong for download processes
                    logger.info(f"Process {message.process_id} unloaded model {message.horde_model_name}")

            # If the process is sending us an inference job result:
            # - if its a faulted job, log an error and add it to the list of completed jobs to be sent to the API
            # - if its a completed job, add it to the list of jobs pending safety checks
            if isinstance(message, HordeInferenceResultMessage):
                _num_jobs_in_progress = len(self.jobs_in_progress)

                # Remove the job from the jobs in progress by matching the job ID (.id_)
                self.jobs_in_progress = [job for job in self.jobs_in_progress if job.id_ != message.job_info.id_]

                if len(self.jobs_in_progress) != _num_jobs_in_progress - 1:
                    logger.warning(
                        "Expected to remove 1 job from the jobs in progress, but removed "
                        f"{len(self.jobs_in_progress) - _num_jobs_in_progress} jobs",
                    )
                    logger.debug(f"Jobs in progress: {self.jobs_in_progress}")

                for job in self.job_deque:
                    if job.id_ == message.job_info.id_:
                        self.job_deque.remove(job)
                        break

                self.total_num_completed_jobs += 1
                if message.time_elapsed is not None:
                    logger.info(
                        f"Inference finished for job {message.job_info.id_} on process {message.process_id}. "
                        f"It took {round(message.time_elapsed, 2)} seconds",
                    )
                else:
                    logger.info(f"Inference finished for job {message.job_info.id_}")
                    logger.debug(f"Job didn't include time_elapsed: {message.job_info}")

                if message.state != GENERATION_STATE.faulted:
                    self.jobs_pending_safety_check.append(
                        CompletedJobInfo(
                            job_info=message.job_info,
                            job_result_images_base64=message.job_result_images_base64,
                            state=message.state,
                            time_to_generate=message.time_elapsed if message.time_elapsed is not None else 0,
                        ),
                    )
                else:
                    logger.error(
                        f"Job {message.job_info.id_} faulted on process {message.process_id}: {message.info}",
                    )

                    self.completed_jobs.append(
                        CompletedJobInfo(
                            job_info=message.job_info,
                            job_result_images_base64=None,
                            state=message.state,
                            time_to_generate=message.time_elapsed if message.time_elapsed is not None else 0,
                        ),
                    )

            # If the process is sending us a safety job result:
            # - if an unexpected error occurred, log an error a
            # - if the job was censored, replace the images with the replacement images
            # - add the job to the list of completed jobs to be sent to the API
            elif isinstance(message, HordeSafetyResultMessage):
                completed_job_info: CompletedJobInfo | None = None
                for i, job_being_safety_checked in enumerate(self.jobs_being_safety_checked):
                    if job_being_safety_checked.job_info.id_ == message.job_id:
                        completed_job_info = self.jobs_being_safety_checked.pop(i)
                        break

                if completed_job_info is None or completed_job_info.job_result_images_base64 is None:
                    raise ValueError(
                        f"Expected to find a completed job with ID {message.job_id} but none was found",
                    )

                num_images_censored = 0
                num_images_csam = 0

                any_safety_failed = False

                for i in range(len(completed_job_info.job_result_images_base64)):
                    replacement_image = message.safety_evaluations[i].replacement_image_base64

                    if message.safety_evaluations[i].failed:
                        logger.error(
                            f"Job {message.job_id} image #{i} faulted during safety checks. "
                            "Check the safety process logs for more information.",
                        )
                        any_safety_failed = True
                        continue

                    if replacement_image is not None:
                        completed_job_info.job_result_images_base64[i] = replacement_image
                        num_images_censored += 1
                        if message.safety_evaluations[i].is_csam:
                            num_images_csam += 1

                logger.debug(
                    f"Job {message.job_id} had {num_images_censored} images censored and took "
                    f"{message.time_elapsed:.2f} seconds to check safety",
                )

                if any_safety_failed:
                    completed_job_info.state = GENERATION_STATE.faulted
                elif num_images_censored > 0:
                    completed_job_info.censored = True
                    if num_images_csam > 0:
                        completed_job_info.state = GENERATION_STATE.csam
                    else:
                        completed_job_info.state = GENERATION_STATE.censored
                else:
                    completed_job_info.censored = False

                self.completed_jobs.append(completed_job_info)

    def preload_models(self) -> bool:
        """Preload models that are likely to be used soon.

        Returns:
            True if a model was preloaded, False otherwise.
        """

        # Starting from the left of the deque, preload models that are not yet loaded up to the
        # number of inference processes that are available
        for job in self.job_deque:
            model_is_loaded = False

            if job.model is None:
                raise ValueError(f"job.model is None ({job})")

            for process in self._process_map.values():
                if process.loaded_horde_model_name == job.model:
                    model_is_loaded = True
                    break

            for model in self._horde_model_map.root.values():
                if model.horde_model_name == job.model and (
                    model.horde_model_load_state.is_loaded() or model.horde_model_load_state == ModelLoadState.LOADING
                ):
                    model_is_loaded = True
                    break

            if model_is_loaded:
                continue

            available_process = self._process_map.get_first_available_inference_process()

            if available_process is None:
                return False

            logger.debug(f"Preloading model {job.model} on process {available_process.process_id}")
            logger.debug(f"Available inference processes: {self._process_map}")
            logger.debug(f"Horde model map: {self._horde_model_map}")

            will_load_loras = job.payload.loras is not None and len(job.payload.loras) > 0
            seamless_tiling_enabled = job.payload.tiling is not None and job.payload.tiling

            available_process.pipe_connection.send(
                HordePreloadInferenceModelMessage(
                    control_flag=HordeControlFlag.PRELOAD_MODEL,
                    horde_model_name=job.model,
                    will_load_loras=will_load_loras,
                    seamless_tiling_enabled=seamless_tiling_enabled,
                ),
            )

            self._horde_model_map.update_entry(
                horde_model_name=job.model,
                load_state=ModelLoadState.LOADING,
                process_id=available_process.process_id,
            )

            self._process_map.update_entry(
                process_id=available_process.process_id,
                loaded_horde_model_name=job.model,
            )

            return True

        return False

    def start_inference(self) -> None:
        """Start inference for the next job in the deque, if possible."""

        if len(self.jobs_in_progress) >= self.max_concurrent_inference_processes:
            return

        # Get the first job in the deque that is not already in progress
        next_job: ImageGenerateJobPopResponse | None = None
        for job in self.job_deque:
            if job in self.jobs_in_progress:
                continue
            next_job = job
            break

        if next_job is None:
            return

        if next_job.model is None:
            raise ValueError(f"next_job.model is None ({next_job})")

        if self._horde_model_map.is_model_loaded(next_job.model):
            process_with_model = self._process_map.get_process_by_horde_model_name(next_job.model)

            if process_with_model is None:
                logger.error(
                    f"Expected to find a process with model {next_job.model} but none was found",
                )
                logger.debug(f"Horde model map: {self._horde_model_map}")
                logger.debug(f"Process map: {self._process_map}")
                return

            if not process_with_model.can_accept_job():
                return

            # Unload all models from vram from any other process that isn't running a job
            for process_info in self._process_map.values():
                if process_info.process_id == process_with_model.process_id:
                    continue

                if process_info.is_process_busy():
                    continue

                if process_info.loaded_horde_model_name is None:
                    continue

                if len(self.job_deque) == len(self.jobs_in_progress) + len(self.jobs_pending_safety_check):
                    logger.debug("Not unloading models from VRAM because there are no jobs to make room for.")
                    continue

                next_n_models = list(self.get_next_n_models(self.max_inference_processes))

                # If the model would be used by another process soon, don't unload it
                if (
                    self.max_concurrent_inference_processes > 1
                    and process_info.loaded_horde_model_name
                    in next_n_models[: self.max_concurrent_inference_processes - 1]
                ):
                    continue

                process_info.pipe_connection.send(
                    HordeControlModelMessage(
                        control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM,
                        horde_model_name=process_info.loaded_horde_model_name,
                    ),
                )

            logger.info(f"Starting inference for job {next_job.id_} on process {process_with_model.process_id}")
            # region Log job info
            logger.info(f"Model: {next_job.model}")
            if next_job.source_image is not None:
                logger.info(f"Using {next_job.source_processing}")

            extra_info = ""
            if next_job.payload.control_type is not None:
                extra_info += f"Control type: {next_job.payload.control_type}"
            if next_job.payload.loras:
                if extra_info:
                    extra_info += ", "
                extra_info += f"{len(next_job.payload.loras)} LoRAs"
            if next_job.payload.tis:
                if extra_info:
                    extra_info += ", "
                extra_info += f"{len(next_job.payload.tis)} TIs"
            if next_job.payload.post_processing is not None and len(next_job.payload.post_processing) > 0:
                if extra_info:
                    extra_info += ", "
                extra_info += f"Post processing: {next_job.payload.post_processing}"
            if next_job.payload.hires_fix:
                if extra_info:
                    extra_info += ", "
                extra_info += "HiRes fix"

            if extra_info:
                logger.info(extra_info)

            logger.info(
                f"{next_job.payload.width}x{next_job.payload.height} for {next_job.payload.ddim_steps} steps "
                f"with sampler {next_job.payload.sampler_name}",
            )
            # endregion

            self.jobs_in_progress.append(next_job)
            process_with_model.pipe_connection.send(
                HordeInferenceControlMessage(
                    control_flag=HordeControlFlag.START_INFERENCE,
                    horde_model_name=next_job.model,
                    job_info=next_job,
                ),
            )

    def unload_from_ram(self, process_id: int) -> None:
        """Unload models from a process, either from VRAM or both VRAM and system RAM.

        Args:
            process_id: The process to unload models from.
        """

        if process_id not in self._process_map:
            raise ValueError(f"process_id {process_id} is not in the process map")

        process_info = self._process_map[process_id]

        if process_info.loaded_horde_model_name is None:
            raise ValueError(f"process_id {process_id} is not loaded with a model")

        if not self._horde_model_map.is_model_loaded(process_info.loaded_horde_model_name):
            raise ValueError(f"process_id {process_id} is loaded with a model that is not loaded")

        process_info.pipe_connection.send(
            HordeControlModelMessage(
                control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_RAM,
                horde_model_name=process_info.loaded_horde_model_name,
            ),
        )

        self._horde_model_map.update_entry(
            horde_model_name=process_info.loaded_horde_model_name,
            load_state=ModelLoadState.ON_DISK,
            process_id=process_id,
        )

        self._process_map.update_entry(
            process_id=process_id,
            loaded_horde_model_name=None,
        )

    def get_next_n_models(self, n: int) -> set[str]:
        """Get the next n models that will be used in the job deque.

        Args:
            n: The number of models to get.

        Returns:
            A set of the next n models that will be used in the job deque.
        """

        next_n_models: set[str] = set()
        jobs_traversed = 0
        while len(next_n_models) < n:
            if jobs_traversed >= len(self.job_deque):
                break

            model_name = self.job_deque[jobs_traversed].model

            if model_name is None:
                raise ValueError(f"job_deque[{jobs_traversed}].model is None")

            if model_name not in next_n_models:
                next_n_models.add(model_name)

            jobs_traversed += 1

        return next_n_models

    def unload_models(self) -> None:
        """Unload models that are no longer needed and would use above the limit specified."""

        if len(self.job_deque) == 0:
            return

        if len(self.job_deque) == len(self.jobs_in_progress):
            return

        if len(self.job_deque) == len(self.jobs_in_progress) + len(self.jobs_pending_safety_check):
            return

        next_n_models: set[str] = self.get_next_n_models(self.max_inference_processes)

        for process_info in self._process_map.values():
            if process_info.is_process_busy():
                continue

            if process_info.loaded_horde_model_name is None:
                continue

            if self._horde_model_map.is_model_loading(process_info.loaded_horde_model_name):
                continue

            if (
                self._horde_model_map.root[process_info.loaded_horde_model_name].horde_model_load_state
                == ModelLoadState.IN_USE
            ):
                continue

            if process_info.loaded_horde_model_name in next_n_models:
                continue

            if self.get_process_total_ram_usage() > self.target_ram_bytes_used:
                self.unload_from_ram(process_info.process_id)

    def start_evaluate_safety(self) -> None:
        """Start evaluating the safety of the next job pending a safety check, if any."""
        if len(self.jobs_pending_safety_check) == 0:
            return

        safety_process = self._process_map.get_first_available_safety_process()

        if safety_process is None:
            return

        completed_job_info = self.jobs_pending_safety_check[0]

        if completed_job_info.job_result_images_base64 is None:
            raise ValueError("completed_job_info.job_result_images_base64 is None")

        if len(completed_job_info.job_result_images_base64) > 1:
            raise NotImplementedError("Only single image jobs are supported right now")  # TODO

        if completed_job_info.job_info.id_ is None:
            raise ValueError("completed_job_info.job_info.id_ is None")

        if completed_job_info.job_info.model is None:
            raise ValueError("completed_job_info.job_info.model is None")

        if self.stable_diffusion_reference is None:
            raise ValueError("stable_diffusion_reference is None")

        if completed_job_info.job_info.payload.prompt is None:
            raise ValueError("completed_job_info.job_info.payload.prompt is None")

        self.jobs_pending_safety_check.remove(completed_job_info)
        self.jobs_being_safety_checked.append(completed_job_info)

        safety_process.pipe_connection.send(
            HordeSafetyControlMessage(
                control_flag=HordeControlFlag.EVALUATE_SAFETY,
                job_id=completed_job_info.job_info.id_,
                images_base64=completed_job_info.job_result_images_base64,
                prompt=completed_job_info.job_info.payload.prompt,
                censor_nsfw=completed_job_info.job_info.payload.use_nsfw_censor,
                sfw_worker=not self.bridge_data.nsfw,
                horde_model_info=self.stable_diffusion_reference.root[completed_job_info.job_info.model].model_dump(),
                # TODO: update this to use a class instead of a dict?
            ),
        )

    def base64_image_to_stream_buffer(self, image_base64: str) -> BytesIO | None:
        """Convert a base64 image to a BytesIO stream buffer.

        Args:
            image_base64: The base64 image to convert.

        Returns:
            A BytesIO stream buffer containing the image, or None if the conversion failed.
        """
        try:
            image_as_pil = PIL.Image.open(BytesIO(base64.b64decode(image_base64)))
            image_buffer = BytesIO()
            image_as_pil.save(
                image_buffer,
                format="WebP",
                quality=95,  # FIXME # TODO
                method=6,
            )

            return image_buffer
        except Exception as e:
            logger.error(f"Failed to convert base64 image to stream buffer: {e}")
            return None

    _consecutive_failed_job_submits = 0
    """The number of consecutive failed attempts to submit a job result to the API."""
    _max_consecutive_failed_job_submits = 10
    """The maximum number of consecutive failed attempts to submit a job result to the API."""

    async def api_submit_job(self) -> None:
        """Submit a job result to the API, if any are completed (safety checked too) and ready to be submitted."""
        if len(self.completed_jobs) == 0:
            return

        completed_job_info = self.completed_jobs[0]
        job_info = completed_job_info.job_info

        submit_job_request_type = job_info.get_follow_up_default_request_type()

        if completed_job_info.job_result_images_base64 is not None:
            if len(completed_job_info.job_result_images_base64) > 1:
                raise NotImplementedError("Only single image jobs are supported right now")

            if completed_job_info.censored is None:
                raise ValueError("completed_job_info.censored is None")
        if job_info.id_ is None:
            raise ValueError("job_info.id_ is None")

        if job_info.payload.seed is None:
            raise ValueError("job_info.payload.seed is None")

        if job_info.r2_upload is None:  # TODO: r2_upload should be being set somewhere
            raise ValueError("job_info.r2_upload is None")

        # TODO: n_iter support

        try:
            if self._consecutive_failed_job_submits >= self._max_consecutive_failed_job_submits:
                async with self._completed_jobs_lock:
                    self.completed_jobs.remove(completed_job_info)
                    self._consecutive_failed_job_submits = 0
                    self._consecutive_failed_jobs += 1
                    return

            if completed_job_info.job_result_images_base64 is not None:
                image_in_buffer = self.base64_image_to_stream_buffer(completed_job_info.job_result_images_base64[0])

                if image_in_buffer is None:
                    logger.critical(
                        f"There is an invalid image in the job results for {job_info.id_}, "
                        "removing from completed jobs",
                    )
                    async with self._completed_jobs_lock:
                        self.completed_jobs.remove(completed_job_info)
                        self._consecutive_failed_job_submits = 0

                        for follow_up_request in completed_job_info.job_info.get_follow_up_failure_cleanup_request():
                            follow_up_response = self.horde_client_session.submit_request(
                                follow_up_request,
                                JobSubmitResponse,
                            )

                            if isinstance(follow_up_response, RequestErrorResponse):
                                logger.error(f"Failed to submit followup request: {follow_up_response}")
                        return

                async with self._aiohttp_session.put(
                    yarl.URL(job_info.r2_upload, encoded=True),
                    data=image_in_buffer.getvalue(),
                    skip_auto_headers=["content-type"],
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to upload image to R2: {response}")
                        self._consecutive_failed_job_submits += 1
                        return

            if completed_job_info.state == GENERATION_STATE.faulted:
                logger.error(
                    f"Job {job_info.id_} faulted, removing from completed jobs",
                )
                self._consecutive_failed_jobs += 1

            submit_job_request = submit_job_request_type(
                apikey=self.bridge_data.api_key,
                id=job_info.id_,
                seed=int(job_info.payload.seed),
                generation="R2",  # TODO # FIXME
                state=completed_job_info.state,
                censored=bool(completed_job_info.censored),  # TODO: is this cast problematic?
            )

            job_submit_response = await self.horde_client_session.submit_request(submit_job_request, JobSubmitResponse)

            # If the job submit response is an error, log it and increment the number of consecutive failed job submits
            if isinstance(job_submit_response, RequestErrorResponse):
                if (
                    "Processing Job with ID" in job_submit_response.message
                    and "does not exist" in job_submit_response.message
                ):
                    logger.warning(f"Job {job_info.id_} does not exist, removing from completed jobs")
                    async with self._completed_jobs_lock:
                        self.completed_jobs.remove(completed_job_info)
                        self._consecutive_failed_job_submits = 0

                    return

                if "already submitted" in job_submit_response.message:
                    logger.debug(f"Job {job_info.id_} has already been submitted, removing from completed jobs")
                    async with self._completed_jobs_lock:
                        self.completed_jobs.remove(completed_job_info)
                        self._consecutive_failed_job_submits = 0

                    return

                error_string = "Failed to submit job (API Error)"
                error_string += f"{self._consecutive_failed_job_submits}/{self._max_consecutive_failed_job_submits} "
                error_string += f": {job_submit_response}"
                logger.error(error_string)
                self._consecutive_failed_job_submits += 1
                return

            # Get the time the job was popped from the job deque
            async with self._job_pop_timestamps_lock:
                time_popped = self.job_pop_timestamps.pop(str(completed_job_info.job_info.id_))

            time_taken = round(time.time() - time_popped, 2)

            kudos_per_second = job_submit_response.reward / completed_job_info.time_to_generate

            # If the job was not faulted, log the job submission as a success
            if completed_job_info.state != GENERATION_STATE.faulted:
                logger.success(
                    f"Submitted job {job_info.id_} (model: {job_info.model}) for {job_submit_response.reward:.2f} "
                    f"kudos. Job popped {time_taken} seconds ago and took {completed_job_info.time_to_generate:.2f} "
                    f"to generate. ({kudos_per_second:.2f} kudos/second. 0.4 or greater is ideal)",
                )
            # If the job was faulted, log an error
            else:
                logger.error(
                    f"{job_info.id_} faulted, not submitting for kudos. Job popped {time_taken} seconds ago and took "
                    f"{completed_job_info.time_to_generate:.2f} to generate.",
                )

            # If the job took a long time to generate, log a warning (unless speed warnings are suppressed)
            if not self.bridge_data.suppress_speed_warnings:
                if job_submit_response.reward > 0 and (job_submit_response.reward / time_taken) < 0.1:
                    logger.warning(
                        f"This job ({job_info.id_}) may have been in the queue for a long time. ",
                    )

                if job_submit_response.reward > 0 and kudos_per_second < 0.4:
                    logger.warning(
                        f"This job ({job_info.id_}) took longer than is ideal; if this persists consider "
                        "lowering your max_power, using less threads, disabling post processing and/or controlnets.",
                    )

            self.kudos_generated_this_session += job_submit_response.reward

            # Finally, remove the job from the completed jobs list and reset the number of consecutive failed job
            async with self._completed_jobs_lock:
                self.completed_jobs.remove(completed_job_info)
                self._consecutive_failed_job_submits = 0

        except Exception as e:
            logger.error(f"Failed to submit job (Unexpected Error): {e}")
            self._consecutive_failed_job_submits += 1
            return

        await asyncio.sleep(self._api_call_loop_interval)

    # _testing_max_jobs = 10000
    # _testing_jobs_added = 0
    # _testing_job_queue_length = 1

    _default_job_pop_frequency = 1.0
    """The default frequency at which to pop jobs from the API."""
    _error_job_pop_frequency = 5.0
    """The frequency at which to pop jobs from the API when an error occurs."""
    _job_pop_frequency = 1.0
    """The frequency at which to pop jobs from the API. Can be altered if an error occurs."""
    _last_job_pop_time = 0.0
    """The time at which the last job was popped from the API."""

    _max_pending_megapixelsteps = 45
    """The maximum number of megapixelsteps that can be pending in the job deque before job pops are paused."""
    _triggered_max_pending_megapixelsteps_time = 0.0
    """The time at which the number of megapixelsteps in the job deque exceeded the limit."""
    _triggered_max_pending_megapixelsteps = False
    """Whether the number of megapixelsteps in the job deque exceeded the limit."""

    _consecutive_failed_jobs = 0

    def get_pending_megapixelsteps(self) -> int:
        """Get the number of megapixelsteps that are pending in the job deque."""
        job_deque_mps = sum(job.payload.width * job.payload.height * job.payload.ddim_steps for job in self.job_deque)

        for _ in self.completed_jobs:
            job_deque_mps += 1_000_000 * 4

        return int((job_deque_mps) / 1_000_000)

    def should_wait_for_pending_megapixelsteps(self) -> bool:
        """Check if the number of megapixelsteps in the job deque is above the limit."""
        # TODO: Option to increase the limit for higher end GPUs

        return (
            self.get_pending_megapixelsteps() > self._max_pending_megapixelsteps
            if not self.bridge_data.auto_dual_gpu
            else self.get_pending_megapixelsteps() > self._max_pending_megapixelsteps * 2.5  # FIXME
        )

    async def _get_source_images(self, job_pop_response: ImageGenerateJobPopResponse) -> ImageGenerateJobPopResponse:
        # TODO: Move this into horde_sdk
        for field_name in ["source_image", "source_mask"]:
            field_value = getattr(job_pop_response, field_name)
            if field_value is not None and "https://" in field_value:
                fail_count = 0
                while True:
                    try:
                        if fail_count >= 10:
                            logger.error(f"Failed to download {field_name} after {fail_count} attempts")
                            break
                        response = await self._aiohttp_session.get(
                            field_value,
                            timeout=aiohttp.ClientTimeout(total=10),
                        )
                        response.raise_for_status()
                        new_response_dict = job_pop_response.model_dump(by_alias=True)

                        content = await response.content.read()

                        new_response_dict[field_name] = base64.b64encode(content).decode("utf-8")
                        job_pop_response = ImageGenerateJobPopResponse(**new_response_dict)
                        logger.debug(f"Downloaded {field_name} for job {job_pop_response.id_}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to download {field_name}: {e}")
                        fail_count += 1
                        time.sleep(0.5)

        return job_pop_response

    async def api_job_pop(self) -> None:
        """If the job deque is not full, add any jobs that are available to the job deque."""
        if self._shutting_down:
            return

        if self._consecutive_failed_jobs >= 3:
            logger.error(
                "Too many consecutive failed jobs, pausing job pops. "
                "Please look into what happened and let the devs know. ",
                "Waiting 300 seconds...",
            )
            time.sleep(300)
            return

        if len(self.job_deque) >= self.bridge_data.queue_size + 1:  # FIXME?
            return

        # We let the first job run through to make sure things are working
        # (if we're doomed to fail with 1 job, we're doomed to fail with 2 jobs)
        if len(self.job_deque) != 0 and self.completed_jobs == 0:
            return

        # if self._testing_jobs_added >= self._testing_max_jobs:
        #   return

        # Don't start jobs if we can't evaluate safety (NSFW/CSAM)
        if self._process_map.get_first_available_safety_process() is None:
            return

        # Don't start jobs if we can't run inference
        if self._process_map.get_first_available_inference_process() is None:
            return

        # If there are long running jobs, don't start any more even if there is space in the deque
        if self.should_wait_for_pending_megapixelsteps():
            if self._triggered_max_pending_megapixelsteps is False:
                self._triggered_max_pending_megapixelsteps = True
                self._triggered_max_pending_megapixelsteps_time = time.time()
                logger.info(
                    f"Paused job pops for pending megapixelsteps to decrease below {self._max_pending_megapixelsteps}",
                )
                return

            # Assuming a megapixelstep takes 0.75 seconds, if 2/3 of the time has passed since the limit was triggered,
            # we can assume that the pending megapixelsteps will be below the limit soon. Otherwise we continue to wait

            if not (time.time() - self._triggered_max_pending_megapixelsteps_time) > (
                (self._max_pending_megapixelsteps * 0.75) * (2 / 3)
            ):
                return

            self._triggered_max_pending_megapixelsteps = False
            logger.info(
                f"Pending megapixelsteps decreased below {self._max_pending_megapixelsteps}, continuing with job pops",
            )

        self._triggered_max_pending_megapixelsteps = False

        # We don't want to pop jobs too frequently, so we wait a bit between each pop
        if time.time() - self._last_job_pop_time < self._job_pop_frequency:
            return

        self._last_job_pop_time = time.time()

        # dummy_jobs = get_n_dummy_jobs(1)
        # async with self._job_deque_lock:
        #     self.job_deque.extend(dummy_jobs)
        # logger.debug(f"Added {len(dummy_jobs)} dummy jobs to the job deque")
        # # log a list of the current model names in the deque
        # logger.debug(f"Current models in job deque: {[job.model for job in self.job_deque]}")

        try:
            job_pop_request = ImageGenerateJobPopRequest(
                apikey=self.bridge_data.api_key,
                name=self.bridge_data.dreamer_worker_name,
                bridge_agent="AI Horde Worker reGen:2:https://github.com/Haidra-Org/horde-worker-reGen/#20231009_3",
                bridge_version=2,
                models=self.bridge_data.image_models_to_load,
                nsfw=self.bridge_data.nsfw,
                threads=self.max_concurrent_inference_processes,
                max_pixels=self.bridge_data.max_power * 8 * 64 * 64,
                require_upfront_kudos=self.bridge_data.require_upfront_kudos,
                allow_img2img=self.bridge_data.allow_img2img,
                allow_painting=self.bridge_data.allow_inpainting,
                allow_unsafe_ipaddr=self.bridge_data.allow_unsafe_ip,
                allow_post_processing=self.bridge_data.allow_post_processing,
                allow_controlnet=self.bridge_data.allow_controlnet,
                allow_lora=self.bridge_data.allow_lora,  # TODO loras broken
            )

            job_pop_response = await self.horde_client_session.submit_request(
                job_pop_request,
                ImageGenerateJobPopResponse,
            )

            # TODO: horde_sdk should handle this and return a field with a enum(?) of the reason
            if isinstance(job_pop_response, RequestErrorResponse):
                if "maintenance mode" in job_pop_response.message:
                    logger.warning(f"Failed to pop job (Maintenance Mode): {job_pop_response}")
                else:
                    logger.error(f"Failed to pop job (API Error): {job_pop_response}")
                self._job_pop_frequency = self._error_job_pop_frequency
                return

        except Exception as e:
            if self._job_pop_frequency == self._error_job_pop_frequency:
                logger.error(f"Failed to pop job (Unexpected Error): {e}")
            else:
                logger.warning(f"Failed to pop job (Unexpected Error): {e}")

            self._job_pop_frequency = self._error_job_pop_frequency
            return

        self._job_pop_frequency = self._default_job_pop_frequency

        info_string = "No job available. "
        if len(self.job_deque) > 0:
            info_string += f"Current number of popped jobs: {len(self.job_deque)}. "

        info_string += f"(Skipped reasons: {job_pop_response.skipped.model_dump(exclude_defaults=True)})"

        if job_pop_response.id_ is None:
            logger.info(info_string)
            return

        logger.info(f"Popped job {job_pop_response.id_} (model: {job_pop_response.model})")

        # region TODO: move to horde_sdk
        if job_pop_response.payload.seed is None:  # TODO # FIXME
            logger.warning(f"Job {job_pop_response.id_} has no seed!")
            new_response_dict = job_pop_response.model_dump(by_alias=True)
            new_response_dict["payload"]["seed"] = random.randint(0, (2**32) - 1)

        if job_pop_response.payload.denoising_strength is not None and job_pop_response.source_image is None:
            logger.debug(f"Job {job_pop_response.id_} has denoising_strength but no source image!")
            new_response_dict = job_pop_response.model_dump(by_alias=True)
            new_response_dict["payload"]["denoising_strength"] = None

        if job_pop_response.payload.seed is None or (
            job_pop_response.payload.denoising_strength is not None and job_pop_response.source_image is None
        ):
            job_pop_response = ImageGenerateJobPopResponse(**new_response_dict)

        job_pop_response = await self._get_source_images(job_pop_response)

        # endregion

        if job_pop_response.id_ is None:
            logger.error("Job has no id!")
            return

        async with self._job_deque_lock, self._job_pop_timestamps_lock:
            self.job_deque.append(job_pop_response)
            # self._testing_jobs_added += 1
            self.job_pop_timestamps[str(job_pop_response.id_)] = time.time()

    _user_info_failed = False
    _user_info_failed_reason: str | None = None

    async def api_get_user_info(self) -> None:
        if self._shutting_down:
            return

        request = FindUserRequest(apikey=self.bridge_data.api_key)
        try:
            response = await self.horde_client_session.submit_request(request, FindUserResponse)
            if isinstance(response, RequestErrorResponse):
                logger.error(f"Failed to get user info (API Error): {response}")
                self._user_info_failed = True
                return
            # if self.user_info is None:
            # logger.info(f"Got user info: {response}")  # FIXME

            self.user_info = response
            self._user_info_failed = False
            self._user_info_failed_reason = None

            if self.user_info.kudos_details is not None:
                # print kudos this session and kudos per hour based on self.session_start_time
                kudos_per_hour = self.kudos_generated_this_session / (time.time() - self.session_start_time) * 3600

                if self.kudos_generated_this_session > 0:
                    logger.success(
                        f"Kudos this session: {self.kudos_generated_this_session:.2f} "
                        f"(~{kudos_per_hour:.2f} kudos/hour)",
                    )

                logger.info(f"Worker Kudos Accumulated: {self.user_info.kudos_details.accumulated:.2f}")

        except ClientError as e:
            self._user_info_failed = True
            self._user_info_failed_reason = f"HTTP error (({type(e).__name__}) {e})"

        except Exception as e:
            self._user_info_failed = True
            self._user_info_failed_reason = f"Unexpected error (({type(e).__name__}) {e})"

        finally:
            if self._user_info_failed:
                logger.debug(f"Failed to get user info: {self._user_info_failed_reason}")
            await logger.complete()

    _job_submit_loop_interval = 0.02

    async def _job_submit_loop(self) -> None:
        """Main loop for the job submitter."""
        logger.debug("In _job_submit_loop")
        while True:
            with logger.catch():
                try:
                    await self.api_submit_job()
                    if self.is_time_for_shutdown():
                        break
                except CancelledError:
                    self._shutting_down = True

            await asyncio.sleep(self._job_submit_loop_interval)

    async def _api_call_loop(self) -> None:
        """Main loop for the API calls."""
        logger.debug("In _api_call_loop")
        self._aiohttp_session = ClientSession(requote_redirect_url=False)
        async with self._aiohttp_session as aiohttp_session:
            self.horde_client_session = AIHordeAPIAsyncClientSession(aiohttp_session=aiohttp_session)
            self.horde_client = AIHordeAPIAsyncSimpleClient(
                aiohttp_session=None,
                horde_client_session=self.horde_client_session,
            )
            async with self.horde_client_session:
                while True:
                    with logger.catch():
                        try:
                            if self._user_info_failed:
                                await asyncio.sleep(5)

                            tasks = [self.api_job_pop()]

                            if self._last_get_user_info_time + self._api_get_user_info_interval < time.time():
                                self._last_get_user_info_time = time.time()
                                tasks.append(self.api_get_user_info())

                            if len(tasks) > 0:
                                results = await asyncio.gather(*tasks, return_exceptions=True)

                                # Print all exceptions
                                for result in results:
                                    if isinstance(result, Exception):
                                        logger.exception(f"Exception in api call loop: {result}")

                                if self._user_info_failed:
                                    logger.error("The server failed to respond. Is the horde or your internet down?")

                            if self.is_time_for_shutdown():
                                break
                        except CancelledError:
                            self._shutting_down = True

                    await asyncio.sleep(self._api_call_loop_interval)

    _status_message_frequency = 20.0
    _last_status_message_time = 0.0

    async def _process_control_loop(self) -> None:
        self.start_safety_processes()
        self.start_inference_processes()

        while True:
            try:
                if self.stable_diffusion_reference is None:
                    return

                async with self._job_deque_lock, self._jobs_safety_check_lock, self._completed_jobs_lock:
                    self.receive_and_handle_process_messages()

                if len(self.jobs_pending_safety_check) > 0:
                    async with self._jobs_safety_check_lock:
                        self.start_evaluate_safety()

                if self.is_free_inference_process_available() and len(self.job_deque) > 0:
                    async with self._job_deque_lock, self._jobs_safety_check_lock, self._completed_jobs_lock:
                        # So long as we didn't preload a model this cycle, we can start inference
                        # We want to get any messages next cycle from preloading processes to make sure
                        # the state of everything is up to date
                        if not self.preload_models():
                            self.start_inference()
                        await asyncio.sleep(self._loop_interval / 2)

                async with self._job_deque_lock, self._jobs_safety_check_lock, self._completed_jobs_lock:
                    self.receive_and_handle_process_messages()

                self.unload_models()

                if self._shutting_down:
                    self.end_inference_processes()

                if self.is_time_for_shutdown():
                    self._start_timed_shutdown()
                    break

                if time.time() - self._last_status_message_time > self._status_message_frequency:
                    logger.info(f"{self._process_map}")
                    logger.info(f"Threads being used: {self._max_concurrent_inference_processes}")
                    logger.info(f"Number of jobs popped: {len(self.job_deque)}")
                    logger.info(f"Number of jobs in progress: {len(self.jobs_in_progress)}")
                    logger.info(f"Number of jobs pending safety check: {len(self.jobs_pending_safety_check)}")
                    logger.info(f"Number of jobs being safety checked: {len(self.jobs_being_safety_checked)}")
                    logger.info(f"Number of jobs completed: {len(self.completed_jobs)}")
                    # TODO: Faulted
                    logger.info(f"Number of jobs submitted: {self.total_num_completed_jobs}")

                    self._last_status_message_time = time.time()

                await asyncio.sleep(self._loop_interval / 2)
            except CancelledError:
                self._shutting_down = True

        while len(self.job_deque) > 0:
            await asyncio.sleep(0.2)
            async with self._job_deque_lock, self._jobs_safety_check_lock, self._completed_jobs_lock:
                self.receive_and_handle_process_messages()
            await asyncio.sleep(0.2)

        self.end_safety_processes()
        await asyncio.sleep(0.2)
        self.receive_and_handle_process_messages()

        logger.info("Shutting down process manager")

        for process in self._process_map.values():
            process.mp_process.terminate()
            process.mp_process.terminate()
            process.mp_process.terminate()

            process.mp_process.join(0.2)

        sys.exit(0)

    _bridge_data_loop_interval = 1.0
    _last_bridge_data_reload_time = 0.0

    _bridge_data_last_modified_time = 0.0

    def get_bridge_data_from_disk(self) -> None:
        try:
            self.bridge_data = BridgeDataLoader.load(
                file_path=BRIDGE_CONFIG_FILENAME,
                horde_model_reference_manager=self.horde_model_reference_manager,
            )

            if self.bridge_data.auto_dual_gpu:
                if self.bridge_data.max_threads < 2:
                    logger.warning(
                        f"max_threads in {BRIDGE_CONFIG_FILENAME} must be 2 to use auto_dual_gpu. "
                        "Setting max_threads to 2.",
                    )
                    self.bridge_data.max_threads = 2

                if self.bridge_data.queue_size == 0:
                    logger.warning(
                        f"queue_size in {BRIDGE_CONFIG_FILENAME} must be greater than 0 to use auto_dual_gpu. "
                        "Setting queue_size to 1.",
                    )
                    self.bridge_data.queue_size = 1
            else:
                if self.bridge_data.max_threads != self._max_concurrent_inference_processes:
                    logger.warning(
                        f"max_threads in {BRIDGE_CONFIG_FILENAME} cannot be changed while the worker is running.",
                    )
        except Exception as e:
            logger.debug(e)

            if "No such file or directory" in str(e):
                logger.error(f"Could not find {BRIDGE_CONFIG_FILENAME}. Please create it and try again.")

            if isinstance(e, ValidationError):
                # Print a list of fields that failed validation
                logger.error(f"The following fields in {BRIDGE_CONFIG_FILENAME} failed validation:")
                for error in e.errors():
                    logger.error(f"{error['loc'][0]}: {error['msg']}")

            return

    async def _bridge_data_loop(self) -> None:
        while True:
            try:
                if self._shutting_down:
                    break

                self._bridge_data_last_modified_time = os.path.getmtime(BRIDGE_CONFIG_FILENAME)

                if self._last_bridge_data_reload_time < self._bridge_data_last_modified_time:
                    logger.info(f"Reloading {BRIDGE_CONFIG_FILENAME}")
                    self.get_bridge_data_from_disk()
                    self._last_bridge_data_reload_time = time.time()
                    logger.success(f"Reloaded {BRIDGE_CONFIG_FILENAME}")
                await asyncio.sleep(self._bridge_data_loop_interval)
            except CancelledError:
                self._shutting_down = True

    async def _main_loop(self) -> None:
        # Run both loops concurrently
        await asyncio.gather(
            asyncio.create_task(self._process_control_loop(), name="process_control_loop"),
            asyncio.create_task(self._api_call_loop(), name="api_call_loop"),
            asyncio.create_task(self._job_submit_loop(), name="job_submit_loop"),
            asyncio.create_task(self._bridge_data_loop(), name="bridge_data_loop"),
        )

    _caught_sigints = 0

    def start(self) -> None:
        """Start the process manager."""
        import signal

        signal.signal(signal.SIGINT, self.signal_handler)
        asyncio.run(self._main_loop())

    def signal_handler(self, sig: int, frame: object) -> None:
        """Handle SIGINT and SIGTERM."""

        if self._caught_sigints >= 2:
            logger.warning("Caught SIGINT or SIGTERM twice, exiting immediately")
            sys.exit(1)

        self._caught_sigints += 1
        logger.warning("Shutting down after current jobs are finished...")
        self._shutting_down = True

    def _start_timed_shutdown(self) -> None:
        import threading

        def shutdown() -> None:
            # Just in case the process manager gets stuck on shutdown
            time.sleep((len(self.jobs_pending_safety_check) * 4) + 2)

            for process in self._process_map.values():
                process.mp_process.kill()
                process.mp_process.kill()

                process.mp_process.join()

            sys.exit(0)

        threading.Thread(target=shutdown).start()
