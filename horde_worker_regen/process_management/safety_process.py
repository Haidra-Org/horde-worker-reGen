"""Contains the classes to form a safety process, which is responsible for evaluating the safety of images."""
import base64
import enum
import time
from enum import auto
from io import BytesIO
from typing import TYPE_CHECKING

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore
from multiprocessing.synchronize import Lock

import PIL
import PIL.Image
from loguru import logger
from typing_extensions import override

from horde_worker_regen import ASSETS_FOLDER_PATH
from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.horde_process import HordeProcess
from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeProcessState,
    HordeSafetyControlMessage,
    HordeSafetyEvaluation,
    HordeSafetyResultMessage,
)

if TYPE_CHECKING:
    from horde_safety.deep_danbooru_model import DeepDanbooruModel
    from horde_safety.interrogate import Interrogator
    from horde_safety.nsfw_checker_class import NSFWChecker, NSFWResult
else:

    class Interrogator:
        """Dummy class to prevent type errors."""

    class NSFWChecker:
        """Dummy class to prevent type errors."""

    class NSFWResult:
        """Dummy class to prevent type errors."""

    class DeepDanbooruModel:
        """Dummy class to prevent type errors."""


class CensorReason(enum.Enum):
    """The reason for censoring an image."""

    CSAM = auto()
    CENSORLIST = auto()
    SFW_REQUEST = auto()
    SFW_WORKER = auto()


class HordeSafetyProcess(HordeProcess):
    """The safety process, which is responsible for evaluating the safety of images."""

    _interrogator: Interrogator
    _deep_danbooru_model: DeepDanbooruModel

    _nsfw_checker: NSFWChecker

    censor_csam_image_base64: str
    censor_censorlist_image_base64: str
    censor_sfw_request_image_base64: str
    censor_sfw_worker_image_base64: str

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        cpu_only: bool = True,
    ) -> None:
        """Initialise the safety process.

        Args:
            process_id (int): The ID of the process.
            process_message_queue (ProcessQueue): The process message queue.
            pipe_connection (Connection): The connection to the parent process.
            disk_lock (Lock): The lock to use when accessing the disk.
            cpu_only (bool, optional): Whether to only use the CPU. Defaults to True.
        """

        super().__init__(process_id, process_message_queue, pipe_connection, disk_lock)

        from horde_safety.deep_danbooru_model import get_deep_danbooru_model
        from horde_safety.interrogate import get_interrogator_no_blip

        self._deep_danbooru_model = get_deep_danbooru_model(device="cpu" if cpu_only else "cuda")
        self._interrogator = get_interrogator_no_blip(device="cpu" if cpu_only else "cuda")

        from horde_safety.nsfw_checker_class import NSFWChecker

        self._nsfw_checker = NSFWChecker(
            self._interrogator,
            self._deep_danbooru_model,  # Optional, significantly improves results for anime images
        )

        self.load_censor_files()

        info_message = "Horde safety process started."

        logger.info(info_message)
        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info=info_message,
        )

        logger.info(
            "The first job will always take several seconds longer when on CPU. Subsequent jobs will be faster.",
        )

    def _set_censor_image(self, reason: CensorReason, image_base64: str) -> None:
        if reason == CensorReason.CSAM:
            self.censor_csam_image_base64 = image_base64
        elif reason == CensorReason.CENSORLIST:
            self.censor_censorlist_image_base64 = image_base64
        elif reason == CensorReason.SFW_REQUEST:
            self.censor_sfw_request_image_base64 = image_base64
        elif reason == CensorReason.SFW_WORKER:
            self.censor_sfw_worker_image_base64 = image_base64
        else:
            raise ValueError(f"Unknown censor reason: {reason}")

    def load_censor_files(self) -> None:
        """Load the censor images from disk."""
        file_lookup = {
            CensorReason.CSAM: "nsfw_censor_csam.png",
            CensorReason.CENSORLIST: "nsfw_censor_censorlist.png",
            CensorReason.SFW_REQUEST: "nsfw_censor_sfw_request.png",
            CensorReason.SFW_WORKER: "nsfw_censor_sfw_worker.png",
        }

        for reason in CensorReason:
            with open(ASSETS_FOLDER_PATH / file_lookup[reason], "rb") as f:
                self._set_censor_image(reason, base64.b64encode(f.read()).decode("utf-8"))

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        if not isinstance(message, HordeSafetyControlMessage):
            raise TypeError(f"Expected {HordeSafetyControlMessage}, got {type(message)}")

        if message.control_flag != HordeControlFlag.EVALUATE_SAFETY:
            raise ValueError(f"Expected {HordeControlFlag.EVALUATE_SAFETY}, got {message.control_flag}")

        self.send_memory_report_message(include_vram=False)

        time_start = time.time()

        logger.info(
            f"Horde safety process received job {message.job_id}. Number of images: {len(message.images_base64)}",
        )

        safety_evaluations: list[HordeSafetyEvaluation] = []

        for image_base64 in message.images_base64:
            # Decode the image from base64
            image_bytes = BytesIO(base64.b64decode(image_base64))
            try:
                image_as_pil = PIL.Image.open(image_bytes)
            except Exception as e:
                logger.error(f"Failed to open image: {type(e).__name__} {e}")
                safety_evaluations.append(
                    HordeSafetyEvaluation(
                        is_nsfw=True,
                        is_csam=True,
                        replacement_image_base64=None,
                        failed=True,
                    ),
                )

                continue

            nsfw_result: NSFWResult | None = self._nsfw_checker.check_for_nsfw(
                image=image_as_pil,
                prompt=message.prompt,
                model_info=message.horde_model_info,
            )

            if nsfw_result is None:
                raise RuntimeError("NSFW result is None")

            replacement_image_base64: str | None = None

            if nsfw_result.is_csam:
                replacement_image_base64 = self.censor_csam_image_base64
                logger.debug(f"CSAM detected in image {message.job_id}. Image is deleted.")
            elif message.sfw_worker and nsfw_result.is_nsfw:
                replacement_image_base64 = self.censor_sfw_worker_image_base64
                logger.info(f"SFW worker detected NSFW in image {message.job_id}.")
            elif message.censor_nsfw and nsfw_result.is_nsfw:
                replacement_image_base64 = self.censor_sfw_request_image_base64
                logger.info(f"Censor list detected NSFW in image {message.job_id}.")

            safety_evaluations.append(
                HordeSafetyEvaluation(
                    is_nsfw=nsfw_result.is_nsfw,
                    is_csam=nsfw_result.is_csam,
                    replacement_image_base64=replacement_image_base64,
                ),
            )

        time_elapsed = time.time() - time_start

        info_message = f"Finished evaluating safety for job {message.job_id}"
        logger.info(info_message)

        self.process_message_queue.put(
            HordeSafetyResultMessage(
                process_id=self.process_id,
                info=info_message,
                time_elapsed=time_elapsed,
                job_id=message.job_id,
                safety_evaluations=safety_evaluations,
            ),
        )
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")

    @override
    def cleanup_for_exit(self) -> None:
        return super().cleanup_for_exit()
