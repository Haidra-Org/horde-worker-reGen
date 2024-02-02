import contextlib
import sys

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore
from multiprocessing.synchronize import Lock, Semaphore

from loguru import logger

from horde_worker_regen.process_management._aliased_types import ProcessQueue


def start_inference_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    inference_semaphore: Semaphore,
    disk_lock: Lock,
    amd: bool = False,
) -> None:
    """Start an inference process.

    Args:
        process_id (int): The ID of the process. This is not the same as the PID.
        process_message_queue (ProcessQueue): The queue to send messages to the main process.
        pipe_connection (Connection): Receives `HordeControlMessage`s from the main process.
        inference_semaphore (Semaphore): The semaphore to use to limit concurrent inference.
        disk_lock (Lock): The lock to use for disk access.
    """
    with contextlib.nullcontext():  # contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        logger.remove()

        try:
            import hordelib
            from hordelib.utils.logger import HordeLog

            HordeLog.initialise(
                setup_logging=True,
                process_id=process_id,
                verbosity_count=5,  # FIXME
            )

            with logger.catch(reraise=True):
                hordelib.initialise(
                    setup_logging=None,
                    process_id=process_id,
                    logging_verbosity=0,
                    extra_comfyui_args=None if not amd else ["--directml"],
                )
        except Exception as e:
            logger.critical(f"Failed to initialise hordelib: {type(e).__name__} {e}")
            sys.exit(1)

        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        worker_process = HordeInferenceProcess(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            inference_semaphore=inference_semaphore,
            disk_lock=disk_lock,
        )

        worker_process.main_loop()


def start_safety_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    disk_lock: Lock,
    cpu_only: bool = True,
    amd: bool = False,
) -> None:
    """Start a safety process.

    Args:
        process_id (int): The ID of the process. This is not the same as the PID.
        process_message_queue (ProcessQueue): The queue to send messages to the main process.
        pipe_connection (Connection): Receives `HordeControlMessage`s from the main process.
        disk_lock (Lock): The lock to use for disk access.
        cpu_only (bool, optional): _description_. Defaults to True.
    """
    with contextlib.nullcontext():  # contextlib.redirect_stdout(), contextlib.redirect_stderr():
        logger.remove()

        try:
            import hordelib
            from hordelib.utils.logger import HordeLog

            HordeLog.initialise(
                setup_logging=True,
                process_id=process_id,
                verbosity_count=5,  # FIXME
            )
            with logger.catch(reraise=True):
                hordelib.initialise(
                    setup_logging=None,
                    process_id=process_id,
                    logging_verbosity=0,
                    extra_comfyui_args=None if not amd else ["--directml"],
                )
        except Exception as e:
            logger.critical(f"Failed to initialise hordelib: {type(e).__name__} {e}")
            sys.exit(1)

        from horde_worker_regen.process_management.safety_process import HordeSafetyProcess

        worker_process = HordeSafetyProcess(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            cpu_only=cpu_only,
        )

        worker_process.main_loop()
