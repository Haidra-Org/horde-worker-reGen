import contextlib
import os
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
    CUDA_VISIBLE_DEVICES: str | None = None,
) -> None:
    if CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
        logger.info(f"Set CUDA_VISIBLE_DEVICES to {CUDA_VISIBLE_DEVICES}")

    with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        logger.remove()
        import hordelib

        try:
            hordelib.initialise(
                setup_logging=None,
                process_id=process_id,
                logging_verbosity=0,
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
) -> None:
    with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        logger.remove()
        import hordelib

        try:
            hordelib.initialise(
                setup_logging=None,
                process_id=process_id,
                logging_verbosity=0,
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
