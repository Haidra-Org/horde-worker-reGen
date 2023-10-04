from __future__ import annotations

import abc
import enum
import signal
import sys
import time
from abc import abstractmethod
from enum import auto

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore
from multiprocessing.synchronize import Lock
from typing import TYPE_CHECKING

import psutil
from loguru import logger

from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeProcessHeartbeatMessage,
    HordeProcessMemoryMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
)

if TYPE_CHECKING:
    from hordelib.nodes.node_model_loader import HordeCheckpointLoader
else:
    # Create a dummy class to prevent type errors
    class HordeCheckpointLoader:
        pass


class HordeProcessType(enum.Enum):
    INFERENCE = auto()
    SAFETY = auto()


class HordeProcess(abc.ABC):
    process_id: int
    """The ID of the process. This is not the same as the process PID."""
    process_type: HordeProcessType
    """The type of process. This distinguishes between inference, safety, and potentially other process types."""
    process_message_queue: ProcessQueue
    """The queue the main process uses to receive messages from all worker processes."""
    pipe_connection: Connection  # FIXME # TODO - this could be a Queue?
    """Receives `HordeControlMessage`s from the main process."""

    disk_lock: Lock
    """A lock used to prevent multiple processes from accessing disk at the same time."""

    _loop_interval: float = 0.1
    """The time to sleep between each loop iteration."""

    _end_process: bool = False

    _memory_report_interval: float = 5.0
    """The time to wait between each memory report."""

    _last_sent_process_state: HordeProcessState = HordeProcessState.PROCESS_STARTING

    _vram_total_bytes: int = 0

    def get_vram_usage_bytes(self) -> int:
        from hordelib.comfy_horde import get_torch_free_vram_mb, get_torch_total_vram_mb

        return get_torch_total_vram_mb() - get_torch_free_vram_mb()

    def get_vram_total_bytes(self) -> int:
        from hordelib.comfy_horde import get_torch_total_vram_mb

        return get_torch_total_vram_mb()

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
    ) -> None:
        self.process_id = process_id
        self.process_message_queue = process_message_queue
        self.pipe_connection = pipe_connection
        self.disk_lock = disk_lock

        # Remove all handlers from the root logger
        logger.remove()
        from hordelib.utils.logger import HordeLog

        HordeLog.initialise(
            setup_logging=True,
            process_id=process_id,
            verbosity_count=5,  # FIXME
        )

        self.send_process_state_change_message(
            process_state=HordeProcessState.PROCESS_STARTING,
            info="Process starting",
        )

    def send_process_state_change_message(
        self,
        process_state: HordeProcessState,
        info: str,
        time_elapsed: float | None = None,
    ) -> None:
        message = HordeProcessStateChangeMessage(
            process_state=process_state,
            process_id=self.process_id,
            info=info,
            time_elapsed=time_elapsed,
        )
        self.process_message_queue.put(message)
        self._last_sent_process_state = process_state

    _heartbeat_limit_interval_seconds: float = 5.0
    _last_heartbeat_time: float = 0.0

    def send_heartbeat_message(self) -> None:
        """Send a heartbeat message to the main process."""

        if (time.time() - self._last_heartbeat_time) < self._heartbeat_limit_interval_seconds:
            return

        message = HordeProcessHeartbeatMessage(
            process_id=self.process_id,
            info="Heartbeat",
            time_elapsed=None,
        )
        self.process_message_queue.put(message)

        self._last_heartbeat_time = time.time()

    @abstractmethod
    def cleanup_and_exit(self) -> None:
        """Cleanup and exit the process."""

    def send_memory_report_message(
        self,
        include_vram: bool = False,
    ) -> None:
        """Send a memory report message to the main process."""
        message = HordeProcessMemoryMessage(
            process_id=self.process_id,
            info="Memory report",
            time_elapsed=None,
            ram_usage_bytes=psutil.Process().memory_info().rss,
        )

        if include_vram:
            message.vram_usage_bytes = self.get_vram_usage_bytes()
            message.vram_total_bytes = self.get_vram_total_bytes()

        self.process_message_queue.put(message)

    @abstractmethod
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        """Receive and handle a control message from the main process."""

    def receive_and_handle_control_messages(self) -> None:
        """Get and handle any control messages pending from the main process."""
        while self.pipe_connection.poll():
            message = self.pipe_connection.recv()

            if not isinstance(message, HordeControlMessage):
                logger.critical(f"Received unexpected message type: {type(message).__name__}")
                continue

            if message.control_flag == HordeControlFlag.END_PROCESS:
                self._end_process = True
                logger.info("Received end process message")
                return

            self._receive_and_handle_control_message(message)

    def worker_cycle(self) -> None:
        """Called after messages have been received and handled. Override this to implement any additional logic."""
        return

    def main_loop(self) -> None:
        """The main loop of the worker process."""
        signal.signal(signal.SIGINT, signal_handler)

        while not self._end_process:
            try:
                time.sleep(self._loop_interval)

                self.receive_and_handle_control_messages()

                self.worker_cycle()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")

        self.send_process_state_change_message(
            process_state=HordeProcessState.PROCESS_ENDING,
            info="Process ending",
        )

        self.cleanup_and_exit()

        logger.info("Process ended")
        self.send_process_state_change_message(
            process_state=HordeProcessState.PROCESS_ENDED,
            info="Process ended",
        )
        sys.exit(1)


def signal_handler(sig: int, frame: object) -> None:
    print("You pressed Ctrl+C!")
