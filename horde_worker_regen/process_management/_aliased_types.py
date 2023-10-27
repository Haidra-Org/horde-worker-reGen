"""Contains some work arounds for typing failing at runtime."""
import multiprocessing.queues
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # noqa: SIM108 (breaks mypy)
    from horde_worker_regen.process_management.messages import HordeProcessMessage

    ProcessQueue = multiprocessing.Queue[HordeProcessMessage]
    # Pylance (maybe others?) seem to understand the meaning here (the queue contains `HordeProcessMessage`)
    # but using this syntax causes an exception at runtime
else:
    ProcessQueue = multiprocessing.Queue
