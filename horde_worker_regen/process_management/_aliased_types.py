import multiprocessing.queues
from typing import TYPE_CHECKING

from horde_worker_regen.process_management.messages import HordeProcessMessage

if TYPE_CHECKING:  # noqa: SIM108 (breaks mypy)
    ProcessQueue = multiprocessing.Queue[HordeProcessMessage]
else:
    ProcessQueue = multiprocessing.Queue
