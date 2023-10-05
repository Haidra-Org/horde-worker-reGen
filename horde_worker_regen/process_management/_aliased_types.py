import multiprocessing.queues
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # noqa: SIM108 (breaks mypy)
    from horde_worker_regen.process_management.messages import HordeProcessMessage

    ProcessQueue = multiprocessing.Queue[HordeProcessMessage]
else:
    ProcessQueue = multiprocessing.Queue
