from multiprocessing.context import BaseContext

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager


def start_working(ctx: BaseContext, bridge_data: reGenBridgeData) -> None:
    process_manager = HordeWorkerProcessManager(
        ctx=ctx,
        bridge_data=bridge_data,
    )

    process_manager.start()
