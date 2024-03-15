"""The main entry point for the reGen worker."""

import argparse
import contextlib
import io
import multiprocessing
import os
import sys
import time
from multiprocessing.context import BaseContext

from loguru import logger


def main(ctx: BaseContext, load_from_env_vars: bool = False) -> None:
    """Check for a valid config and start the driver ('main') process for the reGen worker."""
    from horde_model_reference.model_reference_manager import ModelReferenceManager
    from pydantic import ValidationError

    from horde_worker_regen.bridge_data.load_config import BridgeDataLoader, reGenBridgeData
    from horde_worker_regen.consts import BRIDGE_CONFIG_FILENAME
    from horde_worker_regen.process_management.main_entry_point import start_working

    def ensure_model_db_downloaded() -> ModelReferenceManager:
        horde_model_reference_manager = ModelReferenceManager(
            download_and_convert_legacy_dbs=False,
            override_existing=True,
        )

        while True:
            try:
                with logger.catch(reraise=True):
                    if not horde_model_reference_manager.download_and_convert_all_legacy_dbs(override_existing=True):
                        logger.error("Failed to download and convert legacy DBs. Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        return horde_model_reference_manager
            except Exception as e:
                logger.error(f"Failed to download and convert legacy DBs: ({type(e).__name__}) {e}")
                logger.error("Retrying in 5 seconds...")
                time.sleep(5)

    horde_model_reference_manager = ensure_model_db_downloaded()

    bridge_data: reGenBridgeData | None = None
    try:
        if load_from_env_vars:
            bridge_data = BridgeDataLoader.load_from_env_vars(
                horde_model_reference_manager=horde_model_reference_manager,
            )

            if len(bridge_data.api_key) == 10:
                logger.error(
                    "API key is the default. This is almost certainly not what you want. "
                    "Please check your environment variables are being set correctly and try again.",
                )

                logger.error("Exiting...")
                return
        else:
            bridge_data = BridgeDataLoader.load(
                file_path=BRIDGE_CONFIG_FILENAME,
                horde_model_reference_manager=horde_model_reference_manager,
            )
    except ConnectionRefusedError:
        logger.error("Could not connect to the the horde. Is it down?")
        input("Press any key to exit...")
        return
    except Exception as e:
        logger.exception(e)

        if "No such file or directory" in str(e):
            logger.error(f"Could not find {BRIDGE_CONFIG_FILENAME}. Please create it and try again.")

        if isinstance(e, ValidationError):
            # Print a list of fields that failed validation
            logger.error(f"The following fields in {BRIDGE_CONFIG_FILENAME} failed validation:")
            for error in e.errors():
                logger.error(f"{error['loc'][0]}: {error['msg']}")

        input("Press any key to exit...")
        return

    if not bridge_data:
        logger.error("Failed to load bridge data. Exiting...")
        return

    bridge_data.load_env_vars()

    start_working(
        ctx=ctx,
        bridge_data=bridge_data,
        horde_model_reference_manager=horde_model_reference_manager,
    )


class LogConsoleRewriter(io.StringIO):
    """Makes the console output more readable by shortening certain strings."""

    def __init__(self, original_stdout: io.TextIOBase) -> None:
        """Initialise the rewriter."""
        self.original_stdout = original_stdout

    def write(self, message: str) -> int:
        """Rewrite the message to make it more readable where possible."""
        replacements = [
            ("horde_worker_regen.process_management.process_manager", "[HWRPM]"),
            ("horde_worker_regen.", "[HWR]"),
        ]

        for old, new in replacements:
            message = message.replace(old, new)

        return sys.__stdout__.write(message)

    def flush(self) -> None:
        """Flush the buffer to the original stdout."""
        self.original_stdout.flush()


def init() -> None:
    """Initialise the worker, including logging, environment variables, and other housekeeping."""
    with contextlib.suppress(Exception):
        multiprocessing.set_start_method("spawn", force=True)

    print(f"Multiprocessing start method: {multiprocessing.get_start_method()}")

    # Create args for -v, allowing -vvv
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action="count", default=0, help="Increase verbosity of output")
    parser.add_argument("--no-logging", action="store_true", help="Disable logging to the console")
    parser.add_argument(
        "-e",
        "--load-config-from-env-vars",
        action="store_true",
        default=False,
        help="Load the config only from environment variables. This is useful for running the worker in a container.",
    )

    args = parser.parse_args()

    os.environ["HORDE_SDK_DISABLE_CUSTOM_SINKS"] = "1"

    from horde_worker_regen.load_env_vars import load_env_vars_from_config

    if not args.load_config_from_env_vars:
        # Note: 'load_env_vars_from_config' means to translate the config file to environment variables
        # if 'load_config_from_env_vars' is True, then we are ignoring the config file
        load_env_vars_from_config()

    from horde_worker_regen.version_meta import do_version_check

    do_version_check()

    rewriter = LogConsoleRewriter(sys.stdout)  # type: ignore
    sys.stdout = rewriter

    logger.remove()
    from hordelib.utils.logger import HordeLog

    target_verbosity = args.v

    if args.no_logging:
        target_verbosity = 0  # Disable logging to the console
    elif args.v == 0:
        target_verbosity = 3  # Default to INFO or higher (Warning, Error, Critical)

    # Initialise logging with loguru
    HordeLog.initialise(
        setup_logging=True,
        process_id=None,
        verbosity_count=target_verbosity,
    )

    # We only need to download the legacy DBs once, so we do it here instead of in the worker processes

    main(multiprocessing.get_context("spawn"), args.load_config_from_env_vars)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    init()
