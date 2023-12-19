"""Contains the code to download all models specified in the config file. Executable as a standalone script."""

from load_env_vars import load_env_vars

load_env_vars()

import argparse
import time

from horde_model_reference.model_reference_manager import ModelReferenceManager
from loguru import logger

from horde_worker_regen.bridge_data.load_config import BridgeDataLoader, reGenBridgeData
from horde_worker_regen.consts import BRIDGE_CONFIG_FILENAME


def download_all_models(purge_unused_loras: bool = False) -> None:
    """Download all models specified in the config file."""
    horde_model_reference_manager = ModelReferenceManager(
        download_and_convert_legacy_dbs=True,
        override_existing=True,
    )

    if not horde_model_reference_manager.download_and_convert_all_legacy_dbs(override_existing=True):
        logger.error("Failed to download and convert legacy DBs. Retrying in 5 seconds...")

    bridge_data: reGenBridgeData
    try:
        bridge_data = BridgeDataLoader.load(
            file_path=BRIDGE_CONFIG_FILENAME,
            horde_model_reference_manager=horde_model_reference_manager,
        )
        bridge_data.load_env_vars()
    except Exception as e:
        logger.error(e)
        input("Press any key to exit...")

    import hordelib
    from horde_safety.deep_danbooru_model import get_deep_danbooru_model
    from horde_safety.interrogate import get_interrogator_no_blip

    _ = get_deep_danbooru_model()
    del _
    _ = get_interrogator_no_blip()
    del _

    hordelib.initialise()
    from hordelib.shared_model_manager import SharedModelManager

    SharedModelManager.load_model_managers()

    if purge_unused_loras:
        logger.info("Purging unused LORAs...")
        if SharedModelManager.manager.lora is None:
            logger.error("Failed to load LORA model manager")
            exit(1)
        deleted_loras = SharedModelManager.manager.lora.delete_unused_loras(30)
        logger.success(f"Purged {len(deleted_loras)} unused LORAs.")

    if bridge_data.allow_lora:
        if SharedModelManager.manager.lora is None:
            logger.error("Failed to load LORA model manager")
            exit(1)
        SharedModelManager.manager.lora.download_default_loras()

        while SharedModelManager.manager.lora.are_downloads_complete() is False:
            logger.info("Waiting for LORA downloads to complete...")
            time.sleep(8)

    if bridge_data.allow_controlnet:
        if SharedModelManager.manager.controlnet is None:
            logger.error("Failed to load controlnet model manager")
            exit(1)
        SharedModelManager.manager.controlnet.download_all_models()

    if bridge_data.allow_post_processing:
        if SharedModelManager.manager.gfpgan is None:
            logger.error("Failed to load GFPGAN model manager")
            exit(1)
        if SharedModelManager.manager.esrgan is None:
            logger.error("Failed to load ESRGAN model manager")
            exit(1)
        if SharedModelManager.manager.codeformer is None:
            logger.error("Failed to load codeformer model manager")
            exit(1)

        if not SharedModelManager.manager.gfpgan.download_all_models():
            logger.error("Failed to download all GFPGAN models")
        else:
            logger.success("Downloaded all GFPGAN models")
        if not SharedModelManager.manager.esrgan.download_all_models():
            logger.error("Failed to download all ESRGAN models")
        else:
            logger.success("Downloaded all ESRGAN models")
        if not SharedModelManager.manager.codeformer.download_all_models():
            logger.error("Failed to download all codeformer models")
        else:
            logger.success("Downloaded all codeformer models")

    if SharedModelManager.manager.compvis is None:
        logger.error("Failed to load compvis model manager")
        exit(1)

    any_model_failed_to_download = False
    for model in bridge_data.image_models_to_load:
        if not SharedModelManager.manager.compvis.download_model(model):
            logger.error(f"Failed to download model {model}")
            any_model_failed_to_download = True

    if any_model_failed_to_download:
        logger.error("Failed to download all models.")
    else:
        logger.success("Downloaded all compvis (Stable Diffusion) models.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all models specified in the config file.")
    parser.add_argument(
        "--purge-unused-loras",
        action="store_true",
        help="Purge unused LORAs from the cache",
    )

    args = parser.parse_args()

    download_all_models(purge_unused_loras=args.purge_unused_loras)
