"""Contains the code to download all models specified in the config file. Executable as a standalone script."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hordelib.shared_model_manager import SharedModelManager

    from horde_worker_regen.bridge_data.data_model import reGenBridgeData


def download_all_models(
    *,
    load_config_from_env_vars: bool = False,
    purge_unused_loras: bool = False,
    directml: int | None = None,
) -> None:
    """Download all models specified in the config file."""
    from horde_worker_regen.load_env_vars import load_env_vars_from_config

    if not load_config_from_env_vars:
        load_env_vars_from_config()

    from horde_model_reference.model_reference_manager import ModelReferenceManager
    from loguru import logger

    from horde_worker_regen.bridge_data.load_config import BridgeDataLoader
    from horde_worker_regen.consts import BRIDGE_CONFIG_FILENAME

    horde_model_reference_manager = ModelReferenceManager(
        download_and_convert_legacy_dbs=True,
        override_existing=True,
    )

    if not horde_model_reference_manager.download_and_convert_all_legacy_dbs(override_existing=True):
        logger.error("Failed to download and convert legacy DBs. Retrying in 5 seconds...")

    bridge_data: reGenBridgeData | None = None
    try:
        if not load_config_from_env_vars:
            bridge_data = BridgeDataLoader.load(
                file_path=BRIDGE_CONFIG_FILENAME,
                horde_model_reference_manager=horde_model_reference_manager,
            )
            bridge_data.load_env_vars()
            bridge_data.prepare_custom_models()
        else:
            bridge_data = BridgeDataLoader.load_from_env_vars(
                horde_model_reference_manager=horde_model_reference_manager,
            )
    except Exception as e:
        logger.error(e)
        input("Press any key to exit...")

    if bridge_data is None:
        logger.error("Failed to load bridge data")
        exit(1)

    import hordelib
    from horde_safety.deep_danbooru_model import download_deep_danbooru_model
    from horde_safety.interrogate import get_interrogator_no_blip

    download_deep_danbooru_model()
    _ = get_interrogator_no_blip()
    del _

    extra_comfyui_args = []
    if directml is not None:
        extra_comfyui_args.append(f"--directml={directml}")

    hordelib.initialise(extra_comfyui_args=extra_comfyui_args)
    from hordelib.shared_model_manager import SharedModelManager

    SharedModelManager.load_model_managers()

    _log_pending_download_summary(bridge_data, SharedModelManager)

    if bridge_data.allow_lora:
        if SharedModelManager.manager.lora is None:
            logger.error("Failed to load LORA model manager")
            exit(1)
        SharedModelManager.manager.lora.reset_adhoc_loras()
        SharedModelManager.manager.lora.download_default_loras(bridge_data.nsfw)
        SharedModelManager.manager.lora.wait_for_downloads(600)
        SharedModelManager.manager.lora.wait_for_adhoc_reset(120)

    if purge_unused_loras or bridge_data.purge_loras_on_download:
        logger.info("Purging unused LORAs...")
        if SharedModelManager.manager.lora is None:
            logger.error("Failed to load LORA model manager")
            exit(1)
        deleted_loras = SharedModelManager.manager.lora.delete_unused_loras(30)
        logger.success(f"Purged {len(deleted_loras)} unused LORAs.")

    if bridge_data.allow_controlnet:
        if SharedModelManager.manager.controlnet is None:
            logger.error("Failed to load controlnet model manager")
            exit(1)
        for cn_model in SharedModelManager.manager.controlnet.model_reference:
            if (
                cn_model not in SharedModelManager.manager.controlnet.available_models
                and "sdxl" in cn_model.lower()
                and not bridge_data.allow_sdxl_controlnet
            ):
                logger.warning(f"Skipping download of {cn_model} because `allow_sdxl_controlnet` is false.")
                continue

            SharedModelManager.manager.controlnet.download_model(cn_model)
            SharedModelManager.manager.controlnet.download_model(cn_model)
        if not SharedModelManager.preload_annotators():
            logger.error("Failed to download the controlnet annotators")
            exit(1)

    if bridge_data.allow_sdxl_controlnet:
        if SharedModelManager.manager.miscellaneous is None:
            logger.error("Failed to load miscellaneous model manager")
            exit(1)
        SharedModelManager.manager.miscellaneous.download_all_models()
        SharedModelManager.manager.miscellaneous.download_all_models()
        for model in SharedModelManager.manager.miscellaneous.model_reference:
            if not SharedModelManager.manager.miscellaneous.validate_model(
                model,
            ) and not SharedModelManager.manager.miscellaneous.download_model(model):
                logger.error(f"Failed to download model {model}")
                exit(1)
        else:
            logger.success("Downloaded all Miscellaneous models")

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

        SharedModelManager.manager.gfpgan.download_all_models()
        SharedModelManager.manager.gfpgan.download_all_models()
        for model in SharedModelManager.manager.gfpgan.model_reference:
            if not SharedModelManager.manager.gfpgan.validate_model(
                model,
            ) and not SharedModelManager.manager.gfpgan.download_model(model):
                logger.error(f"Failed to download model {model}")
                exit(1)
        else:
            logger.success("Downloaded all GFPGAN models")

        SharedModelManager.manager.esrgan.download_all_models()
        SharedModelManager.manager.esrgan.download_all_models()
        for model in SharedModelManager.manager.esrgan.model_reference:
            if not SharedModelManager.manager.esrgan.validate_model(
                model,
            ) and not SharedModelManager.manager.esrgan.download_model(model):
                logger.error(f"Failed to download model {model}")
                exit(1)
        else:
            logger.success("Downloaded all ESRGAN models")

        SharedModelManager.manager.codeformer.download_all_models()
        SharedModelManager.manager.codeformer.download_all_models()
        for model in SharedModelManager.manager.codeformer.model_reference:
            if not SharedModelManager.manager.codeformer.validate_model(
                model,
            ) and not SharedModelManager.manager.codeformer.download_model(model):
                logger.error(f"Failed to download model {model}")
                exit(1)

    if SharedModelManager.manager.compvis is None:
        logger.error("Failed to load compvis model manager")
        exit(1)

    any_compvis_model_failed_to_download = False
    for model in bridge_data.image_models_to_load:
        if not SharedModelManager.manager.compvis.download_model(model):
            logger.error(f"Failed to download model {model}")
            any_compvis_model_failed_to_download = True

        # This will check the SHA of the model and redownload it if it's corrupted or the model reference entry changed
        if not SharedModelManager.manager.compvis.validate_model(model):  # noqa: SIM102
            if not SharedModelManager.manager.compvis.download_model(model):
                logger.error(f"Failed to redownload model {model}")
                any_compvis_model_failed_to_download = True

    if any_compvis_model_failed_to_download:
        logger.error("Failed to download all models.")
        exit(1)
    else:
        logger.success("Downloaded all compvis (Stable Diffusion) models.")


def _log_pending_download_summary(
    bridge_data: reGenBridgeData,
    shared_model_manager: type[SharedModelManager],
) -> None:
    """Log how many model files are about to be downloaded, their total size, and whether they fit on disk.

    Mirrors the download decisions made in `download_all_models` (see issues #400 and #267). LoRA downloads
    are resolved dynamically by the LoRA model manager and are not included in the summary.
    """
    from loguru import logger

    from horde_worker_regen.download_summary import (
        PendingDownload,
        collect_pending_downloads,
        log_download_summary,
        populate_download_sizes,
    )

    manager = shared_model_manager.manager
    pending_by_category: dict[str, list[PendingDownload]] = {}

    if bridge_data.allow_controlnet and manager.controlnet is not None:
        controlnet_models = [
            cn_model
            for cn_model in manager.controlnet.model_reference
            if bridge_data.allow_sdxl_controlnet or "sdxl" not in cn_model.lower()
        ]
        pending_by_category["ControlNet"] = collect_pending_downloads(manager.controlnet, controlnet_models)

    if bridge_data.allow_sdxl_controlnet and manager.miscellaneous is not None:
        pending_by_category["Miscellaneous"] = collect_pending_downloads(
            manager.miscellaneous,
            manager.miscellaneous.model_reference,
        )

    if bridge_data.allow_post_processing:
        post_processing_managers = {
            "GFPGAN": manager.gfpgan,
            "ESRGAN": manager.esrgan,
            "CodeFormer": manager.codeformer,
        }
        for category, post_processing_manager in post_processing_managers.items():
            if post_processing_manager is not None:
                pending_by_category[category] = collect_pending_downloads(
                    post_processing_manager,
                    post_processing_manager.model_reference,
                )

    if manager.compvis is not None:
        pending_by_category["Stable Diffusion"] = collect_pending_downloads(
            manager.compvis,
            bridge_data.image_models_to_load,
        )

    all_pending = [pending for category_pending in pending_by_category.values() for pending in category_pending]
    if all_pending:
        logger.info(f"Checking the size of {len(all_pending)} model file(s) to download...")
        populate_download_sizes(all_pending)

    download_target_dir = manager.compvis.model_folder_path if manager.compvis is not None else None
    log_download_summary(pending_by_category, download_target_dir=download_target_dir)
