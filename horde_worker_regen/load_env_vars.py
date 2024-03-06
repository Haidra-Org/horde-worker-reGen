"""Contains the functions to load the environment variables from the config file."""

import datetime
import os
import pathlib
import time

import semver
from dotenv import load_dotenv
from loguru import logger
from ruamel.yaml import YAML

import horde_worker_regen
from horde_worker_regen.version_meta import VersionMeta, get_local_version_meta, get_remote_version_meta

load_dotenv()


def load_env_vars() -> None:  # FIXME: there is a dynamic way to do this
    """Load the environment variables from the config file."""
    yaml = YAML()
    config_file = "bridgeData.yaml"
    template_file = "bridgeData_template.yaml"

    if not pathlib.Path(config_file).exists():
        if pathlib.Path(template_file).exists():
            raise FileNotFoundError(f"{template_file} found. Please set variables and rename it to {config_file}.")
        raise FileNotFoundError(f"{config_file} not found")

    with open(config_file, encoding="utf-8") as f:
        config = yaml.load(f)

    if "cache_home" in config:
        if os.getenv("AIWORKER_CACHE_HOME") is None:
            os.environ["AIWORKER_CACHE_HOME"] = config["cache_home"]
        else:
            print(
                "AIWORKER_CACHE_HOME environment variable already set. "
                "This will override the value for `cache_home` in the config file.",
            )

    if "max_lora_cache_size" in config:
        if os.getenv("AIWORKER_LORA_CACHE_SIZE") is None:
            try:
                int(config["max_lora_cache_size"])
            except ValueError as e:
                raise ValueError(
                    "max_lora_cache_size must be an integer, but is not.",
                ) from e
            os.environ["AIWORKER_LORA_CACHE_SIZE"] = str(config["max_lora_cache_size"])
        else:
            print(
                "AIWORKER_LORA_CACHE_SIZE environment variable already set. "
                "This will override the value for `max_lora_cache_size` in the config file.",
            )
    if "civitai_api_token" in config:
        if os.getenv("CIVIT_API_TOKEN") is None:
            os.environ["CIVIT_API_TOKEN"] = config["civitai_api_token"]
        else:
            print(
                "CIVIT_API_TOKEN environment variable already set. "
                "This will override the value for `civitai_api_token` in the config file.",
            )

    version_meta: VersionMeta
    try:
        version_meta = get_remote_version_meta()
    except Exception as e:
        logger.warning(f"Failed to get remote version meta: {e}")
        logger.warning("Using local version meta instead.")
        logger.warning("If this keeps happening, please check your internet connection and try again.")
        version_meta = get_local_version_meta()

    # If the required_min_version is not satisfied, raise an error
    if not semver.compare(horde_worker_regen.__version__, version_meta.required_min_version) >= 0:
        # Get the reason for the required update
        reason_for_update = version_meta.required_min_version_info[version_meta.required_min_version].reason_for_update

        reason_for_update_str = f"Reason for update: {reason_for_update}" if reason_for_update else ""

        # UTC time
        current_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

        # If we're before the required_version_update_date, just warn, otherwise raise an error
        if current_date < version_meta.required_min_version_update_date:
            logger.warning(
                f"Current worker version {horde_worker_regen.__version__} has a required update to "
                f"{version_meta.required_min_version}. "
                f"Please update to the required version by {version_meta.required_min_version_update_date}.",
            )
            logger.warning(reason_for_update_str)

            os.environ["AIWORKER_NOT_REQUIRED_VERSION"] = "1"

        else:
            logger.error(
                f"Current worker version {horde_worker_regen.__version__} has a required update to "
                f"{version_meta.required_min_version}. We are past the date specified by the developers to update to "
                f"{version_meta.required_min_version_update_date}. Please update to the required version "
                "by running `git pull` and `update-runtime` (or the appropriate `pip install` "
                "if you're using your own venv.)",
            )
            logger.error(reason_for_update_str)

            input("Press Enter to continue...")
            exit(1)

    if not semver.compare(horde_worker_regen.__version__, version_meta.recommended_version) >= 0:
        logger.warning(
            f"Current worker version {horde_worker_regen.__version__} is not the recommended version. "
            f"Please consider updating to {version_meta.recommended_version}.",
        )
        os.environ["AIWORKER_NOT_RECOMMENDED_VERSION"] = "1"

    if version_meta.beta_version_info:
        current_version_semver = semver.VersionInfo.parse(horde_worker_regen.__version__)
        current_version_simple = (
            f"{current_version_semver.major}.{current_version_semver.minor}.{current_version_semver.patch}"
        )

        if current_version_simple in version_meta.beta_version_info:
            beta_info = version_meta.beta_version_info[current_version_simple]

            already_set_branch = os.getenv("HORDE_MODEL_REFERENCE_GITHUB_BRANCH")
            if already_set_branch is None and not time.strftime("%Y-%m-%d") > beta_info.beta_expiry_date:
                logger.info(
                    f"Current worker version {horde_worker_regen.__version__} is a beta version. "
                    f"Using the model reference branch {beta_info.horde_model_reference_branch}.",
                )
                os.environ["HORDE_MODEL_REFERENCE_GITHUB_BRANCH"] = beta_info.horde_model_reference_branch


if __name__ == "__main__":
    load_env_vars()
    logger.info("Environment variables loaded.")
