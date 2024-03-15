import datetime
import json
import os
import time

import semver
from loguru import logger
from pydantic import BaseModel

import horde_worker_regen
from horde_worker_regen.consts import VERSION_META_REMOTE_URL


class RequiredVersionInfo(BaseModel):
    reason_for_update: str


class BetaVersionInfo(BaseModel):
    horde_model_reference_branch: str
    beta_expiry_date: str


class VersionMeta(BaseModel):
    recommended_version: str
    required_min_version: str
    required_min_version_update_date: str
    beta_version_info: dict[str, BetaVersionInfo]
    required_min_version_info: dict[str, RequiredVersionInfo]


def get_local_version_meta() -> VersionMeta:
    with open("horde_worker_regen/_version_meta.json") as f:
        data = json.load(f)
        return VersionMeta(**data)


def get_remote_version_meta() -> VersionMeta:
    import requests

    data = requests.get(VERSION_META_REMOTE_URL).json()
    return VersionMeta(**data)


def do_version_check() -> None:
    """Check if the current worker version satisfies the required and recommended versions.

    Note that this function sets environment variables to indicate if the worker version is not the required or
    recommended version. It can also change the github branch used for the model reference if the current version is a
    beta version.
    """
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
