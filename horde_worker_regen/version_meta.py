import json

from pydantic import BaseModel

from horde_worker_regen.consts import VERSION_META_REMOTE_URL


class RequiredVersionInfo(BaseModel):
    reason_for_update: str


class BetaVersionInfo(BaseModel):
    horde_model_reference_branch: str
    beta_expiry_date: str


class VersionMeta(BaseModel):
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
