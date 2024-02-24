import json

from pydantic import BaseModel


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


if __name__ == "__main__":
    with open("horde_worker_regen/_version_meta.json") as f:
        data = json.load(f)
        version_meta = VersionMeta(**data)
        print(version_meta)
