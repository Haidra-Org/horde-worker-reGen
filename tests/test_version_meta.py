import json

from horde_worker_regen.version_meta import VersionMeta


def test_version_meta_local() -> None:
    with open("horde_worker_regen/_version_meta.json") as f:
        data = json.load(f)
        VersionMeta(**data)
