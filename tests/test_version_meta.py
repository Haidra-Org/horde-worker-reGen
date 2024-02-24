from horde_worker_regen.version_meta import get_local_version_meta, get_remote_version_meta


def test_get_local_version_meta() -> None:
    get_local_version_meta()


def test_version_meta_remote() -> None:
    get_remote_version_meta()
