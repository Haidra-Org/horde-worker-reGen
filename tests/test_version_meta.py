from horde_worker_regen.version_meta import get_local_version_meta, get_remote_version_meta


def test_get_local_version_meta() -> None:
    """Test that the local copy of the version meta can be retrieved."""
    get_local_version_meta()


def test_version_meta_remote() -> None:
    """Test that the remote version meta can be retrieved."""
    get_remote_version_meta()
