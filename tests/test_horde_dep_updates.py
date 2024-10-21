from pathlib import Path

from loguru import logger

HORDE_BRIDGE_SCRIPT = Path(__file__).parent.parent / "horde-bridge.cmd"


def test_horde_bridge_updating(horde_dependency_versions: dict[str, str]) -> None:
    """Check that the versions of horde deps. in horde-bridge.cmd match the versions in requirements.txt."""
    haidra_dep_string = "horde"
    haidra_deps = [(dep, version) for dep, version in horde_dependency_versions.items() if haidra_dep_string in dep]

    script_lines = HORDE_BRIDGE_SCRIPT.read_text().split("\n")

    found_line = False
    for line in script_lines:
        if "python -s -m pip install" in line:
            found_line = True
            assert "-U" in line, "No -U flag found in pip install command"
            for dep, version in haidra_deps:
                assert f"{dep}~={version}" in line, f"Dependency {dep} not found in pip install command"
    assert found_line, "No pip install command found in horde-bridge.cmd"


HORDE_UPDATE_RUNTIME_SCRIPT = Path(__file__).parent.parent / "update-runtime.cmd"


def test_horde_update_runtime_updating(horde_dependency_versions: dict[str, str]) -> None:
    """Check that the versions of horde deps. in update-runtime.cmd match the versions in requirements.txt."""
    torch_dep_string = "torch"
    torch_version = horde_dependency_versions["torch"]

    script_lines = HORDE_UPDATE_RUNTIME_SCRIPT.read_text().split("\n")

    found_line = False
    for line in script_lines:
        if "python -s -m pip install torch==" in line:
            found_line = True
            assert (
                f"{torch_dep_string}=={torch_version}" in line
            ), f"Torch {torch_version} not found in initial torch install command"

    assert found_line, "No initial torch install command found"


def test_different_requirements_files_match(
    horde_dependency_versions: dict[str, str],
    rocm_horde_dependency_versions: list[tuple[str, str]],
) -> None:
    """Check that the versions of horde deps. in the main and rocm requirements files match."""
    rocm_deps = dict(rocm_horde_dependency_versions)

    for dep in horde_dependency_versions:
        if dep == "torch":
            logger.warning(
                f"Skipping torch version check (main: {horde_dependency_versions[dep]}, rocm: {rocm_deps[dep]})"
            )
            continue

        assert dep in rocm_deps, f"Dependency {dep} not found in rocm requirements file"
        assert (
            horde_dependency_versions[dep] == rocm_deps[dep]
        ), f"Dependency {dep} has different versions in main and rocm requirements files"

    for dep in rocm_deps:
        if dep == "torch":
            logger.warning(
                f"Skipping torch version check (main: {horde_dependency_versions[dep]}, rocm: {rocm_deps[dep]})"
            )
            continue

        assert dep in horde_dependency_versions, f"Dependency {dep} not found in main requirements file"
        assert (
            rocm_deps[dep] == horde_dependency_versions[dep]
        ), f"Dependency {dep} has different versions in main and rocm requirements files"
