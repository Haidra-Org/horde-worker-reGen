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


def check_dependency_versions(
    main_deps: dict[str, str],
    other_deps: dict[str, str],
    other_name: str,
) -> None:
    """Check that the main requirements file is consistent with the other requirements file.

    Args:
        main_deps (dict[str, str]): The versions of the dependencies in the main requirements file.
        other_deps (dict[str, str]): The versions of the dependencies in the other requirements file.
        other_name (str): The name of the other requirements file.

    Raises:
        AssertionError: If the versions of the dependencies are inconsistent.
    """
    for dep in main_deps:
        if dep == "torch":
            logger.warning(
                f"Skipping torch version check (main: {main_deps[dep]}, {other_name}: {other_deps[dep]})",
            )
            continue

        assert dep in other_deps, f"Dependency {dep} not found in {other_name} requirements file"
        assert (
            main_deps[dep] == other_deps[dep]
        ), f"Dependency {dep} has different versions in main and {other_name} requirements files"

    for dep in other_deps:
        if dep == "torch":
            logger.warning(
                f"Skipping torch version check (main: {main_deps[dep]}, {other_name}: {other_deps[dep]})",
            )
            continue

        assert dep in main_deps, f"Dependency {dep} not found in main requirements file"
        assert (
            other_deps[dep] == main_deps[dep]
        ), f"Dependency {dep} has different versions in main and {other_name} requirements files"


def test_different_requirements_files_match(
    horde_dependency_versions: dict[str, str],
    rocm_horde_dependency_versions: list[tuple[str, str]],
    directml_horde_dependency_versions: list[tuple[str, str]],
) -> None:
    """Check that the versions of horde deps. in the all of the various requirements files are consistent."""
    rocm_deps = dict(rocm_horde_dependency_versions)
    directml_deps = dict(directml_horde_dependency_versions)

    check_dependency_versions(horde_dependency_versions, rocm_deps, "rocm")
    check_dependency_versions(horde_dependency_versions, directml_deps, "directml")
