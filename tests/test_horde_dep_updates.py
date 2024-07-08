from pathlib import Path

HORDE_BRIDGE_SCRIPT = Path(__file__).parent.parent / "horde-bridge.cmd"


def test_horde_bridge_updating(horde_dependency_versions: list[tuple[str, str]]) -> None:
    """Check that the versions of horde deps. in horde-bridge.cmd match the versions in requirements.txt."""
    haidra_dep_string = "horde"
    haidra_deps = [(dep, version) for dep, version in horde_dependency_versions if haidra_dep_string in dep]

    script_lines = HORDE_BRIDGE_SCRIPT.read_text().split("\n")

    found_line = False
    for line in script_lines:
        if "python -s -m pip install" in line:
            found_line = True
            assert "-U" in line, "No -U flag found in pip install command"
            for dep, version in haidra_deps:
                assert f"{dep}~={version}" in line, f"Dependency {dep} not found in pip install command"

    assert found_line


def test_different_requirements_files_match(
    horde_dependency_versions: list[tuple[str, str]],
    rocm_horde_dependency_versions: list[tuple[str, str]],
) -> None:
    """Check that the versions of horde deps. in the main and rocm requirements files match."""
    main_deps = dict(horde_dependency_versions)
    rocm_deps = dict(rocm_horde_dependency_versions)

    for dep in main_deps:
        assert dep in rocm_deps, f"Dependency {dep} not found in rocm requirements file"
        assert (
            main_deps[dep] == rocm_deps[dep]
        ), f"Dependency {dep} has different versions in main and rocm requirements files"

    for dep in rocm_deps:
        assert dep in main_deps, f"Dependency {dep} not found in main requirements file"
        assert (
            rocm_deps[dep] == main_deps[dep]
        ), f"Dependency {dep} has different versions in main and rocm requirements files"
