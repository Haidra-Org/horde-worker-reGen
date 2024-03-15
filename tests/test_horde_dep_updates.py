from pathlib import Path

HORDE_BRIDGE_SCRIPT = Path(__file__).parent.parent / "horde-bridge.cmd"


def test_horde_bridge_updating(horde_dependency_versions: list[tuple[str, str]]) -> None:
    """Check that the versions of horde deps. in horde-bridge.cmd match the versions in requirements.txt."""
    script_lines = HORDE_BRIDGE_SCRIPT.read_text().split("\n")

    found_line = False
    for line in script_lines:
        if "python -s -m pip install" in line:
            found_line = True
            assert "-U" in line, "No -U flag found in pip install command"
            for dep, version in horde_dependency_versions:
                assert f"{dep}~={version}" in line, f"Dependency {dep} not found in pip install command"

    assert found_line
