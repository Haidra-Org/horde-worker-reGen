from pathlib import Path

import yaml

PRECOMMIT_FILE_PATH = Path(__file__).parent.parent / ".pre-commit-config.yaml"
REQUIREMENTS_FILE_PATH = Path(__file__).parent.parent / "requirements.txt"


def test_pre_commit_dep_versions(horde_dependency_versions: list[tuple[str, str]]) -> None:
    """Check that the versions of horde deps. in .pre-commit-config.yaml match the versions in requirements.txt.

    Checked dependencies at the time of writing:
    - horde_sdk
    - horde_engine
    - horde_model_reference
    """
    # Make sure horde-engine and horde_sdk version pins match
    with open(PRECOMMIT_FILE_PATH) as f:
        precommit_config = yaml.safe_load(f)

    horde_sdk_version = None
    horde_engine_version = None
    horde_model_reference_version = None

    for repo in precommit_config["repos"]:
        if "mypy" in repo["repo"]:
            # Check additional_dependencies for horde_sdk, horde-engine or horde_model_reference
            for dep in repo["hooks"][0]["additional_dependencies"]:
                if dep.startswith("horde_sdk"):
                    horde_sdk_version = dep.split("==")[1]

                if dep.startswith("horde-engine"):
                    horde_engine_version = dep.split("==")[1]

                if dep.startswith("horde_model_reference"):
                    horde_model_reference_version = dep.split("==")[1]

    assert horde_sdk_version is not None
    assert horde_engine_version is not None
    assert horde_model_reference_version is not None

    matches = 0
    for dep, version in horde_dependency_versions:
        if dep == "horde_sdk" and version == horde_sdk_version:
            matches += 1
        if dep == "horde-engine" and version == horde_engine_version:
            matches += 1
        if dep == "horde_model_reference" and version == horde_model_reference_version:
            matches += 1

    assert matches == len(horde_dependency_versions)
