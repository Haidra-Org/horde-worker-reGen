from pathlib import Path

import yaml

PRECOMMIT_FILE_PATH = Path(__file__).parent.parent / ".pre-commit-config.yaml"
REQUIREMENTS_FILE_PATH = Path(__file__).parent.parent / "requirements.txt"


def test_pre_commit_dep_versions() -> None:
    # Make sure hordelib and horde_sdk version pins match
    with open(PRECOMMIT_FILE_PATH) as f:
        precommit_config = yaml.safe_load(f)

    NUM_TRACKED_REPOS = 3
    horde_sdk_version = None
    hordelib_version = None
    horde_model_reference_version = None

    for repo in precommit_config["repos"]:
        if "mypy" in repo["repo"]:
            # Check additional_dependencies for horde_sdk, hordelib or horde_model_reference
            for dep in repo["hooks"][0]["additional_dependencies"]:
                if dep.startswith("horde_sdk"):
                    horde_sdk_version = dep.split("==")[1]

                if dep.startswith("hordelib"):
                    hordelib_version = dep.split("==")[1]

                if dep.startswith("horde_model_reference"):
                    horde_model_reference_version = dep.split("==")[1]

    assert horde_sdk_version is not None
    assert hordelib_version is not None
    assert horde_model_reference_version is not None

    with open(REQUIREMENTS_FILE_PATH) as f:
        requirements = f.readlines()

    matches = 0
    for req in requirements:
        if req.startswith("horde_sdk"):
            req_version = req.split("~=")[1].strip()
            assert horde_sdk_version == req_version
            matches += 1
        if req.startswith("hordelib"):
            req_version = req.split("~=")[1].strip()
            assert hordelib_version == req_version
            matches += 1
        if req.startswith("horde_model_reference"):
            req_version = req.split("~=")[1].strip()
            assert horde_model_reference_version == req_version
            matches += 1

    assert matches == NUM_TRACKED_REPOS
