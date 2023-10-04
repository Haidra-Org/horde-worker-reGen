import os
import pathlib

from ruamel.yaml import YAML


def load_env_vars() -> None:
    yaml = YAML()

    if not pathlib.Path("bridgeData.yaml").exists():
        raise FileNotFoundError("bridgeData.yaml not found")

    with open("bridgeData.yaml") as f:
        config = yaml.load(f)

    if "cache_home" in config:
        os.environ["AIWORKER_CACHE_HOME"] = config["cache_home"]
