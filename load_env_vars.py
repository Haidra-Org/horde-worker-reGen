import os
import pathlib

from dotenv import load_dotenv
from ruamel.yaml import YAML

load_dotenv()


def load_env_vars() -> None:
    yaml = YAML()

    if not pathlib.Path("bridgeData.yaml").exists():
        raise FileNotFoundError("bridgeData.yaml not found")

    with open("bridgeData.yaml") as f:
        config = yaml.load(f)

    if "cache_home" in config:
        if os.getenv("AIWORKER_CACHE_HOME") is None:
            os.environ["AIWORKER_CACHE_HOME"] = config["cache_home"]
        else:
            print(
                "AIWORKER_CACHE_HOME environment variable already set. "
                "This will override the value for `cache_home` in the config file.",
            )
