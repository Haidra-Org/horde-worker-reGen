"""Contains the functions to load the environment variables from the config file."""
import os
import pathlib

from dotenv import load_dotenv
from ruamel.yaml import YAML

load_dotenv()


def load_env_vars() -> None:  # FIXME: there is a dynamic way to do this
    """Load the environment variables from the config file."""
    yaml = YAML()

    if not pathlib.Path("bridgeData.yaml").exists():
        raise FileNotFoundError("bridgeData.yaml not found")

    with open("bridgeData.yaml", encoding="utf-8") as f:
        config = yaml.load(f)

    if "cache_home" in config:
        if os.getenv("AIWORKER_CACHE_HOME") is None:
            os.environ["AIWORKER_CACHE_HOME"] = config["cache_home"]
        else:
            print(
                "AIWORKER_CACHE_HOME environment variable already set. "
                "This will override the value for `cache_home` in the config file.",
            )

    if "max_lora_cache_size" in config:
        if os.getenv("AIWORKER_LORA_CACHE_SIZE") is None:
            try:
                int(config["max_lora_cache_size"])
            except ValueError as e:
                raise ValueError(
                    "max_lora_cache_size must be an integer, but is not.",
                ) from e
            os.environ["AIWORKER_LORA_CACHE_SIZE"] = str(config["max_lora_cache_size"])
        else:
            print(
                "AIWORKER_LORA_CACHE_SIZE environment variable already set. "
                "This will override the value for `max_lora_cache_size` in the config file.",
            )
