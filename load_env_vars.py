"""Contains the functions to load the environment variables from the config file."""
import os
import pathlib

from dotenv import load_dotenv
from ruamel.yaml import YAML

load_dotenv()


def load_env_vars() -> None:  # FIXME: there is a dynamic way to do this
    """Load the environment variables from the config file."""
    yaml = YAML()
    config_file = "bridgeData.yaml"
    template_file = "bridgeData_template.yaml"

    if not pathlib.Path(config_file).exists():
        if pathlib.Path(template_file).exists():
            raise FileNotFoundError(f"{template_file} found. Please set variables and rename it to {config_file}.")
        raise FileNotFoundError(f"{config_file} not found")

    with open(config_file, encoding="utf-8") as f:
        config = yaml.load(f)

    if "cache_home" in config:
        if os.getenv("AIWORKER_CACHE_HOME") is None:
            os.environ["AIWORKER_CACHE_HOME"] = config["cache_home"]
        else:
            print(
                "AIWORKER_CACHE_HOME environment variable already set. "
                "This will override the value for `cache_home` in the config file.",
            )
