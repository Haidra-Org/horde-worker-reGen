"""Contains the functions to load the environment variables from the config file."""

import os
import pathlib

from dotenv import load_dotenv
from loguru import logger
from ruamel.yaml import YAML

load_dotenv()


def load_env_vars_from_config() -> None:  # FIXME: there is a dynamic way to do this
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

    # See data_model.py's `def load_env_vars(self) -> None:`
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
    if "civitai_api_token" in config:
        if os.getenv("CIVIT_API_TOKEN") is None:
            os.environ["CIVIT_API_TOKEN"] = config["civitai_api_token"]
        else:
            print(
                "CIVIT_API_TOKEN environment variable already set. "
                "This will override the value for `civitai_api_token` in the config file.",
            )

    if "horde_url" in config:
        known_ai_horde_urls = [
            "stablehorde.net",
            "aihorde.net",
        ]

        custom_horde_url = config["horde_url"]
        if custom_horde_url and any(url in custom_horde_url for url in known_ai_horde_urls):
            logger.debug("Using default AI Horde URL.")
        else:
            logger.warning(
                f"Using custom AI Horde URL `{custom_horde_url}`. Make sure this is correct and ends in `/api/`.",
            )
            os.environ["AI_HORDE_URL"] = custom_horde_url


if __name__ == "__main__":
    load_env_vars_from_config()
    logger.info("Environment variables loaded.")
