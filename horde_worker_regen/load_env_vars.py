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

    # Users on windows occasionally use backslashes in their paths, which causes issues on loading.
    # We're going to load the file as text and print the lines with backslashes to the user, and instruct them to
    # replace them with forward slashes.

    with open(config_file, encoding="utf-8") as f:
        lines = f.readlines()
        found_backslashes = False
        for line in lines:
            if "\\" in line:
                print(f"Backslashes found in the following line:\n{line}")
                found_backslashes = True

                print(
                    "Please replace backslashes with forward slashes in the config file, "
                    "as backslashes are not supported.",
                )

                corrected_line = line.replace("\\", "/")
                print(f"Corrected line:\n{corrected_line}")

    if found_backslashes:
        import sys

        sys.exit(1)

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

    if "load_large_models" in config and os.getenv("AI_HORDE_MODEL_META_LARGE_MODELS") is None:
        config_value = config["load_large_models"]
        if config_value is True:
            os.environ["AI_HORDE_MODEL_META_LARGE_MODELS"] = "1"

    if "limited_console_messages" in config and os.getenv("AIWORKER_LIMITED_CONSOLE_MESSAGES") is None:
        config_value = config["limited_console_messages"]
        if config_value is True:
            os.environ["AIWORKER_LIMITED_CONSOLE_MESSAGES"] = "1"


if __name__ == "__main__":
    load_env_vars_from_config()
    logger.info("Environment variables loaded.")
