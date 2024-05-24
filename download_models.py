import argparse

from horde_worker_regen.download_models import download_all_models
from horde_worker_regen.version_meta import do_version_check

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all models specified in the config file.")
    parser.add_argument(
        "--purge-unused-loras",
        action="store_true",
        help="Purge unused LORAs from the cache",
    )
    parser.add_argument(
        "-e",
        "--load-config-from-env-vars",
        action="store_true",
        default=False,
        help="Load the config only from environment variables. This is useful for running the worker in a container.",
    )

    args = parser.parse_args()

    do_version_check()

    download_all_models(
        purge_unused_loras=args.purge_unused_loras,
        load_config_from_env_vars=args.load_config_from_env_vars,
    )
