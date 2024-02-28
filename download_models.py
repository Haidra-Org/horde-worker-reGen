import argparse

from horde_worker_regen.download_models import download_all_models


def main() -> None:
    download_all_models()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all models specified in the config file.")
    parser.add_argument(
        "--purge-unused-loras",
        action="store_true",
        help="Purge unused LORAs from the cache",
    )

    args = parser.parse_args()

    download_all_models(purge_unused_loras=args.purge_unused_loras)
