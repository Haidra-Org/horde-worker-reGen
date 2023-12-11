# Contributing to horde_worker_reGen

## Code Quality Tools

* [tox](https://tox.wiki/)
  - Creates virtual environments for CI or local pytest runs.
    - Note that the CI does not current execute calls to the production API by default.
  - Run `tox list` or see `tox.ini` for more info
* [pre-commit](https://pre-commit.com/)
  - Creates virtual environments for formatting and linting tools
  - Run `pre-commit run --all-files` or see `.pre-commit-config.yaml` for more info.
* [black](https://github.com/psf/black)
  - Whitespace formatter
* [ruff](https://github.com/astral-sh/ruff)
  - Linting rules from a wide variety of selectable rule sets
  - See `pyproject.toml` for the rules used.
  - See all rules (but not necessarily used in the project) availible in rust [here](https://beta.ruff.rs/docs/rules/).
* [mypy](https://mypy-lang.org/)
  - Static type safety
  - I recommending using the [mypy daemon](https://mypy.readthedocs.io/en/stable/mypy_daemon.html) instead of periodically running `pre-commit` (or `mypy` directly.).
    - If you are using VSCode, I recommend the `matangover.mypy` extension, which implements this nicely.

## Things to know

  * The `AI_HORDE_DEV_URL` environment variable overrides `AI_HORDE_URL`. This is useful for testing changes locally.
  * pytest files which end in `_api_calls.py` run last, and never run during the CI. It is currently incumbent on individual developers to confirm that these tests run successfully locally. In the future, part of the CI will be to spawn an AI-Horde and worker instances and test it there.
