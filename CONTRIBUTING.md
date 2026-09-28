# Contributing

Please open an issue with the command, software versions, complete traceback,
and a minimal example. Use synthetic data or images you are permitted to share.

For code changes, create a branch, add a regression test for the behavior,
and run `python -m pytest -q`. Keep scientific changes separate from packaging
changes. Document changes to normalization, splitting, augmentation, losses,
checkpoint selection, or metric aggregation because these affect comparability.

Never commit datasets, checkpoints, credentials, personal paths, or submission
correspondence. Submit contributions under this project's MIT license.
