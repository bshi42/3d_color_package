# Python

Never use from __future__ import annotations. Assume python 3.13+
Never use try-imports. Assuminga module is importable if it is used.
Don't duplicate functionality.
Use click instead of argparse.


# Architecture
Don't duplicate functionality. Functions which are general purpose and reusable should be put in a utils module.