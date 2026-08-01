"""
Make `routingdrift` importable from the test suite without requiring an install.

This is the pattern from the Hitchhiker's Guide to Python ("Structuring Your Project"):
keep the one sys.path manipulation in a single, obvious place so individual test modules
never have to do it, and so `pytest` works on a fresh clone before `pip install -e .`.

Usage, at the top of a test module:

    import context  # noqa: F401
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
