"""
tests/conftest.py
=================
Pytest configuration ensuring the project directory is on sys.path,
allowing tests to be executed from either the project directory or workspace root.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
