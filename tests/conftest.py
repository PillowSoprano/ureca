"""Test configuration for ensuring project modules are importable.

Pytest's console entry point can set ``sys.path[0]`` to the interpreter's
binary directory rather than the current working directory. Adding the
repository root explicitly keeps absolute imports (e.g., ``mbr_actions``)
working regardless of how pytest is invoked.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
