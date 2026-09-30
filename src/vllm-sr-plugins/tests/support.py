"""Import paths for the plugin tests: the plugin itself and the Decision 2.0 sources."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
DECISION2 = PLUGIN_ROOT.parents[1] / "src" / "training" / "decision2"

for path in (PLUGIN_ROOT, DECISION2):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None
