"""Make the recipe_lab package importable regardless of the pytest rootdir.

Repo-level targets (``make test-calibration``) invoke pytest from the
repository root; without this bootstrap the ``recipe_lab`` package next to
this file is not on ``sys.path``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
