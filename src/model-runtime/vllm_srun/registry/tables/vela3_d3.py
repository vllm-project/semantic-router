"""Vela 3.0 d3 sizes (``vela3/3``): vela3-lite, -mini, -flash and -turbo.

Entries pin a revision, the SHA-256 of every file the family loads (the
adapters, the span heads, the configs, the router questions and the label
definition cache), the expected identity (which covers the base) and the
Decision 3.0 base each package's ``vela3_config.json`` pins (``base``).
"""

from __future__ import annotations

from .common import BuiltinModel

MODELS: tuple[BuiltinModel, ...] = ()
