"""Vela 2.0 (Phase 3, private preview): 0.3B, 4B and 9B.

Entries pin a revision, the SHA-256 of every file the family loads, the
expected identity and parameter count; references live in
``registry/golden_answers_vela2.json``.
"""

from __future__ import annotations

from .common import BuiltinModel, with_recorded

MODELS: tuple[BuiltinModel, ...] = ()
MODELS = with_recorded(MODELS, "golden_answers_vela2.json", "answers", "golden_answers")
