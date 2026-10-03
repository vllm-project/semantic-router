"""Vela 1.0 Omni prepared bundles (Phase 3): Nano and Mini.

Entries pin a revision, the SHA-256 of every file the family loads, the
expected identity and parameter count; references live in
``registry/golden_answers_omni.json``.
"""

from __future__ import annotations

from .common import BuiltinModel, with_recorded

MODELS: tuple[BuiltinModel, ...] = ()
MODELS = with_recorded(MODELS, "golden_answers_omni.json", "answers", "golden_answers")
