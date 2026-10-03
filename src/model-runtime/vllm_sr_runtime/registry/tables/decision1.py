"""Decision 1.0 (Phase 2): Kai, Lex and Route on the Vela encoder; Eos, Sol, Nox and Lux on Qwen3.5.

Entries pin a revision, the SHA-256 of every file the family loads, the
expected identity and parameter count; references live in
``registry/golden_answers_decision1.json``.
"""

from __future__ import annotations

from .common import BuiltinModel, with_recorded

MODELS: tuple[BuiltinModel, ...] = ()
MODELS = with_recorded(
    MODELS, "golden_answers_decision1.json", "answers", "golden_answers"
)
