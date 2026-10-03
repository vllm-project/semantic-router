"""Vela 1.0 encoder task models (Phase 3): Domain, Guard, Safety, Shield, FactCheck, Feedback, Modality, Hazard, PII, Halu, Embedding and Reranker at the revisions the router pins.

Entries pin a revision, the SHA-256 of every file the family loads, the
expected identity and parameter count; references live in
``registry/golden_answers_vela1.json``.
"""

from __future__ import annotations

from .common import BuiltinModel, with_recorded

MODELS: tuple[BuiltinModel, ...] = ()
MODELS = with_recorded(MODELS, "golden_answers_vela1.json", "answers", "golden_answers")
