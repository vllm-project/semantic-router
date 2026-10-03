"""First-party models the runtime serves out of the box, pinned by revision and identity.

A new model revision is a new entry; a built-in model is never resolved
through a moving branch. Each family keeps its table in ``registry/tables``.
Golden requests gate readiness; their reference answers per device class live
in ``golden_answers*.json``, recorded with ``tools/golden_answers.py`` for the
pinned revision. ``kernel_choices.json`` holds, per device class, the
autotuned kernel configurations the released runtime ran with
(``tools/kernel_choices.py``); the runtime pins them so every process
computes the released numerics.
"""

from __future__ import annotations

from .tables import decision1, decision2, omni, vela1, vela2
from .tables.common import ORG, BuiltinModel, with_recorded

__all__ = [
    "DECISION2_MODELS",
    "ORG",
    "BuiltinModel",
    "all_models",
    "by_identity",
    "lookup",
    "with_recorded",
]

DECISION2_MODELS = decision2.MODELS
MODELS: tuple[BuiltinModel, ...] = (
    *decision2.MODELS,
    *decision1.MODELS,
    *vela1.MODELS,
    *vela2.MODELS,
    *omni.MODELS,
)

_BY_REPO = {model.repo_id.lower(): model for model in MODELS}
_BY_NAME = {model.repo_id.split("/", 1)[1].lower(): model for model in MODELS}


def lookup(model: str) -> BuiltinModel | None:
    """A built-in model by repository ID or bare model name (case-insensitive)."""
    key = model.strip().lower()
    return _BY_REPO.get(key) or _BY_NAME.get(key)


def by_identity(model_sha256: str) -> BuiltinModel | None:
    for model in MODELS:
        if model.model_sha256 == model_sha256:
            return model
    return None


def all_models(family: str | None = None) -> tuple[BuiltinModel, ...]:
    if family is None:
        return MODELS
    return tuple(model for model in MODELS if model.family == family)
