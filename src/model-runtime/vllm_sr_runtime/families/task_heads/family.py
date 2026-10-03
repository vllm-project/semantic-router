"""Encoder task heads (Phase 3): Vela 1.0 and compatible HF ModernBERT task models.

Heads: ``sequence`` (softmax distribution), ``scores`` (independent sigmoid
per label, packaged operating point), ``token`` (BIO spans), ``grounded``
(answer spans against a context), ``pooled`` (embeddings with Matryoshka
dimensions and layer exits) and ``relevance`` (reranker exits). The family
serves ``/v1/classify``, ``/v1/embeddings`` and ``/v1/rerank``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...errors import PackageError
from ...plugins.base import (
    EngineModel,
    LoadedModel,
    ModelFamily,
    ModelSpec,
    PackageRef,
    VerifiedPackage,
)


def _config(root: Path, name: str = "config.json") -> dict[str, Any]:
    try:
        value = json.loads((root / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


ENCODER_TYPES = ("modernbert",)
ARCHITECTURES = {
    "ModernBertForSequenceClassification",
    "ModernBertForTokenClassification",
    "ModernBertModel",
}


class TaskHeadsFamily(ModelFamily):
    name = "task_heads"
    surfaces = frozenset({"classify", "embeddings", "rerank"})

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {"surfaces": sorted(cls.surfaces), "formats": ["hf-modernbert"]}

    def detect(self, package: PackageRef) -> bool:
        config = _config(package.root)
        if config.get("model_type") not in ENCODER_TYPES:
            return False
        architectures = set(config.get("architectures") or ())
        return bool(architectures & ARCHITECTURES)

    def verify(self, package: PackageRef) -> VerifiedPackage:
        raise PackageError(f"the {self.name} family cannot load packages yet")

    def describe(self, package: VerifiedPackage) -> ModelSpec:
        raise PackageError(f"the {self.name} family cannot load packages yet")

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> LoadedModel:
        raise PackageError(f"the {self.name} family cannot load packages yet")
