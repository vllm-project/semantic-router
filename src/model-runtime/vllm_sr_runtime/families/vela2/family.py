"""The Vela 2.0 model family (Phase 3).

``vela2-unified`` (0.3B: the Vela 307M ModernBERT encoder with eight marker
tokens) and ``vela2-decoder`` (4B, 9B: a Qwen3.5 backbone used as an encoder
with one block per question) answer Choice, Noul, Score, Set and Span
questions over typed parts on ``/v1/decisions``. The family reimplements the
packages' engine (``vela2_inference.py``); it never imports it.
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


MODEL_TYPES = ("vela2-unified", "vela2-decoder")


class Vela2Family(ModelFamily):
    name = "vela2"
    surfaces = frozenset({"decisions"})

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {
            "surfaces": sorted(cls.surfaces),
            "formats": ["vela2-unified/1", "vela2-decoder/1"],
        }

    def detect(self, package: PackageRef) -> bool:
        config = _config(package.root)
        return (
            config.get("model_type") in MODEL_TYPES
            and config.get("format_version") == 1
        )

    def verify(self, package: PackageRef) -> VerifiedPackage:
        raise PackageError(f"the {self.name} family cannot load packages yet")

    def describe(self, package: VerifiedPackage) -> ModelSpec:
        raise PackageError(f"the {self.name} family cannot load packages yet")

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> LoadedModel:
        raise PackageError(f"the {self.name} family cannot load packages yet")
