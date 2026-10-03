"""The Decision 1.0 model family (Phase 2).

Decision 1.0 packages carry the root pointer ``{"decision_format":
"vllm-sr-decision", "format_version": 1, "runtime_family": ...}`` with two
runtimes: ``vela-encoder`` (Kai, Lex, Route: the Vela 307M ModernBERT encoder,
choice and score encoders and K-slot decision heads) and
``qwen3.5-decision`` (Eos, Sol, Nox, Lux: a Qwen3.5 backbone and an FP32
decision head). The family reimplements the packages' bundled runtime on the
native engine; it never imports the bundled Python.
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


RUNTIME_FAMILIES = ("vela-encoder", "qwen3.5-decision")


class Decision1Family(ModelFamily):
    name = "decision1"
    surfaces = frozenset({"decisions"})

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {"surfaces": sorted(cls.surfaces), "formats": ["vllm-sr-decision/1"]}

    def detect(self, package: PackageRef) -> bool:
        pointer = _config(package.root)
        return (
            pointer.get("decision_format") == "vllm-sr-decision"
            and pointer.get("format_version") == 1
            and pointer.get("runtime_family") in RUNTIME_FAMILIES
        )

    def verify(self, package: PackageRef) -> VerifiedPackage:
        raise PackageError(f"the {self.name} family cannot load packages yet")

    def describe(self, package: VerifiedPackage) -> ModelSpec:
        raise PackageError(f"the {self.name} family cannot load packages yet")

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> LoadedModel:
        raise PackageError(f"the {self.name} family cannot load packages yet")
