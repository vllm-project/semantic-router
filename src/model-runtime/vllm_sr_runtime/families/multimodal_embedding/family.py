"""Multimodal embeddings (Phase 3): the Vela 1.0 Omni prepared bundles.

A bundle (``tools/models/vela_omni``) holds four verified ONNX graphs (text,
image, CLAP, audio) and the exact processors, described by
``vela_omni_manifest.json``. The family serves text, image and audio
embeddings in one space on ``/v1/embeddings`` with the ``onnxruntime`` engine.
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


BUNDLE_MANIFEST = "vela_omni_manifest.json"


class MultimodalEmbeddingFamily(ModelFamily):
    name = "multimodal_embedding"
    surfaces = frozenset({"embeddings"})

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {"surfaces": sorted(cls.surfaces), "formats": ["vela-omni-bundle"]}

    def detect(self, package: PackageRef) -> bool:
        return (package.root / BUNDLE_MANIFEST).is_file()

    def verify(self, package: PackageRef) -> VerifiedPackage:
        raise PackageError(f"the {self.name} family cannot load packages yet")

    def describe(self, package: VerifiedPackage) -> ModelSpec:
        raise PackageError(f"the {self.name} family cannot load packages yet")

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> LoadedModel:
        raise PackageError(f"the {self.name} family cannot load packages yet")
