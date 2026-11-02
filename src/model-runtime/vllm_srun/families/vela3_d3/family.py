"""Vela 3.0 d3 sizes (``vela3/3``): vela3-lite, -mini, -flash and -turbo.

Each package runs a pinned Decision 3.0 package unchanged and adds gated LoRA
adapters over its text model and span heads over its final-norm states. The
family verifies the package, fetches and verifies the base with the
``decision3`` family, and folds the base's identity into its own.
"""

from __future__ import annotations

from dataclasses import replace
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
from ...registry import builtin
from ...registry.resolve import download
from ..decision3.family import Decision3Family
from . import package as pkg


class Vela3D3Family(ModelFamily):
    name = "vela3_d3"
    surfaces = frozenset({"decisions"})
    builtin_table = "vllm_srun.registry.tables.vela3_d3"
    fixture_writer = "vllm_srun.testing.vela3_d3"

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {
            "surfaces": sorted(cls.surfaces),
            "formats": [pkg.FORMAT],
            "question_types": ["choice", "noul", "score", "span"],
        }

    def detect(self, package: PackageRef) -> bool:
        return pkg.member_of(package.root)

    def _base_root(self, base: pkg.Base) -> Path:
        if self.options.base_path is not None:
            return Path(self.options.base_path).resolve()
        return download(
            base.repo_id,
            base.revision,
            cache_dir=self.options.cache_dir,
            offline=self.options.offline,
        )

    def verify(self, package: PackageRef) -> VerifiedPackage:
        known = builtin.lookup(package.repo_id) if package.repo_id else None
        pinned = (
            known if known is not None and known.revision == package.revision else None
        )
        details = pkg.verify(package.root, dict(pinned.files) if pinned else None)
        base = details.base
        if pinned is not None and pinned.base != (base.repo_id, base.revision):
            raise PackageError(
                f"{package.repo_id}@{package.revision} pins base "
                f"{base.repo_id}@{base.revision}, not the built-in {pinned.base}"
            )
        base_package = Decision3Family(self.options).verify(
            PackageRef(self._base_root(base), base.repo_id, base.revision)
        )
        model_sha256 = pkg.identity(details, base_package.model_sha256)
        if pinned is not None and model_sha256 != pinned.model_sha256:
            raise PackageError(
                f"{package.repo_id}@{package.revision} differs from the built-in identity"
            )
        details = replace(details, base_package=base_package, model_sha256=model_sha256)
        name = (
            package.repo_id
            or (details.vela3.get("size") or {}).get("name")
            or package.root.name
        ).split("/")[-1]
        return VerifiedPackage(
            ref=package,
            family=self.name,
            model_name=name,
            manifest={},
            manifest_sha256="",
            model_sha256=model_sha256,
            max_input_tokens=base_package.max_input_tokens,
            licence=base_package.licence,
            loaded_parameters=pinned.loaded_parameters if pinned else None,
            details={"package": details},
        )

    def describe(self, package: VerifiedPackage) -> ModelSpec:
        raise PackageError("vela3/3 packages are not servable yet")

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> LoadedModel[Any, Any]:
        raise PackageError("vela3/3 packages are not servable yet")
