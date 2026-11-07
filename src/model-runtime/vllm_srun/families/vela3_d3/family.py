"""Vela 3.0 d3 sizes (``vela3/3``): vela3-lite, -mini, -flash and -turbo.

Each package runs a pinned Decision 3.0 package unchanged and adds gated LoRA
adapters over its text model and span heads over its final-norm states. The
family verifies the package, fetches and verifies the base with the
``decision3`` family, and folds the base's identity into its own. Where the
base has recorded kernel choices for a device class, the package runs them.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ...errors import PackageError
from ...heads.span import SpanHead
from ...plugins.base import (
    DeviceInfo,
    EngineModel,
    GatedAdapterSpec,
    LoadedModel,
    ModelFamily,
    ModelSpec,
    PackageRef,
    VerifiedPackage,
)
from ...registry import builtin
from ...registry.resolve import download
from ..decision3.family import Decision3Family
from ..decision3.package import is_state_once
from ..vela3_shared import PRESETS
from . import package as pkg
from .gates import Gates
from .model import GatedDecisions, TurboDecisions, Vela3D3Model
from .span_policy import Policy
from .span_read import SpanReader
from .span_render import Renderer
from .spans import Spans

# The one way a state-once base answers its gated rows again (``decisions.attention``): vela3-turbo's.
NONCAUSAL_REPLAYS = "noncausal"


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
        if is_state_once(base_package.details["package"].decision_config):
            replays = (details.vela3.get("decisions") or {}).get("attention")
            if replays != NONCAUSAL_REPLAYS:
                raise PackageError(
                    f"a state-once base needs decisions.attention {NONCAUSAL_REPLAYS!r}; "
                    f"the package declares {replays!r}"
                )
        model_sha256 = pkg.identity(details, base_package.model_sha256)
        if pinned is not None and model_sha256 != pinned.model_sha256:
            raise PackageError(
                f"{package.repo_id}@{package.revision} differs from the built-in identity"
            )
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
            details={"package": details, "base": base_package},
        )

    def kernel_choices(
        self, package: VerifiedPackage, device: DeviceInfo
    ) -> dict[str, Any]:
        """The base model's recorded choices: every pass runs its kernels at its shapes.

        A base pinned at another revision than its built-in entry takes that entry's choices.
        """
        base: VerifiedPackage = package.details["base"]
        recorded = builtin.kernel_choices(
            base.model_sha256, device.accelerator, device.arch
        )
        known = builtin.lookup(base.ref.repo_id or "")
        if recorded or known is None or not device.arch:
            return recorded
        choices: dict[str, Any] = known.kernel_choices.get(
            f"{device.accelerator}:{device.arch}", {}
        )
        return choices

    def describe(self, package: VerifiedPackage) -> ModelSpec:
        """The base's own spec with every text adapter as a gated adapter on its Linears."""
        details: pkg.Vela3D3Package = package.details["package"]
        spec = Decision3Family(self.options).describe(package.details["base"])
        gated = {
            name: GatedAdapterSpec(
                details.root / adapter.file, adapter.rank, adapter.alpha
            )
            for name, adapter in details.adapters.items()
        }
        return replace(
            spec,
            name=package.model_name,
            backbone=replace(spec.backbone, gated=gated),
        )

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> LoadedModel[Any, Any]:
        details: pkg.Vela3D3Package = package.details["package"]
        base = package.details["base"]
        model = Decision3Family(self.options).load_as(
            base, spec, engine_model, _model_class(base)
        )
        heads = {
            route: _head(details.root / file, engine_model.device)
            for route, file in details.span.heads.items()
        }
        model.attach(
            Gates.from_config(
                details.vela3, details.router_questions, details.adapters
            ),
            Spans(
                policy=Policy(details.span, _names_only),
                renderer=Renderer(model.tokenizer, model.codes),
                reader=SpanReader(engine_model, heads),
            ),
        )
        model.info = replace(
            model.info,
            id=package.model_name,
            family=self.name,
            repo=package.ref.repo_id,
            revision=package.ref.revision,
            model_sha256=package.model_sha256,
            manifest_sha256=package.manifest_sha256,
            question_types=(*model.info.question_types, "span"),
            presets=tuple(name for name in PRESETS if name in details.span.schemas),
        )
        return model


def _head(path: Path, device: torch.device) -> SpanHead:
    """A span head (``SpanHeadV2``) at the projection and slot count its file holds, FP32 on ``device``."""
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    hidden = tensors["wn.weight"].shape[0]
    head = SpanHead(
        hidden,
        projection=tensors["K.weight"].shape[0],
        slots=tensors["slot.weight"].shape[0],
    )
    head.load_state_dict(tensors, strict=True)
    return head.to(device).float().eval()


def _names_only(names: list[str], only: set[str]) -> dict[str, str]:
    """No definition writer yet: labels without a description run by name.

    The packages' writers (vela3-mini's base, vela3-flash's grafted head) take
    this place in #5703.
    """
    return {}


def _model_class(base: VerifiedPackage) -> type[Vela3D3Model]:
    if is_state_once(base.details["package"].decision_config):
        return TurboDecisions
    return GatedDecisions
