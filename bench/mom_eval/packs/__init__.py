"""EvaluationPack protocol and registry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bench.mom_eval.common import REPO_ROOT, load_pack_registry, load_yaml


@dataclass(frozen=True)
class RunStep:
    step_id: str
    harness: str
    benchmark_id: str
    metric: str
    limit: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PackResult:
    pack_id: str
    metrics: dict[str, dict[str, Any]]
    artifacts: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class EvaluationPack(Protocol):
    pack_id: str

    def validate(self, manifest: dict[str, Any], entrypoint: str) -> list[str]: ...

    def plan(self, manifest: dict[str, Any], entrypoint: str, run_mode: str) -> list[RunStep]: ...

    def collect(self, raw_dir: Path, steps: list[RunStep]) -> PackResult: ...


def load_pack_manifest(pack_id: str) -> dict[str, Any]:
    registry = load_pack_registry()
    packs = registry.get("packs") or {}
    if pack_id not in packs:
        raise KeyError(f"unknown pack id: {pack_id}")
    manifest_path = REPO_ROOT / str(packs[pack_id]["manifest"])
    return load_yaml(manifest_path)


def get_pack_module_name(pack_id: str) -> str:
    registry = load_pack_registry()
    return str(registry["packs"][pack_id]["module"])


def import_pack(pack_id: str) -> EvaluationPack:
    module_name = get_pack_module_name(pack_id)
    import importlib

    module = importlib.import_module(module_name)
    factory = getattr(module, "create_pack", None)
    if factory is None:
        raise AttributeError(f"{module_name} must export create_pack()")
    pack = factory()
    return pack
