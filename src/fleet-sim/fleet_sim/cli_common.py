"""Shared helpers for the fleet-sim CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

from .gpu_profiles.profiles import A10G, A100_80GB, H100_80GB, GpuProfile

GPU_REGISTRY = {"a100": A100_80GB, "h100": H100_80GB, "a10g": A10G}


def load_cdf(path: str) -> list:
    raw = json.loads(Path(path).read_text())
    cdf = raw["cdf"] if isinstance(raw, dict) else raw
    return [(int(threshold), float(frac)) for threshold, frac in cdf]


def resolve_gpu(name: str | None, default: GpuProfile) -> GpuProfile:
    return GPU_REGISTRY.get(name or "", default)


def write_json_output(path: str | None, payload: object) -> None:
    if not path:
        return
    Path(path).write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to {path}")
