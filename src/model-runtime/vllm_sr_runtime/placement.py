"""Choose an accelerator and device for a model under a memory budget."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import PlacementError
from .plugins import registry
from .plugins.base import Accelerator, DeviceInfo, ModelSpec

DEVICE = re.compile(r"(?P<kind>auto|cpu|cuda|rocm|xpu|mps)(?::(?P<index>\d+))?\Z")
AUTO_ORDER = ("rocm", "cuda")  # validated GPUs; xpu and mps only when named
BYTES_PER_PARAMETER = 4
ACTIVATION_RESERVE = 2 << 30


@dataclass(frozen=True)
class Placement:
    accelerator: Accelerator
    device: DeviceInfo
    estimated_bytes: int


def parse_device(value: str) -> tuple[str, int | None]:
    match = DEVICE.fullmatch(value.strip().lower())
    if not match:
        raise PlacementError(
            f"--device must be auto, cpu, cuda[:N], rocm[:N], xpu[:N] or mps; got {value!r}"
        )
    index = match["index"]
    return match["kind"], int(index) if index is not None else None


def estimate_bytes(parameters: int) -> int:
    """Weights in FP32 (BF16-resident Linear weights only lower it) plus an activation reserve."""
    return parameters * BYTES_PER_PARAMETER + ACTIVATION_RESERVE


def place(
    spec: ModelSpec,
    device: str,
    parameters: int,
    memory_budget_gib: float | None = None,
) -> Placement:
    kind, index = parse_device(device)
    needed = estimate_bytes(parameters)
    budget = int(memory_budget_gib * (1 << 30)) if memory_budget_gib else None
    if budget is not None and needed > budget:
        raise PlacementError(
            f"{spec.name} needs about {needed / (1 << 30):.1f} GiB, above --memory-budget {memory_budget_gib} GiB"
        )
    candidates = (*AUTO_ORDER, "cpu") if kind == "auto" else (kind,)
    reasons = []
    for name in candidates:
        try:
            accelerator = registry.instantiate("accelerators", name)
        except KeyError as exc:
            reasons.append(str(exc))
            continue
        if not accelerator.available():
            reasons.append(f"{name}: not available on this host")
            continue
        devices = accelerator.devices()
        if index is not None:
            devices = [d for d in devices if d.index == index]
        for info in devices:
            if name != "cpu" and spec.dtype.autocast == "bfloat16" and not info.bf16:
                reasons.append(f"{info.label}: no BF16 support")
                continue
            if (
                name != "cpu"
                and info.free_memory is not None
                and info.free_memory < needed
            ):
                reasons.append(
                    f"{info.label}: {info.free_memory / (1 << 30):.1f} GiB free"
                )
                continue
            return Placement(
                accelerator=accelerator, device=info, estimated_bytes=needed
            )
        if not devices:
            reasons.append(
                f"{name}: no device {index}"
                if index is not None
                else f"{name}: no devices"
            )
    raise PlacementError(f"no device can serve {spec.name}: " + "; ".join(reasons))
