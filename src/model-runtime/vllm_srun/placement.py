"""Choose an accelerator and device for a model under a memory budget."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import PlacementError, UnsupportedDeviceError
from .plugins import registry
from .plugins.base import Accelerator, DeviceInfo, ModelSpec

DEVICE = re.compile(r"(?P<kind>[a-z][a-z0-9_]*)(?::(?P<index>\d+))?\Z")
BYTES_PER_PARAMETER = 4
ACTIVATION_RESERVE = 2 << 30


@dataclass(frozen=True)
class Placement:
    accelerator: Accelerator
    device: DeviceInfo
    estimated_bytes: int


def parse_device(value: str) -> tuple[str, int | None]:
    """``auto``, or a registered accelerator's name with an optional ``:N`` device index."""
    match = DEVICE.fullmatch(value.strip().lower())
    names = registry.names("accelerators")
    if not match or (match["kind"] != "auto" and match["kind"] not in names):
        raise PlacementError(
            f"--device must be auto or one of {', '.join(names)}, optionally with :N; got {value!r}"
        )
    index = match["index"]
    return match["kind"], int(index) if index is not None else None


def auto_order() -> list[str]:
    """The accelerators ``--device auto`` tries, by their ``auto_priority``."""
    ranked = []
    for name in registry.names("accelerators"):
        priority = registry.plugin("accelerators", name).load().auto_priority
        if priority is not None:
            ranked.append((priority, name))
    return [name for _, name in sorted(ranked)]


def device_kind(device: str) -> str:
    """The accelerator a ``--device`` value lands on; ``auto`` takes the first available in ``auto_order``."""
    kind, _ = parse_device(device)
    if kind != "auto":
        return kind
    for name in auto_order():
        if registry.instantiate("accelerators", name).available():
            return name
    return "cpu"


def check_device(model: str, device: str) -> None:
    """Refuse a ``--device`` this host has no such device for, before the model is resolved."""
    kind, index = parse_device(device)
    if kind == "auto":
        return
    accelerator = registry.instantiate("accelerators", kind)
    if not accelerator.available():
        reason = f"{kind}: not available on this host"
    elif index is not None and all(d.index != index for d in accelerator.devices()):
        reason = f"{kind}: no device {index}"
    else:
        return
    raise PlacementError(f"no device can serve {model}: {reason}")


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
    candidates = auto_order() if kind == "auto" else [kind]
    reasons = []
    unsupported = []
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
        required = spec.requires.get(name, ())
        for info in devices:
            if required:
                offered = accelerator.capabilities(info)
                lacking = [need for need in required if not offered.get(need)]
                if lacking:
                    unsupported += [accelerator.lacks(info, need) for need in lacking]
                    continue
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
    message = f"no device can serve {spec.name}: " + "; ".join(reasons + unsupported)
    if unsupported and not reasons:
        raise UnsupportedDeviceError(message)
    raise PlacementError(message)
