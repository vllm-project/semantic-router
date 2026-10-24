"""Choose an accelerator and device for a model under a memory budget."""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import dtypes
from .errors import PlacementError, UnsupportedDeviceError
from .plugins import registry
from .plugins.base import Accelerator, DeviceInfo, ModelSpec

DEVICE = re.compile(r"(?P<kind>[a-z][a-z0-9_]*)(?::(?P<index>\d+))?\Z")
DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2}
FP32_BYTES = DTYPE_BYTES["float32"]
ACTIVATION_RESERVE = 2 << 30
# Weights held exactly in a 16-bit dtype get this share of their size again:
# in Engine mode on MI325X the six Decision 3.0 models peaked 1.4 to 1.7 GiB
# above their BF16 weights while loading and answering the model card's text,
# image and video requests (docs/records/decision3-device-memory.md).
RESIDENT_HEADROOM = 0.10
GIB = 1 << 30


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


def resident_dtype(spec: ModelSpec, accelerator: str) -> str:
    """The dtype ``accelerator`` holds the model's backbone weights in."""
    policy = spec.dtype
    if accelerator == "cpu":
        return policy.cpu_weights or policy.weights
    return policy.gpu_weights or policy.weights


def estimate_bytes(
    spec: ModelSpec, parameters: int, accelerator: str, bf16_parameters: int = 0
) -> int:
    """The weights in the dtype ``accelerator`` holds them in, plus an activation reserve.

    Weights a family holds in BF16 there (``gpu_weights`` / ``cpu_weights``)
    count two bytes each plus ``RESIDENT_HEADROOM``, and so do the
    ``bf16_parameters`` a ``bf16_resident`` family holds in BF16 on a GPU, its
    BF16-stored Linear weights (docs/records/bf16-resident-device-memory.md);
    the rest count as FP32. The reserve covers loading and ordinary requests:
    inputs of tens of thousands of tokens need more, and a forward that runs
    out of device memory fails only its own requests. A quantized checkpoint
    counts its files instead.
    """
    width = DTYPE_BYTES.get(resident_dtype(spec, accelerator), FP32_BYTES)
    held = parameters if width < FP32_BYTES else 0
    if width == FP32_BYTES and accelerator != "cpu" and spec.dtype.bf16_resident:
        width, held = DTYPE_BYTES["bfloat16"], min(bf16_parameters, parameters)
    weights = held * width
    if spec.backbone.quantized is not None:
        # Quantized weights load as stored; the files bound them (the rest is BF16).
        weights = sum(path.stat().st_size for path in spec.backbone.weight_files)
        held = parameters
    weights += int(weights * RESIDENT_HEADROOM)
    return weights + (parameters - held) * FP32_BYTES + ACTIVATION_RESERVE


def place(
    spec: ModelSpec,
    device: str,
    parameters: int,
    memory_budget_gib: float | None = None,
    bf16_parameters: int = 0,
    dtype: str = "auto",
) -> Placement:
    """A device for ``spec`` at ``dtype`` (``--dtype``), which decides the dtypes its weights are held in."""
    kind, index = parse_device(device)
    budget = int(memory_budget_gib * GIB) if memory_budget_gib else None
    candidates = auto_order() if kind == "auto" else [kind]
    gpu_reasons: list[str] = []
    cpu_reasons: list[str] = []
    absent: list[str] = []
    unsupported: list[str] = []
    for name in candidates:
        reasons = cpu_reasons if name == "cpu" else gpu_reasons
        try:
            accelerator = registry.instantiate("accelerators", name)
        except KeyError as exc:
            reasons.append(str(exc))
            continue
        if not accelerator.available():
            absent.append(f"{name}: not available on this host")
            continue
        devices = accelerator.devices()
        if index is not None:
            devices = [d for d in devices if d.index == index]
        required = spec.requires.get(name, ())
        for info in devices:
            # The dtypes a device holds the weights in follow its torch device, as
            # the runtime resolves --dtype and the engines read the policy.
            gpu = accelerator.torch_device(info).type != "cpu"
            running = dtypes.apply(spec, dtype, gpu)
            needed = estimate_bytes(
                running, parameters, name, bf16_parameters if gpu else 0
            )
            if budget is not None and needed > budget:
                reasons.append(
                    f"{name}: {spec.name} needs about {needed / GIB:.1f} GiB, "
                    f"above --memory-budget {memory_budget_gib} GiB"
                )
                break
            if required:
                offered = accelerator.capabilities(info)
                lacking = [need for need in required if not offered.get(need)]
                if lacking:
                    unsupported += [accelerator.lacks(info, need) for need in lacking]
                    continue
            if gpu and not info.bf16 and running.dtype.needs_gpu_bf16:
                reasons.append(f"{info.label}: no BF16 support")
                continue
            if (
                name != "cpu"
                and info.free_memory is not None
                and info.free_memory < needed
            ):
                reasons.append(
                    f"{info.label}: {spec.name} needs about {needed / GIB:.1f} GiB, "
                    f"{info.free_memory / GIB:.1f} GiB free"
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
    # A host with a GPU states the GPU's reason first and leaves out the
    # accelerators it does not have; the CPU's reasons come last.
    shown = (gpu_reasons or absent) + cpu_reasons + unsupported
    message = f"no device can serve {spec.name}: " + "; ".join(shown)
    if unsupported and not (gpu_reasons or cpu_reasons or absent):
        raise UnsupportedDeviceError(message)
    raise PlacementError(message)
