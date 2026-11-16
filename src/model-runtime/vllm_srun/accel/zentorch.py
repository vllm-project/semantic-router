"""zentorch's FP32 linear as the CPU accelerator's approximate ``linear`` kernel.

zentorch (AMD's PyTorch plugin for EPYC, Apache-2.0, on PyPI) reorders a
weight once into ZenDNN's layout (``zentorch_weight_prepack_for_linear``) and
multiplies with it (``zentorch_linear_unary``). Its GEMM sums in another order
than ``F.linear``, so the kernel is not bit-exact against the reference and
registers approximate: it runs only for a family that consents on CPUs
(``DtypePolicy.approximate_cpu_kernels``) at ``--dtype auto``. On vela3-fast at
FP32 it changed no decision and no span over the typed-decisions TEST set and
the vela3 parity requests, with every golden check matched
(``docs/records/cpu-zentorch-linear-vela3.md``).

Three gates, all required, keep it off elsewhere:
- the ``VLLM_SRUN_ZENTORCH`` opt-in (the engine-argument-stays-off-by-default
  rule of the model runtime's AGENTS.md, as ``FLA_NPU_APPROXIMATE``);
- a measured CPU: AMD EPYC Turin onwards (CPU family 1Ah; the record's host is an
  EPYC 9755);
- a zentorch built for the running PyTorch (major.minor) that provides both ops.

Like oneDNN's ``PackedLinear``, the layer keeps only the reordered weight, so
the model's linear weights are held once. FP32 only: a layer in another dtype
keeps ``nn.Linear``.
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Any

import torch
from torch import nn

log = logging.getLogger(__name__)

_OPT_IN = "VLLM_SRUN_ZENTORCH"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# CPU families the kernel was measured on: (vendor, family), from /proc/cpuinfo.
MEASURED = frozenset({("AuthenticAMD", 0x1A)})

_CACHE: dict[str, Any] = {}


def _ops() -> Any:
    """zentorch's op namespace (one seam, so tests can stand in for zentorch)."""
    return torch.ops.zentorch


def opted_in() -> bool:
    """Whether zentorch's approximate CPU kernels are explicitly opted in (off by default)."""
    return os.environ.get(_OPT_IN, "").strip().lower() in _TRUTHY


def cpu_family(cpuinfo: str = "/proc/cpuinfo") -> tuple[str, int] | None:
    """This host's (vendor, family) from ``/proc/cpuinfo``, or None where it cannot be read."""
    vendor = family = None
    try:
        with open(cpuinfo, encoding="utf-8") as lines:
            for line in lines:
                key, _, value = line.partition(":")
                key = key.strip()
                if key == "vendor_id":
                    vendor = value.strip()
                elif key == "cpu family":
                    family = int(value.strip())
                if vendor is not None and family is not None:
                    return vendor, family
    except (OSError, ValueError):
        return None
    return None


def measured() -> bool:
    """Whether this host is a CPU the kernel was measured on."""
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    return cpu_family() in MEASURED


def _load() -> bool:
    try:
        import zentorch  # registers torch.ops.zentorch
    except Exception as error:  # any import failure keeps the reference
        log.info("zentorch not used: import failed (%s)", error)
        return False
    built = ".".join(str(getattr(zentorch, "__version__", "")).split(".")[:2])
    running = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    if built != running:
        log.warning(
            "zentorch not used: zentorch %s is built for PyTorch %s, this is PyTorch %s",
            zentorch.__version__,
            built,
            running,
        )
        return False
    try:
        _ops().zentorch_weight_prepack_for_linear  # noqa: B018
        _ops().zentorch_linear_unary  # noqa: B018
    except (AttributeError, RuntimeError) as error:
        log.warning("zentorch not used: an op is missing (%s)", error)
        return False
    return True


def available() -> bool:
    """Whether zentorch loads, matches this PyTorch and provides the ops (checked once)."""
    if "available" not in _CACHE:
        _CACHE["available"] = _load()
    return bool(_CACHE["available"])


def enabled() -> bool:
    """All three gates: opted in, a measured CPU, a usable zentorch."""
    return opted_in() and measured() and available()


class ZentorchLinear(nn.Module):
    """An FP32 ``nn.Linear`` whose weight zentorch reordered once."""

    packed: torch.Tensor

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight_elements = linear.weight.numel()
        # A copy: the reorder must not alias a tensor the package's weights still hold.
        packed = _ops().zentorch_weight_prepack_for_linear(
            linear.weight.detach().clone()
        )
        self.register_buffer("packed", packed, persistent=False)
        self.bias = (
            None
            if linear.bias is None
            else nn.Parameter(linear.bias.detach(), requires_grad=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = _ops().zentorch_linear_unary(
            x.reshape(-1, self.in_features),
            self.packed,
            self.bias,
            is_weight_prepacked=True,
        )
        return out.reshape(*x.shape[:-1], self.out_features)


def lay_out(linear: nn.Linear) -> nn.Module:
    """The ``linear`` slot's layout: zentorch's for FP32 layers, ``nn.Linear`` otherwise."""
    if linear.weight.dtype != torch.float32:
        return linear
    return ZentorchLinear(linear)
