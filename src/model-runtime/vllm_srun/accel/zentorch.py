"""zentorch's kernels for the CPU accelerator's ``linear`` and ``sdpa`` slots, both approximate.

zentorch (AMD's PyTorch plugin for EPYC, Apache-2.0, on PyPI) reorders a
weight once into ZenDNN's layout (``zentorch_weight_prepack_for_linear``) and
multiplies with it (``zentorch_linear_unary``). Its GEMM sums in another order
than ``F.linear``, so the kernel is not bit-exact against the reference and
registers approximate: it runs only for a family that consents on CPUs
(``DtypePolicy.approximate_cpu_kernels``) at ``--dtype auto``. On vela3-fast at
FP32 it changed no decision and no span over the typed-decisions TEST set and
the vela3 parity requests, with every golden check matched
(``docs/records/cpu-zentorch-linear-vela3.md``). ``zentorch_sdpa`` fills the
``sdpa`` slot the same way; a family consents to it by naming the slot
(``DtypePolicy.approximate_cpu_kernels``), on the strength of its own record.

Three gates, all required, keep it off elsewhere:
- the ``VLLM_SRUN_ZENTORCH`` opt-in (the engine-argument-stays-off-by-default
  rule of the model runtime's AGENTS.md, as ``FLA_NPU_APPROXIMATE``);
- a measured CPU: AMD EPYC Turin onwards (CPU family 1Ah; the record's host is an
  EPYC 9755);
- a zentorch built for the running PyTorch (major.minor); each slot registers
  only where that zentorch provides its ops (``SLOT_OPS``).

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
import torch.nn.functional as F
from torch import nn

log = logging.getLogger(__name__)

_OPT_IN = "VLLM_SRUN_ZENTORCH"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# CPU families the kernel was measured on: (vendor, family), from /proc/cpuinfo.
MEASURED = frozenset({("AuthenticAMD", 0x1A)})

# The zentorch ops each slot needs; a zentorch without them keeps that slot's reference.
SLOT_OPS = {
    "linear": ("zentorch_weight_prepack_for_linear", "zentorch_linear_unary"),
    "sdpa": ("zentorch_sdpa",),
}

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
    return True


def available() -> bool:
    """Whether zentorch loads and matches this PyTorch (checked once)."""
    if "available" not in _CACHE:
        _CACHE["available"] = _load()
    return bool(_CACHE["available"])


def _has(op: str) -> bool:
    try:
        getattr(_ops(), op)
    except (AttributeError, RuntimeError):
        return False
    return True


def slots() -> frozenset[str]:
    """The slots whose ops the installed zentorch provides (checked once)."""
    if "slots" not in _CACHE:
        found = frozenset(
            slot for slot, ops in SLOT_OPS.items() if all(_has(op) for op in ops)
        )
        for slot in sorted(set(SLOT_OPS) - found):
            log.warning("zentorch %s slot not used: an op is missing", slot)
        _CACHE["slots"] = found
    return frozenset(_CACHE["slots"])


def enabled() -> frozenset[str]:
    """The slots to register: opted in, a measured CPU, a usable zentorch with each slot's ops."""
    if not (opted_in() and measured() and available()):
        return frozenset()
    return slots()


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


_SDPA_FALLBACKS: set[str] = set()


def sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    *,
    scale: float,
    is_causal: bool,
    enable_gqa: bool,
) -> torch.Tensor:
    """The ``sdpa`` slot on ``zentorch_sdpa``.

    zentorch takes grouped key and value heads and additive float masks of any
    broadcast shape, but not boolean masks, so a boolean mask becomes 0 where
    allowed and the dtype's minimum elsewhere. An input the op rejects runs the
    reference instead, logged once per reason.
    """
    mask = attn_mask
    if mask is not None and mask.dtype == torch.bool:
        mask = torch.zeros(
            mask.shape, dtype=query.dtype, device=query.device
        ).masked_fill_(~mask, torch.finfo(query.dtype).min)
    try:
        out: torch.Tensor = _ops().zentorch_sdpa(
            query, key, value, 0.0, is_causal, attn_mask=mask, scale=scale
        )[0]
    except RuntimeError as error:
        reason = (
            str(error).splitlines()[0][:200] if str(error) else type(error).__name__
        )
        if reason not in _SDPA_FALLBACKS:
            _SDPA_FALLBACKS.add(reason)
            log.warning("zentorch sdpa fell back to the reference: %s", reason)
        extra = {"enable_gqa": True} if enable_gqa else {}
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            scale=scale,
            is_causal=is_causal,
            **extra,
        )
    return out
