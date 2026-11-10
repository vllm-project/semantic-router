"""Fused torch_npu ops as NPU kernel-slot candidates.

The CUDA accelerator's precedent: the same fused-op family is bit-exact on one
device and not on another (``triton_gfx942`` is exact on ROCm, approximate on
CUDA), so exactness is a per-op, per-device measurement. On the Ascend 950DT
the fused rotate-half op reproduces the reference's compute-in-FP32-cast-back
rounding at fp32 and bf16 (measured, ``docs/records/npu-fused-ascend950dt.md``),
so it registers as an exact kernel there; other Ascend SoCs run the reference
kernels until measured, and the fused attention, RMSNorm and SwiGLU ops
measured inexact against their eager references and stay unregistered.
"""

from __future__ import annotations

from typing import Any

import torch

from .kernels import rotary_half_ref

# The wrappers below, and the torch_npu op each one needs.
WRAPPERS: dict[str, tuple[str, ...]] = {
    # wrapper name -> torch_npu ops it requires
    "rotary_half": ("npu_rotary_mul",),
}

# SoCs the exactness of these wrappers was measured on (the rocm.py precedent:
# exact only where measured). Keyed by the SoC name parsed from
# ``torch.npu.get_device_properties(...).name`` ("Ascend950DT_9572" -> "Ascend950DT").
MEASURED_EXACT = frozenset({"Ascend950DT"})

# npu_rotary_mul's documented operating limits (its doc: B*N < 1000 rows*heads,
# D < 896 and even); beyond them the wrapper falls back to the reference.
MAX_ROTARY_ROWS_HEADS = 1000
MAX_ROTARY_HEAD_DIM = 896


def soc_name(name: str | None) -> str | None:
    """The SoC part of a device properties name ("Ascend950DT_9572" -> "Ascend950DT")."""
    if not name:
        return None
    return name.split("_", 1)[0]


def _torch_npu() -> Any:
    """The ``torch_npu`` module once imported, else ``None`` (the npu.py pattern)."""
    try:
        import torch_npu  # noqa: F401  (also registers torch.npu)
    except Exception:
        return None
    import sys

    return sys.modules.get("torch_npu")


def available_wrappers() -> dict[str, Any]:
    """The wrappers whose torch_npu ops this install ships, by wrapper name."""
    module = _torch_npu()
    if module is None:
        return {}
    return {
        name: rotary_half
        for name, ops in WRAPPERS.items()
        if all(hasattr(module, op) for op in ops)
    }


def rotary_half(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused rotate-half op on FP32 inputs, cast back: the reference's rounding.

    ``npu_rotary_mul``'s half mode computes ``r1 * input + r2 * cat((-x2, x1))``,
    the same algebra as ``kernels.rotary_half_ref``. The reference computes in
    FP32 and casts back, and so does this wrapper -- the op's own BF16 path
    rounds differently (measured on Ascend 950DT: one ulp off), so the FP32
    routing is what makes the slot bit-exact on every dtype.

    Beyond the op's documented limits (B*N < 1000 rows*heads, D < 896 and even)
    the wrapper falls back to the reference, so the slot's exact contract holds
    wherever the shapes land.
    """
    d = query.shape[-1]
    rows_heads = query.shape[0] * query.shape[1]
    if rows_heads >= MAX_ROTARY_ROWS_HEADS or d >= MAX_ROTARY_HEAD_DIM or d % 2:
        return rotary_half_ref(query, key, cos, sin)
    module = _torch_npu()
    cos, sin = cos.float().unsqueeze(1), sin.float().unsqueeze(1)  # [1, 1, S, D]
    dtype = query.dtype
    q = module.npu_rotary_mul(query.float(), cos, sin, rotary_mode="half").to(dtype)
    k = module.npu_rotary_mul(key.float(), cos, sin, rotary_mode="half").to(dtype)
    return q, k
