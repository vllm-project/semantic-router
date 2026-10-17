"""Fused torch_npu ops as NPU kernel-slot candidates.

The CUDA accelerator's precedent: the same fused-op family is bit-exact on one
device and not on another (``triton_gfx942`` is exact on ROCm, approximate on
CUDA), so exactness is a per-op, per-device measurement. On Ascend 950DT the
fused rotate-half op reproduces the reference's compute-in-FP32-cast-back
rounding on every dtype (measured, ``docs/records/npu-fused-ascend950dt.md``),
so it registers as an exact kernel; the fused attention, RMSNorm and SwiGLU
ops measured inexact against their eager references and stay unregistered
until a family consents and a record lands.
"""

from __future__ import annotations

from typing import Any

import torch

# The wrappers below, and the torch_npu op each one needs.
WRAPPERS: dict[str, tuple[str, ...]] = {
    # wrapper name -> torch_npu ops it requires
    "rotary_half": ("npu_rotary_mul",),
}


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
    import sys

    here = sys.modules[__name__]
    return {
        name: getattr(here, name)
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
    """
    module = _torch_npu()
    cos, sin = cos.float().unsqueeze(1), sin.float().unsqueeze(1)  # [1, 1, S, D]
    dtype = query.dtype
    q = module.npu_rotary_mul(query.float(), cos, sin, rotary_mode="half").to(dtype)
    k = module.npu_rotary_mul(key.float(), cos, sin, rotary_mode="half").to(key.dtype)
    return q, k
