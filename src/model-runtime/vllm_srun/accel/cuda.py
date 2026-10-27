"""CUDA accelerator. Validated on NVIDIA GPUs (``docs/records/cuda-*.md``).

With Triton on an Ampere or newer GPU it adds the ROCm build's fused
element-wise kernels (``triton_gfx942``) as approximate kernels, which run
only for a family that consents to them. Their RMSNorm sums follow ATen's ROCm reduction
order, not CUDA's, so they differ from the eager ops by rounding
(``docs/records/cuda-fused-approximate.md``).

On Hopper (compute capability 9.x), when ``flash_attn_interface`` imports, it
also registers FlashAttention-3 for ``branch_attention``. That kernel stays
unselected until the ``flash_attention_3`` engine argument opts into it
(``docs/records/cuda-flash-attention-3.md``). Family consent does not select it.
"""

from __future__ import annotations

from typing import Any

import torch

from ..engines.native.models.branch import _flash
from ..plugins.base import DeviceInfo
from . import flash3
from .gpu import GPUAccelerator, _optional
from .kernels import Kernel, KernelSet

# BF16 Triton kernels need Ampere (compute capability 8.0) or newer.
MIN_FUSED_CAPABILITY = 80
FA2_SOURCE = "flash-attn-2"


def _register_branch_attention(kernels: KernelSet, device: DeviceInfo) -> None:
    """FlashAttention-2 for state-once branches, and FlashAttention-3 on Hopper when it is installed.

    FlashAttention-3 is registered as its own variant, so ``allow_approximate``
    does not select it. ``KernelSet.use_source`` does, when the operator opts in.
    """
    if not hasattr(torch.ops.aten, "_flash_attention_forward"):
        return
    kernels.register(Kernel("branch_attention", _flash, FA2_SOURCE, exact=True))
    if flash3.unavailable(device) is not None:
        return
    kernels.register(
        Kernel(
            "branch_attention",
            flash3.varlen(_flash),
            flash3.SOURCE,
            exact=False,
            variant=flash3.SOURCE,
        )
    )


class CUDAAccelerator(GPUAccelerator):
    name = "cuda"
    validated = True
    auto_priority = 1
    platform_attribute = "cuda"

    def available(self) -> bool:
        return not torch.version.hip and super().available()

    def arch(self, properties: Any) -> str | None:
        major = getattr(properties, "major", None)
        minor = getattr(properties, "minor", None)
        return f"sm_{major}{minor}" if major is not None else None

    def kernels(self, device: DeviceInfo) -> KernelSet:
        kernels = super().kernels(device)
        _register_branch_attention(kernels, device)
        capability = (device.arch or "").removeprefix("sm_")
        if (
            not capability.isdigit()
            or int(capability) < MIN_FUSED_CAPABILITY
            or _optional("triton", "__version__") is None
        ):
            return kernels
        from . import triton_gfx942 as fused
        from .rocm import FUSED, FUSED_GATED_DELTA

        names = list(FUSED)
        if (
            kernels.select("causal_conv1d").source == "causal-conv1d"
            and kernels.select("chunk_gated_delta_rule").source == "fla"
        ):
            names += FUSED_GATED_DELTA
        for name in names:
            kernels.register(
                Kernel(name, getattr(fused, name), "triton-gfx942", exact=False)
            )
        return kernels
