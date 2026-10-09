"""CUDA accelerator. Implemented and unit-tested; not validated on NVIDIA hardware.

With Triton on an Ampere or newer GPU it adds the ROCm build's fused
element-wise kernels (``triton_gfx942``) as approximate kernels, so only
``max_speed`` selects them. Their RMSNorm sums follow ATen's ROCm reduction
order, not CUDA's, so they differ from the eager ops by rounding
(``docs/records/cuda-fused-approximate.md``).
"""

from __future__ import annotations

from typing import Any

import torch

from ..plugins.base import DeviceInfo
from .gpu import GPUAccelerator, _optional
from .kernels import Kernel, KernelSet

# BF16 Triton kernels need Ampere (compute capability 8.0) or newer.
MIN_FUSED_CAPABILITY = 80


class CUDAAccelerator(GPUAccelerator):
    name = "cuda"
    validated = False
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
