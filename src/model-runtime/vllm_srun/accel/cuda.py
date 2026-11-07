"""CUDA accelerator (hardware records: ``docs/records/cuda-*.md``).

Fused Triton kernels require Ampere and family consent; gated-delta prep also
needs FLA, but not causal-conv1d. Optional causal-conv1d is approximate on CUDA,
so exact-only/non-consenting models use the torch reference even when it is
installed. This can differ from their released package runtime; see the
convolution record below.

Evidence under ``docs/records/``: ``cuda-fused-approximate.md``,
``cuda-causal-conv1d-exactness.md`` and ``cuda-gdn-prep-without-causal-conv1d.md``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from ..plugins.base import DeviceInfo
from .gpu import GPUAccelerator, _optional
from .kernels import Kernel, KernelSet

# BF16 Triton kernels need Ampere (compute capability 8.0) or newer.
MIN_FUSED_CAPABILITY = 80


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
        kernels.available["causal_conv1d"] = [
            replace(kernel, exact=False) if kernel.source == "causal-conv1d" else kernel
            for kernel in kernels.available.get("causal_conv1d", [])
        ]
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
        if kernels.select("chunk_gated_delta_rule").source == "fla":
            names += FUSED_GATED_DELTA
        for name in names:
            kernels.register(
                Kernel(name, getattr(fused, name), "triton-gfx942", exact=False)
            )
        return kernels
