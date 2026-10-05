"""ROCm accelerator (MI300X / MI325X, gfx942). Validated by the parity records.

On gfx942 with Triton installed it adds the fused element-wise kernels of
``triton_gfx942`` (bit-exact against the eager ops under BF16 autocast). The
gated-delta prep reproduces causal-conv1d's ROCm build, so it is registered only
where the convolution and the chunked gated delta rule come from causal-conv1d
and FLA, as in the released runtime.
"""

from __future__ import annotations

from typing import Any

from ..plugins.base import DeviceInfo
from .gpu import GPUAccelerator, _optional
from .kernels import Kernel, KernelSet

FUSED = (
    "add_rmsnorm",
    "residual_add",
    "silu_mul",
    "attn_prep",
    "gated_rmsnorm",
    "sigmoid_gate",
)
FUSED_GATED_DELTA = ("gdn_prep",)


class ROCmAccelerator(GPUAccelerator):
    name = "rocm"
    validated = True
    platform_attribute = "hip"

    def arch(self, properties: Any) -> str | None:
        name = getattr(properties, "gcnArchName", None)
        return name.split(":", 1)[0] if isinstance(name, str) else None

    def kernels(self, device: DeviceInfo) -> KernelSet:
        kernels = super().kernels(device)
        if device.arch != "gfx942" or _optional("triton", "__version__") is None:
            return kernels
        from . import triton_gfx942 as fused

        names = list(FUSED)
        if (
            kernels.select("causal_conv1d").source == "causal-conv1d"
            and kernels.select("chunk_gated_delta_rule").source == "fla"
        ):
            names += FUSED_GATED_DELTA
        for name in names:
            kernels.register(
                Kernel(name, getattr(fused, name), "triton-gfx942", exact=True)
            )
        return kernels
