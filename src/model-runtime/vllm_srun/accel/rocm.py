"""ROCm accelerator (MI300X / MI325X, gfx942). Validated by the parity records.

On gfx942 with Triton installed it adds the fused element-wise kernels of
``triton_gfx942`` (bit-exact against the eager ops under BF16 autocast) and
the ``fp64_accumulate`` convolution variant of ``triton_fp64_conv``. The
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
    "rotary_half",
)
FUSED_GATED_DELTA = ("gdn_prep",)


class ROCmAccelerator(GPUAccelerator):
    name = "rocm"
    validated = True
    auto_priority = 0
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
        from .triton_fp64_conv import (
            FP64_ACCUMULATE,
            FP64_NAIVE,
            fp64_conv,
            gdn_prep_fp64,
        )

        if kernels.select("chunk_gated_delta_rule").source == "fla":
            kernels.register(
                Kernel(
                    "gdn_prep",
                    gdn_prep_fp64,
                    "triton-gfx942-fp64-naive",
                    exact=True,
                    variant=FP64_NAIVE,
                )
            )
        default = kernels.select("causal_conv1d").fn
        kernels.register(
            Kernel(
                "causal_conv1d",
                fp64_conv(default),
                "triton-gfx942-fp64",
                exact=True,
                variant=FP64_ACCUMULATE,
            )
        )
        kernels.register(
            Kernel(
                "causal_conv1d",
                fp64_conv(default, minimum=0),
                "triton-gfx942-fp64-naive",
                exact=True,
                variant=FP64_NAIVE,
            )
        )
        return kernels
