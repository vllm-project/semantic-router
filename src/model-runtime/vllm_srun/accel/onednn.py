"""oneDNN's pre-packed FP32 linear: the ``linear`` slot's ``float32-packed`` variant on x86 CPUs.

The weight is reordered once, at load. On EPYC its GEMMs are 2.5-4.4x faster
than ``F.linear`` (MKL) at every batch size, a row's result is the same alone
or inside any batch (MKL's differs below 32 rows), and it stays within 3.3e-6
of ``F.linear``.
"""

from __future__ import annotations

import platform

import torch
from torch import nn

PACKED = "float32-packed"


def available() -> bool:
    """Whether this host and PyTorch build run the packed linear."""
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    if not torch.backends.mkldnn.is_available():
        return False
    try:
        torch.ops.mkldnn._linear_pointwise  # noqa: B018 - resolves the op or raises
    except (AttributeError, RuntimeError):
        return False
    return True


class PackedLinear(nn.Module):
    """An FP32 ``nn.Linear`` whose weight oneDNN reordered once (x86 CPUs)."""

    packed: torch.Tensor

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight_elements = linear.weight.numel()
        # Reordered for single rows: the layout every batch size then shares.
        packed = torch.ops.mkldnn._reorder_linear_weight(linear.weight.detach(), 1)
        self.register_buffer("packed", packed, persistent=False)
        self.bias = (
            None
            if linear.bias is None
            else nn.Parameter(linear.bias.detach(), requires_grad=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.ops.mkldnn._linear_pointwise(
            x.reshape(-1, self.in_features), self.packed, self.bias, "none", [], ""
        )
        return out.reshape(*x.shape[:-1], self.out_features)
