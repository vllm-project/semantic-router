"""Cast owned inference parameters without rounding positional-frequency buffers."""

from __future__ import annotations

import torch
from torch import nn


def cast_parameters_preserving_buffers(
    module: nn.Module, dtype: torch.dtype
) -> nn.Module:
    """Cast floating parameters in place, preserving each buffer's dtype and value.

    Use on an owned export model or independent validation copy before training.
    ModernBERT initializes its RoPE frequencies explicitly in FP32, including
    when Hugging Face loads FP16/BF16 weights. Module.half()/to(dtype) would also
    round those buffers. This helper preserves existing buffer types rather than
    assuming every floating buffer should be FP32; it cannot recover a buffer
    that a previous cast has already rounded. Tied Parameter objects stay tied.
    """
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("Inference parameters require a floating-point dtype")
    parameters = list(module.parameters())
    if any(parameter.grad is not None for parameter in parameters):
        raise ValueError(
            "Cast an independent inference copy before accumulating gradients"
        )
    with torch.no_grad():
        for parameter in parameters:
            if parameter.is_floating_point():
                parameter.data = parameter.data.to(dtype=dtype)
    return module
