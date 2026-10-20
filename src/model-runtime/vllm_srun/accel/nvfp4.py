"""NVFP4 numerics in plain torch: the format compressed-tensors' ``nvfp4-pack-quantized`` stores (``quantization.NVFP4``).

A value is an FP4 (E2M1) code times its block's FP8 (E4M3) scale divided by a
per-tensor FP32 global scale: ``x ~ e2m1 * (scale / global)``, one scale per
16 consecutive values of a row. Codes pack two per byte, the even column in
the low nibble. Quantizing a block takes ``scale = amax * (1 / 6) * global``
clamped to the FP8 range (448) and rounded once to E4M3, and rounds
``x / (scale / global)`` to the nearest E2M1 value, ties to the even code, as
compressed-tensors does. The scale multiplies by the FP32 reciprocal of 6
rather than dividing, so every device computes the same bits (torch divides a
CUDA tensor by a scalar that way, and by a CPU tensor exactly).

These references run on any device: the fixture writer and the tests use them,
and ``triton_nvfp4`` computes the same codes and scales on GPUs. cuBLAS reads
block scales in its 32x4x4 layout (``swizzle``).
"""

from __future__ import annotations

import torch

FP4_MAX = 6.0
FP8_MAX = 448.0
GROUP = 16
# The E2M1 magnitudes by code (the sign is bit 3).
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Rounding thresholds between consecutive magnitudes: above one rounds up,
# at one rounds to the even code (``>=`` where the upper code is even).
_UP = (
    (0.25, False),
    (0.75, True),
    (1.25, False),
    (1.75, True),
    (2.5, False),
    (3.5, True),
    (5.0, False),
)


def global_scale(x: torch.Tensor) -> torch.Tensor:
    """The FP32 global scale compressed-tensors gives a tensor: the FP8 and FP4 ranges over its amax."""
    amax = x.float().abs().amax().clamp(min=torch.finfo(torch.float32).tiny)
    return (FP8_MAX * FP4_MAX / amax).reshape(1)


def encode(v: torch.Tensor) -> torch.Tensor:
    """E2M1 codes (int32, sign in bit 3) of ``v`` rounded to the nearest representable value."""
    a = v.abs().clamp(max=FP4_MAX)
    code = torch.zeros_like(a, dtype=torch.int32)
    for threshold, even_up in _UP:
        code += (a >= threshold if even_up else a > threshold).to(torch.int32)
    return code | ((v < 0).to(torch.int32) << 3)


def decode(code: torch.Tensor) -> torch.Tensor:
    """FP32 values of E2M1 codes."""
    magnitude = torch.tensor(E2M1, device=code.device)[code & 7]
    return torch.where(code & 8 > 0, -magnitude, magnitude)


def pack(code: torch.Tensor) -> torch.Tensor:
    """[rows, K] codes -> [rows, K / 2] uint8, the even column in the low nibble."""
    return (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)


def unpack(packed: torch.Tensor) -> torch.Tensor:
    """[rows, K / 2] uint8 -> [rows, K] codes."""
    p = packed.to(torch.int32)
    return torch.stack((p & 0xF, p >> 4), dim=-1).reshape(p.shape[0], -1)


def quantize(
    x: torch.Tensor, global_: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """[rows, K] values -> packed codes [rows, K / 2] (uint8) and block scales [rows, K / 16] (FP8 E4M3)."""
    rows, k = x.shape
    blocks = x.float().reshape(rows, k // GROUP, GROUP)
    g = global_.float().reshape(())
    scale = (blocks.abs().amax(-1) * (1.0 / FP4_MAX) * g).clamp(max=FP8_MAX)
    scale = scale.to(torch.float8_e4m3fn)
    denom = scale.float() / g
    v = blocks / torch.where(denom > 0, denom, torch.ones_like(denom)).unsqueeze(-1)
    return pack(encode(v).reshape(rows, k)), scale


def dequantize(
    packed: torch.Tensor, scale: torch.Tensor, global_: torch.Tensor
) -> torch.Tensor:
    """FP32 [rows, K] values of packed codes, their block scales and the global scale."""
    values = decode(unpack(packed))
    rows, k = values.shape
    blocks = values.reshape(rows, k // GROUP, GROUP) * scale.float().unsqueeze(-1)
    return (blocks / global_.float().reshape(())).reshape(rows, k)


def swizzle(scale: torch.Tensor) -> torch.Tensor:
    """cuBLAS's 32x4x4 layout of a [rows, K / 16] block-scale matrix (rows padded to 128, columns to 4), flat."""
    rows, cols = scale.shape
    pr, pc = -(-rows // 128) * 128, -(-cols // 4) * 4
    if (pr, pc) != (rows, cols):
        padded = torch.zeros(pr, pc, dtype=scale.dtype, device=scale.device)
        padded[:rows, :cols] = scale
        scale = padded
    blocks = scale.view(pr // 128, 128, pc // 4, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()
