"""GPU: ModernBERT's fused rotary equals the reference bit for bit (ROCm gfx942).

Query and key are the halves of a fused QKV projection viewed as
``[B, H, T, D]``, as the backbone passes them; FP32 at the released head size,
an odd row count, a head size whose half is not a power of two, and BF16.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllm_srun.accel.kernels import rotary_half_ref  # noqa: E402
from vllm_srun.accel.rocm import ROCmAccelerator  # noqa: E402

pytestmark = pytest.mark.gpu

SHAPES = [
    (1, 1024, 12, 64, torch.float32),
    (3, 77, 12, 64, torch.float32),
    (2, 130, 4, 96, torch.float32),
    (2, 64, 12, 64, torch.bfloat16),
]


def test_fused_rotary_equals_the_reference():
    accelerator = ROCmAccelerator()
    if not accelerator.available():
        pytest.skip("needs a ROCm GPU")
    device = accelerator.devices()[0]
    if device.arch != "gfx942":
        pytest.skip("the fused kernels are validated on gfx942")
    kernels = accelerator.kernels(device)
    assert kernels.select("rotary_half").source == "triton-gfx942"
    target = accelerator.torch_device(device)
    generator = torch.Generator(device=target).manual_seed(0)
    for batch, width, heads, dim, dtype in SHAPES:
        projected = torch.randn(
            batch, width, 3, heads, dim, device=target, generator=generator
        ).to(dtype)
        query, key = (projected[:, :, part].transpose(1, 2) for part in (0, 1))
        angle = torch.randn(1, width, dim, device=target, generator=generator)
        cos, sin = angle.cos().to(dtype), angle.sin().to(dtype)
        expected = rotary_half_ref(query, key, cos, sin)
        actual = kernels("rotary_half")(query, key, cos, sin)
        assert all(torch.equal(a, e) for a, e in zip(actual, expected, strict=True))
