"""GPU: the FP64-accumulating causal convolution (``accel/triton_fp64_conv``) at any input length.

Each output is the four taps of its channel summed in FP64, where BF16 products and their
sum are exact, rounded to BF16 and passed through SiLU in FP32. The kernel takes the length
and strides at run time, so it compiles once whatever lengths the requests bring.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from vllm_srun.accel.triton_fp64_conv import (  # noqa: E402
    TAPS,
    _causal_silu,
    causal_silu,
)

pytestmark = pytest.mark.gpu


def reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    length = x.shape[-1]
    padded = torch.nn.functional.pad(x.double(), (TAPS - 1, 0))
    taps = weight.double()
    summed = sum(padded[..., k : k + length] * taps[:, k, None] for k in range(TAPS))
    value = summed.float().to(x.dtype).float()
    return (value / (1 + torch.exp(-value))).to(x.dtype)


@pytest.mark.parametrize("batch,length", [(1, 5), (2, 37), (1, 1029), (3, 4352)])
def test_the_convolution_reads_every_length_and_batch(batch, length):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    generator = torch.Generator(device="cuda").manual_seed(length)
    x = torch.randn(
        batch, length, 96, device="cuda", dtype=torch.bfloat16, generator=generator
    ).transpose(1, 2)
    weight = torch.randn(
        96, TAPS, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    got = causal_silu(x, weight)
    assert got.shape == x.shape
    torch.testing.assert_close(got, reference(x, weight))


def test_no_length_or_stride_is_a_compile_time_constant():
    constants = {param.name for param in _causal_silu.params if param.is_constexpr}
    assert constants == {"D", "K", "BLOCK"}
