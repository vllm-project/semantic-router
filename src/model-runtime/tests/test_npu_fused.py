"""The NPU fused-op wrappers with ``torch_npu`` faked: unit-tested without torch_npu or a device."""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch
from vllm_srun.accel import npu_fused
from vllm_srun.accel.kernels import rotary_half_ref
from vllm_srun.accel.npu import NPUAccelerator

HAS_TORCH_NPU = importlib.util.find_spec("torch_npu") is not None


def fake_rotary_mul(
    inp: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, rotary_mode: str = "half"
):
    """The op's documented half-mode algebra, in eager ops (its doc's formula)."""
    assert rotary_mode == "half"
    half = inp.shape[-1] // 2
    rotated = torch.cat((-inp[..., half:], inp[..., :half]), dim=-1)
    return r1 * inp + r2 * rotated


@pytest.fixture
def fake_torch_npu(monkeypatch):
    """A torch_npu module carrying the fused rotate-half op."""
    module = SimpleNamespace(npu_rotary_mul=fake_rotary_mul, __version__="2.10.0.test")
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    return module


def test_rotary_half_registers_as_an_exact_fused_kernel(fake_torch_npu):
    """Where the measured SoC ships the op, the slot's kernel is the fused one, exact."""
    accelerator = NPUAccelerator()
    device = SimpleNamespace(label="npu:0", arch="Ascend950DT")
    kernels = accelerator.kernels(device)
    kernel = kernels.select("rotary_half")
    assert kernel.source == "torch-npu-fused"
    assert kernel.exact is True
    assert kernels.describe()["rotary_half"] == "torch-npu-fused"
    # the reference kernels stay in place for every other slot
    assert kernels.select("sdpa").source == "torch-reference"
    assert kernels.select("geglu").source == "torch-reference"


def test_rotary_half_stays_reference_on_unmeasured_socs(fake_torch_npu):
    """Exactness was measured on the 950DT: other SoCs keep the reference kernel."""
    accelerator = NPUAccelerator()
    kernels = accelerator.kernels(SimpleNamespace(label="npu:0", arch="Ascend910B1"))
    assert kernels.select("rotary_half").source == "torch-reference"


def test_rotary_half_stays_reference_without_the_op(monkeypatch):
    """A torch_npu without the op leaves the reference kernel in the slot."""
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(__version__="2.10.0.test")
    )
    accelerator = NPUAccelerator()
    kernels = accelerator.kernels(SimpleNamespace(label="npu:0", arch="Ascend950DT"))
    assert kernels.select("rotary_half").source == "torch-reference"


def test_reference_kernels_without_torch_npu():
    """On hosts without torch_npu at all there is nothing fused to register."""
    if getattr(torch, "npu", None) is not None or HAS_TORCH_NPU:
        pytest.skip("needs an environment where torch_npu is absent")
    accelerator = NPUAccelerator()
    kernels = accelerator.kernels(SimpleNamespace(label="npu:0", arch="Ascend950DT"))
    assert all(source == "torch-reference" for source in kernels.describe().values())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_half_wrapper_matches_the_reference(fake_torch_npu, dtype):
    """The wrapper's FP32 routing reproduces the reference bit for bit, on every dtype."""
    torch.manual_seed(0)
    query = torch.randn(2, 16, 473, 64, dtype=dtype)
    key = torch.randn(2, 16, 473, 64, dtype=dtype)
    cos = torch.randn(1, 473, 64)
    sin = torch.randn(1, 473, 64)
    expected_q, expected_k = rotary_half_ref(query, key, cos, sin)
    got_q, got_k = npu_fused.rotary_half(query, key, cos, sin)
    assert torch.equal(got_q, expected_q)
    assert torch.equal(got_k, expected_k)


@pytest.mark.parametrize("cos_dtype", [torch.float32, torch.bfloat16])
def test_rotary_half_wrapper_promotes_the_tables(fake_torch_npu, cos_dtype):
    """A BF16 rotary table promotes to FP32 exactly as the reference's broadcast does."""
    torch.manual_seed(0)
    query = torch.randn(1, 8, 195, 64, dtype=torch.bfloat16)
    key = torch.randn(1, 8, 195, 64, dtype=torch.bfloat16)
    cos = torch.randn(1, 195, 64, dtype=cos_dtype)
    sin = torch.randn(1, 195, 64, dtype=cos_dtype)
    expected_q, expected_k = rotary_half_ref(query, key, cos, sin)
    got_q, got_k = npu_fused.rotary_half(query, key, cos, sin)
    assert torch.equal(got_q, expected_q)
    assert torch.equal(got_k, expected_k)


def test_rotary_half_falls_back_beyond_the_op_limits(fake_torch_npu, monkeypatch):
    """Past B*N = 1000 rows*heads the wrapper returns the reference and never calls the op."""
    calls = []
    monkeypatch.setattr(
        npu_fused, "rotary_half_ref", lambda q, k, c, s: ("ref-q", "ref-k")
    )

    def counting_rotary_mul(inp, r1, r2, rotary_mode="half"):
        calls.append(inp.shape)
        return fake_rotary_mul(inp, r1, r2, rotary_mode)

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_rotary_mul=counting_rotary_mul, __version__="2.10.0.test"),
    )
    # 64 heads x 32 rows = 2048 >= 1000: the documented op limit is exceeded
    query = torch.randn(32, 64, 473, 64)
    key = torch.randn(32, 64, 473, 64)
    cos = torch.randn(1, 473, 64)
    sin = torch.randn(1, 473, 64)
    got_q, got_k = npu_fused.rotary_half(query, key, cos, sin)
    assert (got_q, got_k) == ("ref-q", "ref-k")
    assert calls == []


def test_available_wrappers_track_the_op_set(monkeypatch):
    """A torch_npu lacking npu_rotary_mul exposes no wrappers at all."""
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(__version__="2.10.0.test")
    )
    assert npu_fused.available_wrappers() == {}
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_rotary_mul=fake_rotary_mul)
    )
    assert set(npu_fused.available_wrappers()) == {"rotary_half"}
