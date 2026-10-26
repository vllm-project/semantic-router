"""The fla-npu chunk slot with the wrapper faked: unit-tested without fla_npu or a device."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm_srun.accel import npu_fla
from vllm_srun.accel.kernels import chunk_gated_delta_rule_ref
from vllm_srun.accel.npu import NPUAccelerator


def fake_flash_gated_delta_rule(query, key, value, g=None, beta=None, scale=None, **kw):
    """A stand-in wrapper: asserts the head-major input layout, returns plausible outputs.

    The real wrapper reads [B, H, T, D] projections and returns the output
    already in [B, T, H, D] (its internal layout handling), with the recurrent
    state in [B, H, K, V].
    """
    assert (
        query.shape[2] == value.shape[2] and query.shape[1] == key.shape[1]
    ), "the wrapper reads the head-major (NTD) projections: [B, H, T, D]"
    batch, heads, time, dim = query.shape
    output = torch.zeros(batch, time, heads, dim, dtype=query.dtype)
    state = None
    if kw.get("output_final_state"):
        state = torch.zeros(batch, heads, query.shape[-1], dim, dtype=query.dtype)
    return output, state


@pytest.fixture
def fake_wrapper(monkeypatch):
    """The fla_npu package import resolving to the stand-in wrapper."""
    package = ModuleType("fla_npu")
    module = ModuleType("fla_npu.flash_gated_delta_rule")
    module.flash_gated_delta_rule = fake_flash_gated_delta_rule
    package.flash_gated_delta_rule = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fla_npu", package)
    monkeypatch.setitem(sys.modules, "fla_npu.flash_gated_delta_rule", module)
    monkeypatch.setattr(npu_fla, "_wrapper", False)
    return fake_flash_gated_delta_rule


def test_package_import_wins_over_the_examples_path(fake_wrapper, monkeypatch):
    """The loader prefers the installed package's wrapper over FLA_NPU_EXAMPLES."""
    monkeypatch.setenv("FLA_NPU_EXAMPLES", "/nonexistent/examples")
    assert npu_fla.available_wrapper() is fake_flash_gated_delta_rule


def test_chunk_registers_approximate_where_measured(fake_wrapper):
    """On the measured SoC the slot carries the fla-npu kernel, dormant without consent."""
    accelerator = NPUAccelerator()
    kernels = accelerator.kernels(SimpleNamespace(label="npu:0", arch="Ascend950DT"))
    assert kernels.preferred_source("chunk_gated_delta_rule") == "fla-npu-ascendc"
    assert kernels.available["chunk_gated_delta_rule"][0].exact is False
    # without the family's consent the slot serves the reference (dormant)
    assert kernels.select("chunk_gated_delta_rule").source == "torch-reference"


def test_chunk_selects_under_consent(fake_wrapper):
    """With the family's approximate consent the slot selects the fla-npu kernel."""
    accelerator = NPUAccelerator()
    kernels = accelerator.kernels(SimpleNamespace(label="npu:0", arch="Ascend950DT"))
    kernels.allow_approximate = True
    assert kernels.select("chunk_gated_delta_rule").source == "fla-npu-ascendc"


def test_chunk_stays_reference_on_unmeasured_socs(fake_wrapper):
    """Approximate was measured on the 950DT: other SoCs keep the reference kernel."""
    accelerator = NPUAccelerator()
    kernels = accelerator.kernels(SimpleNamespace(label="npu:0", arch="Ascend910B1"))
    assert kernels.select("chunk_gated_delta_rule").source == "torch-reference"


def test_chunk_stays_reference_without_the_wrapper(monkeypatch):
    """An install without fla-npu leaves the reference kernel in the slot."""
    monkeypatch.delenv("FLA_NPU_EXAMPLES", raising=False)
    monkeypatch.setattr(npu_fla, "_wrapper", False)
    accelerator = NPUAccelerator()
    kernels = accelerator.kernels(SimpleNamespace(label="npu:0", arch="Ascend950DT"))
    assert kernels.select("chunk_gated_delta_rule").source == "torch-reference"


def test_wrapper_loaded_from_the_examples_directory(monkeypatch):
    """Without the package's wrapper module, the FLA_NPU_EXAMPLES path supplies it."""
    package = ModuleType("fla_npu")  # the package without the wrapper module
    monkeypatch.setitem(sys.modules, "fla_npu", package)
    monkeypatch.delitem(sys.modules, "fla_npu.flash_gated_delta_rule", raising=False)
    module = ModuleType("flash_gated_delta_rule")
    module.flash_gated_delta_rule = fake_flash_gated_delta_rule
    monkeypatch.setitem(sys.modules, "flash_gated_delta_rule", module)
    monkeypatch.setenv("FLA_NPU_EXAMPLES", "/fake/examples")
    monkeypatch.setattr(npu_fla, "_wrapper", False)
    wrapper = npu_fla._load_wrapper()
    assert wrapper is fake_flash_gated_delta_rule


def test_wrapper_loader_without_any_install(monkeypatch):
    """No package and no examples directory: the loader returns None."""
    monkeypatch.delenv("FLA_NPU_EXAMPLES", raising=False)
    monkeypatch.delitem(sys.modules, "fla_npu", raising=False)
    monkeypatch.delitem(sys.modules, "flash_gated_delta_rule", raising=False)
    monkeypatch.setattr(npu_fla, "_wrapper", False)
    assert npu_fla._load_wrapper() is None


def test_wrapper_transposes_to_the_head_major_layout(fake_wrapper):
    """The wrapper reads [B, H, T, D] and the output comes back in [B, T, H, D]."""
    torch.manual_seed(0)
    batch, time, heads, dim = 1, 96, 32, 128
    # the slot contract: every input carries the v-head count (the caller repeats)
    query = torch.randn(batch, time, heads, dim, dtype=torch.bfloat16)
    key = torch.randn(batch, time, heads, dim, dtype=torch.bfloat16)
    value = torch.randn(batch, time, heads, dim, dtype=torch.bfloat16)
    g = torch.randn(batch, time, heads, dtype=torch.bfloat16)
    beta = torch.randn(batch, time, heads, dtype=torch.bfloat16)
    output, state = npu_fla.chunk_gated_delta_rule(
        query, key, value, g, beta, output_final_state=True
    )
    assert output.shape == query.shape
    assert output.dtype == torch.bfloat16
    assert state is not None and state.shape[1] == heads


def test_wrapper_guards_non_bf16_inputs(fake_wrapper, monkeypatch):
    """An fp32 backbone falls back to the reference: the kernel is bf16-only."""
    calls = []

    def counting_wrapper(*args, **kw):
        calls.append(args)
        return fake_flash_gated_delta_rule(*args, **kw)

    monkeypatch.setattr(npu_fla, "_wrapper", counting_wrapper)
    torch.manual_seed(0)
    batch, time, heads, dim = 1, 96, 32, 128
    query = torch.randn(batch, time, heads, dim, dtype=torch.float32)
    key = torch.randn(batch, time, heads, dim, dtype=torch.float32)
    value = torch.randn(batch, time, heads, dim, dtype=torch.float32)
    g = torch.randn(batch, time, heads)
    beta = torch.randn(batch, time, heads)
    expected, _ = chunk_gated_delta_rule_ref(query, key, value, g, beta)
    got, _ = npu_fla.chunk_gated_delta_rule(query, key, value, g, beta)
    # the fallback IS the reference call: identical outputs (NaN-safe on the
    # degenerate random inputs the eager triangular solve can produce)
    assert torch.allclose(got, expected, equal_nan=True)
    assert calls == []
