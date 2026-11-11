"""The CUDA accelerator offers the fused element-wise kernels only where approximate kernels are allowed.

The kernel module is replaced by a stand-in so the registration runs without Triton or a GPU.
"""

from __future__ import annotations

import sys
import types

import pytest
from vllm_srun.accel import cuda
from vllm_srun.accel.rocm import FUSED
from vllm_srun.plugins.base import DeviceInfo


@pytest.fixture
def fused_module(monkeypatch):
    module = types.ModuleType("vllm_srun.accel.triton_gfx942")
    for name in (*FUSED, "gdn_prep"):
        setattr(module, name, lambda *args, _name=name: _name)
    monkeypatch.setitem(sys.modules, "vllm_srun.accel.triton_gfx942", module)
    monkeypatch.setattr("vllm_srun.accel.triton_gfx942", module, raising=False)
    monkeypatch.setattr(
        cuda,
        "_optional",
        lambda module, attribute: "3.6.0" if module == "triton" else None,
    )
    # Neither causal-conv1d nor FLA, whatever the test environment installs.
    from vllm_srun.accel import gpu

    monkeypatch.setattr(gpu, "_optional", lambda module, attribute: None)
    return module


def device(arch):
    return DeviceInfo(accelerator="cuda", index=0, name="test", bf16=True, arch=arch)


def fused_sources(kernels):
    """The slots served by the fused kernels (rotary_half also has a reference kernel)."""
    return {
        name
        for name in FUSED
        if kernels.has(name) and kernels.select(name).source == "triton-gfx942"
    }


def test_exact_kernels_never_select_the_fused_kernels(fused_module):
    kernels = cuda.CUDAAccelerator().kernels(device("sm_89"))
    assert fused_sources(kernels) == set()


def test_allowed_approximate_kernels_select_the_fused_kernels(fused_module):
    kernels = cuda.CUDAAccelerator().kernels(device("sm_100"))
    kernels.allow_approximate = True
    assert fused_sources(kernels) == set(FUSED)
    # Without FLA, the fused gated-delta layers cannot run.
    assert not kernels.has("gdn_prep")


def test_gpus_before_ampere_get_no_fused_kernels(fused_module):
    kernels = cuda.CUDAAccelerator().kernels(device("sm_75"))
    kernels.allow_approximate = True
    assert fused_sources(kernels) == set()


def test_the_receipt_lists_only_the_kernels_the_model_may_select(fused_module):
    kernels = cuda.CUDAAccelerator().kernels(device("sm_89"))
    assert "add_rmsnorm" not in kernels.describe()
    kernels.allow_approximate = True
    assert kernels.describe()["add_rmsnorm"] == "triton-gfx942"


@pytest.fixture(params=[False, True], ids=["without-FLA", "with-FLA"])
def fla_installed(request):
    return request.param


@pytest.fixture(
    params=[False, True], ids=["without-causal-conv1d", "with-causal-conv1d"]
)
def causal_conv1d_installed(request):
    return request.param


@pytest.fixture
def optional_kernels(monkeypatch, fused_module, fla_installed, causal_conv1d_installed):
    """Exercise all four FLA/causal-conv1d installation combinations."""
    from vllm_srun.accel import gpu

    def conv(x, weight, bias=None, activation=None):
        return x

    def delta(
        q,
        k,
        v,
        g=None,
        beta=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
    ):
        return v, None

    found = {}
    if fla_installed:
        found["fla.ops.gated_delta_rule"] = delta
    if causal_conv1d_installed:
        found["causal_conv1d"] = conv
    monkeypatch.setattr(gpu, "_optional", lambda module, attribute: found.get(module))


@pytest.mark.parametrize("arch", ["sm_70", "sm_75", "sm_89", "sm_121"])
@pytest.mark.parametrize("allow_approximate", [False, True])
def test_optional_kernels_require_consent_for_cuda_conv_and_fused_prep(
    optional_kernels, fla_installed, causal_conv1d_installed, arch, allow_approximate
):
    kernels = cuda.CUDAAccelerator().kernels(device(arch))
    kernels.allow_approximate = allow_approximate
    # Optional CUDA convolution needs consent, independently of FLA and arch.
    # These stubs check selection policy, not numerical exactness.
    assert kernels.select("causal_conv1d").source == (
        "causal-conv1d"
        if causal_conv1d_installed and allow_approximate
        else "torch-reference"
    )
    assert kernels.select("chunk_gated_delta_rule").source == (
        "fla" if fla_installed else "torch-reference"
    )
    expected_prep = fla_installed and allow_approximate and int(arch[3:]) >= 80
    assert kernels.has("gdn_prep") is expected_prep
    if expected_prep:
        assert kernels.select("gdn_prep").source == "triton-gfx942"


@pytest.mark.parametrize("consents", [False, True])
def test_approximate_kernels_need_the_familys_consent(tmp_path, consents):
    from vllm_srun.accel.cpu import CPUAccelerator
    from vllm_srun.engines.native.engine import NativeEngine
    from vllm_srun.plugins.base import (
        BackboneSpec,
        DtypePolicy,
        EngineOptions,
        ModelSpec,
    )
    from vllm_srun.testing.fixtures import random_backbone, save

    from .test_modernbert import CONFIGS

    config = CONFIGS["yarn"]
    state = random_backbone("modernbert", config, seed=7)
    save({f"model.{n}": v for n, v in state.items()}, tmp_path / "model.safetensors")
    spec = ModelSpec(
        "tiny",
        BackboneSpec("modernbert", config, (tmp_path / "model.safetensors",), "model."),
        DtypePolicy(autocast=None, bf16_resident=False, approximate_kernels=consents),
        max_input_tokens=4096,
        encoder=True,
    )
    accelerator = CPUAccelerator()
    model = NativeEngine().load(
        spec,
        accelerator,
        accelerator.devices()[0],
        EngineOptions(exact_kernels_only=False),
    )
    assert model.kernels.allow_approximate is consents
