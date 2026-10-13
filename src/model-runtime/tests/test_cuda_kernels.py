"""The CUDA accelerator offers the fused element-wise kernels to max_speed only.

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


def test_max_speed_selects_the_fused_kernels(fused_module):
    kernels = cuda.CUDAAccelerator().kernels(device("sm_100"))
    kernels.allow_approximate = True
    assert fused_sources(kernels) == set(FUSED)
    # gated-delta prep reproduces causal-conv1d, so it needs causal-conv1d and FLA
    assert not kernels.has("gdn_prep")


def test_gpus_before_ampere_get_no_fused_kernels(fused_module):
    kernels = cuda.CUDAAccelerator().kernels(device("sm_75"))
    kernels.allow_approximate = True
    assert fused_sources(kernels) == set()


def test_the_receipt_lists_only_the_kernels_the_profile_can_select(fused_module):
    kernels = cuda.CUDAAccelerator().kernels(device("sm_89"))
    assert "add_rmsnorm" not in kernels.describe()
    kernels.allow_approximate = True
    assert kernels.describe()["add_rmsnorm"] == "triton-gfx942"


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
