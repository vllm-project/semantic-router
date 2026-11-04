"""FlashAttention-3 is an opt-in for Hopper, not a family's approximate-kernel consent.

Selection tests replace ``flash_attn_interface`` and run without a GPU. A missing
install is ``sys.modules["flash_attn_interface"] = None``: deleting the entry lets
``load`` import a real copy when one is installed. The kernel comparison runs
only on a Hopper GPU where FlashAttention-3 is installed.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
from vllm_srun.accel import cuda, flash3
from vllm_srun.accel.kernels import Kernel, KernelSet
from vllm_srun.engines.native.models.branch import _flash, _varlen_kernel
from vllm_srun.plugins.base import DeviceInfo


def device(arch, accelerator="cuda"):
    return DeviceInfo(
        accelerator=accelerator, index=0, name="test", bf16=True, arch=arch
    )


def install_flash3(monkeypatch):
    """A stand-in ``flash_attn_varlen_func`` that records keyword calls and returns the queries."""
    calls = []

    def flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=None,
        causal=False,
        num_splits=None,
    ):
        calls.append(
            {
                "q": q,
                "k": k,
                "v": v,
                "causal": causal,
                "softmax_scale": softmax_scale,
                "num_splits": num_splits,
            }
        )
        return q

    module = types.ModuleType("flash_attn_interface")
    module.flash_attn_varlen_func = flash_attn_varlen_func
    monkeypatch.setitem(sys.modules, "flash_attn_interface", module)
    return calls


def branch_kernels(arch, monkeypatch, installed=True):
    if installed:
        install_flash3(monkeypatch)
    else:
        monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
    # Fused Triton kernels are a separate path; these tests only watch branch attention.
    monkeypatch.setattr(cuda, "_optional", lambda module, attribute: None)
    return cuda.CUDAAccelerator().kernels(device(arch))


def sources(kernels, name):
    return [kernel.source for kernel in kernels.available.get(name, [])]


@pytest.fixture
def flash_op():
    if not hasattr(torch.ops.aten, "_flash_attention_forward"):
        pytest.skip("this PyTorch has no flash-attention op to register against")


def test_capability_major_reads_hopper_ampere_and_blackwell():
    assert flash3.capability_major("sm_90") == 9
    assert flash3.capability_major("sm_89") == 8
    assert flash3.capability_major("sm_100") == 10
    assert flash3.capability_major("gfx942") is None
    assert flash3.capability_major(None) is None


def test_only_hopper_with_the_import_can_run_flash_attention_3(monkeypatch):
    install_flash3(monkeypatch)
    assert flash3.unavailable(device("sm_90")) is None
    assert "Hopper" in flash3.unavailable(device("sm_80"))
    monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
    assert (
        flash3.unavailable(device("sm_90")) == "flash_attn_interface is not installed"
    )
    assert "not a CUDA device" in flash3.unavailable(device("sm_90", accelerator="cpu"))


def test_hopper_registers_flash_attention_3_without_selecting_it(flash_op, monkeypatch):
    kernels = branch_kernels("sm_90", monkeypatch)
    assert flash3.SOURCE in sources(kernels, "branch_attention")
    assert kernels.select("branch_attention").source == cuda.FA2_SOURCE
    kernels.allow_approximate = True
    assert kernels.select("branch_attention").source == cuda.FA2_SOURCE
    assert kernels.describe()["branch_attention"] == cuda.FA2_SOURCE
    kernels.use_source("branch_attention", flash3.SOURCE)
    assert kernels.select("branch_attention").source == flash3.SOURCE
    assert kernels.describe()["branch_attention"] == flash3.SOURCE
    assert _varlen_kernel(kernels) is kernels.select("branch_attention").fn


def test_older_gpus_and_a_missing_import_do_not_register_flash_attention_3(
    flash_op, monkeypatch
):
    ampere = branch_kernels("sm_80", monkeypatch)
    assert flash3.SOURCE not in sources(ampere, "branch_attention")
    assert ampere.select("branch_attention").source == cuda.FA2_SOURCE
    blackwell = branch_kernels("sm_100", monkeypatch)
    assert flash3.SOURCE not in sources(blackwell, "branch_attention")
    missing = branch_kernels("sm_90", monkeypatch, installed=False)
    assert flash3.SOURCE not in sources(missing, "branch_attention")


def test_the_wrapper_keeps_gqa_and_sets_num_splits_only_for_non_causal(monkeypatch):
    calls = install_flash3(monkeypatch)
    fallback_calls = []

    def fallback(*args):
        fallback_calls.append(args)
        return args[0]

    attend = flash3.varlen(fallback)
    query = torch.zeros(4, 8, flash3.HEAD_DIM, dtype=torch.bfloat16)
    key = torch.zeros(6, 2, flash3.HEAD_DIM, dtype=torch.bfloat16)
    value = key.clone()
    cu_q = torch.tensor([0, 4], dtype=torch.int32)
    cu_k = torch.tensor([0, 6], dtype=torch.int32)
    got = attend(query, key, value, cu_q, cu_k, 4, 6, False, 0.25)
    assert got is query and not fallback_calls
    assert calls[-1]["k"].shape[1] == 2
    assert calls[-1]["num_splits"] == 1
    assert calls[-1]["causal"] is False
    attend(query, key, value, cu_q, cu_k, 4, 6, True, 0.25)
    assert calls[-1]["num_splits"] is None
    assert calls[-1]["causal"] is True
    # Head sizes 64-256 call FlashAttention-3. Outside that, and FP32, the
    # fallback runs. The registered kernel's fallback raises (see below).
    for width in (flash3.HEAD_DIM_MIN, 128):
        supported = torch.zeros(4, 8, width, dtype=torch.bfloat16)
        attend(supported, supported, supported, cu_q, cu_q, 4, 4, False, 0.25)
        assert calls[-1]["q"] is supported
    assert fallback_calls == []
    outside = torch.zeros(4, 8, 32, dtype=torch.bfloat16)
    attend(outside, outside, outside, cu_q, cu_q, 4, 4, False, 0.25)
    assert fallback_calls[-1][0] is outside
    fp32 = query.float()
    before = len(calls)
    attend(fp32, key.float(), value.float(), cu_q, cu_k, 4, 6, False, 0.25)
    assert len(calls) == before
    assert fallback_calls[-1][0] is fp32


def test_opt_in_refuses_a_head_size_outside_64_to_256(monkeypatch):
    install_flash3(monkeypatch)
    small = torch.nn.Module()
    small.head_dim = 16
    reason = flash3.opt_in_refusal(device("sm_90"), small)
    assert reason is not None and "16" in reason
    for width in (flash3.HEAD_DIM_MIN, 128, flash3.HEAD_DIM_MAX):
        module = torch.nn.Module()
        module.head_dim = width
        assert flash3.opt_in_refusal(device("sm_90"), module) is None
    over = torch.nn.Module()
    over.head_dim = flash3.HEAD_DIM_MAX + 8
    assert flash3.opt_in_refusal(device("sm_90"), over) is not None


def test_the_registered_kernel_does_not_fall_back_to_flash_attention_2(
    flash_op, monkeypatch
):
    kernels = branch_kernels("sm_90", monkeypatch)
    kernels.use_source("branch_attention", flash3.SOURCE)
    attend = kernels.select("branch_attention").fn
    cu = torch.tensor([0, 2], dtype=torch.int32)
    supported = torch.zeros(2, 1, 128, dtype=torch.bfloat16)
    assert (
        attend(supported, supported, supported, cu, cu, 2, 2, False, 0.25) is supported
    )
    outside = torch.zeros(2, 1, 32, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="does not fall back"):
        attend(outside, outside, outside, cu, cu, 2, 2, False, 0.25)
    missing_scale = torch.zeros(2, 1, flash3.HEAD_DIM, dtype=torch.bfloat16)

    def no_scale(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
        return q

    module = types.ModuleType("flash_attn_interface")
    module.flash_attn_varlen_func = no_scale
    monkeypatch.setitem(sys.modules, "flash_attn_interface", module)
    refused = flash3.varlen(flash3.refuse)
    with pytest.raises(RuntimeError, match="does not fall back"):
        refused(
            missing_scale,
            missing_scale,
            missing_scale,
            cu,
            cu,
            2,
            2,
            False,
            0.25,
        )


def test_opt_in_does_not_select_another_approximate_kernel():
    kernels = KernelSet(device="cuda:0")
    kernels.register(Kernel("branch_attention", _flash, cuda.FA2_SOURCE, exact=True))
    kernels.register(Kernel("add_rmsnorm", lambda x: x, "triton-gfx942", exact=False))
    kernels.register(
        Kernel(
            "branch_attention",
            lambda *args: args[0],
            flash3.SOURCE,
            exact=False,
            variant=flash3.SOURCE,
        )
    )
    kernels.use_source("branch_attention", flash3.SOURCE)
    kernels.allow_approximate = False
    assert kernels.select("branch_attention").source == flash3.SOURCE
    assert not kernels.has("add_rmsnorm")


def test_flash_attention_3_refuses_to_load_off_hopper(tmp_path):
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
    save(
        {f"model.{name}": value for name, value in state.items()},
        tmp_path / "model.safetensors",
    )
    spec = ModelSpec(
        "tiny",
        BackboneSpec("modernbert", config, (tmp_path / "model.safetensors",), "model."),
        DtypePolicy(autocast=None, bf16_resident=False),
        max_input_tokens=4096,
        encoder=True,
    )
    accelerator = CPUAccelerator()
    with pytest.raises(RuntimeError, match=r"flash_attention_3: .*not a CUDA device"):
        NativeEngine().load(
            spec,
            accelerator,
            accelerator.devices()[0],
            EngineOptions(flash_attention_3=True),
        )


@pytest.mark.gpu
def test_flash_attention_3_matches_the_reference_on_hopper():
    if torch.version.hip or not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major != 9:
        pytest.skip(f"needs Hopper, not compute capability {major}.{minor}")
    if flash3.load() is None:
        pytest.skip("flash_attn_interface is not installed")
    from vllm_srun.engines.native.models.branch import reference_attention

    torch.manual_seed(0)
    heads, kv_heads = 8, 2
    lengths = [16, 24]
    total = sum(lengths)
    device = torch.device("cuda")
    bounds = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    cu = torch.tensor(bounds, device=device, dtype=torch.int32)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("FlashAttention-3 fell back")

    attend = flash3.varlen(fail_if_called)
    for dim in (flash3.HEAD_DIM_MIN, 128, flash3.HEAD_DIM_MAX):
        query = torch.randn(total, heads, dim, device=device, dtype=torch.bfloat16)
        key = torch.randn(total, kv_heads, dim, device=device, dtype=torch.bfloat16)
        value = torch.randn(total, kv_heads, dim, device=device, dtype=torch.bfloat16)
        scale = dim**-0.5
        for causal in (False, True):
            got = attend(
                query, key, value, cu, cu, max(lengths), max(lengths), causal, scale
            )
            ref = reference_attention(
                query,
                key,
                value,
                bounds,
                bounds,
                causal=causal,
                scale=scale,
                groups=heads // kv_heads,
            )
            assert got.shape == ref.shape
            assert torch.allclose(got.float(), ref.float(), atol=5e-2, rtol=5e-2)
