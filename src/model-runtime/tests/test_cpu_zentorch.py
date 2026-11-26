"""zentorch's CPU kernels with the ops faked: the gates, per-slot consent, the linear layout and SDPA."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm_srun.accel import zentorch
from vllm_srun.accel.cpu import CPUAccelerator

ZEN5 = "vendor_id\t: AuthenticAMD\ncpu family\t: 26\nmodel\t\t: 2\n"
ZEN4 = "vendor_id\t: AuthenticAMD\ncpu family\t: 25\nmodel\t\t: 17\n"
INTEL = "vendor_id\t: GenuineIntel\ncpu family\t: 6\nmodel\t\t: 207\n"


def fake_sdpa(query, key, value, dropout_p, is_causal, *, attn_mask=None, scale=None):
    """Like ``zentorch_sdpa``: grouped heads natively, additive masks only, a tuple back."""
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        raise RuntimeError("zentorch_sdpa: boolean masks are not supported")
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=key.shape[1] != query.shape[1],
    )
    return out, None


def fake_ops() -> SimpleNamespace:
    """Stand-ins: the reorder copies the weight, the product is ``F.linear`` on it."""
    return SimpleNamespace(
        zentorch_weight_prepack_for_linear=lambda weight: weight.clone(),
        zentorch_linear_unary=lambda x, weight, bias, is_weight_prepacked: F.linear(
            x, weight, bias
        ),
        zentorch_sdpa=fake_sdpa,
    )


@pytest.fixture
def gates(monkeypatch):
    """zentorch usable on a measured CPU, with the ops faked; the opt-in left to each test."""
    monkeypatch.setattr(zentorch, "_ops", fake_ops)
    monkeypatch.setattr(zentorch, "_CACHE", {"available": True})
    monkeypatch.setattr(zentorch, "measured", lambda: True)
    monkeypatch.delenv("VLLM_SRUN_ZENTORCH", raising=False)


def linear_sources(device=None):
    accelerator = CPUAccelerator()
    kernels = accelerator.kernels(device or accelerator.devices()[0])
    return [(k.source, k.exact, k.variant) for k in kernels.available["linear"]]


def test_off_without_the_opt_in(gates):
    assert ("zentorch", False, None) not in linear_sources()


def test_registers_approximate_without_a_variant_when_opted_in(gates, monkeypatch):
    monkeypatch.setenv("VLLM_SRUN_ZENTORCH", "1")
    assert linear_sources()[0] == ("zentorch", False, None)


def test_unmeasured_cpu_keeps_the_reference(gates, monkeypatch):
    monkeypatch.setenv("VLLM_SRUN_ZENTORCH", "1")
    monkeypatch.setattr(zentorch, "measured", lambda: False)
    assert ("zentorch", False, None) not in linear_sources()


@pytest.mark.parametrize(
    ("cpuinfo", "family"),
    [
        (ZEN5, ("AuthenticAMD", 26)),
        (ZEN4, ("AuthenticAMD", 25)),
        (INTEL, ("GenuineIntel", 6)),
    ],
)
def test_cpu_family_reads_vendor_and_family(tmp_path, cpuinfo, family):
    path = tmp_path / "cpuinfo"
    path.write_text(cpuinfo)
    assert zentorch.cpu_family(str(path)) == family
    assert (family in zentorch.MEASURED) is (cpuinfo == ZEN5)


def test_unreadable_cpuinfo_is_not_measured(tmp_path):
    assert zentorch.cpu_family(str(tmp_path / "missing")) is None


def test_zentorch_for_another_pytorch_is_not_used(monkeypatch):
    module = ModuleType("zentorch")
    module.__version__ = "1.0.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "zentorch", module)
    monkeypatch.setattr(zentorch, "_CACHE", {})
    assert zentorch.available() is False


def test_zentorch_without_the_ops_is_not_used(monkeypatch):
    module = ModuleType("zentorch")
    running = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    module.__version__ = f"{running}.0.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "zentorch", module)
    monkeypatch.setattr(zentorch, "_ops", SimpleNamespace)
    monkeypatch.setattr(zentorch, "_CACHE", {})
    monkeypatch.setattr(zentorch, "measured", lambda: True)
    monkeypatch.setenv("VLLM_SRUN_ZENTORCH", "1")
    assert zentorch.available() is True
    assert zentorch.enabled() == frozenset()


def test_layout_holds_one_copy_and_matches_the_reference(gates):
    torch.manual_seed(0)
    linear = nn.Linear(64, 48)
    laid = zentorch.lay_out(linear)
    assert isinstance(laid, zentorch.ZentorchLinear)
    assert not any(name == "weight" for name, _ in laid.named_parameters())
    x = torch.randn(3, 5, 64)
    torch.testing.assert_close(laid(x), F.linear(x, linear.weight, linear.bias))


def test_layout_takes_bf16_layers(gates):
    torch.manual_seed(0)
    linear = nn.Linear(64, 48).to(torch.bfloat16)
    laid = zentorch.lay_out(linear)
    assert isinstance(laid, zentorch.ZentorchLinear)
    x = torch.randn(3, 5, 64).to(torch.bfloat16)
    torch.testing.assert_close(laid(x), F.linear(x, linear.weight, linear.bias))


def test_layout_leaves_other_dtypes_alone(gates):
    linear = nn.Linear(16, 8).to(torch.float16)
    assert zentorch.lay_out(linear) is linear


def _load_tiny(tmp_path, policy, exact_kernels_only=False):
    from vllm_srun.engines.native.engine import NativeEngine
    from vllm_srun.plugins.base import BackboneSpec, EngineOptions, ModelSpec
    from vllm_srun.testing.fixtures import random_backbone, save

    from .test_modernbert import CONFIGS

    config = CONFIGS["yarn"]
    state = random_backbone("modernbert", config, seed=7)
    save({f"model.{n}": v for n, v in state.items()}, tmp_path / "model.safetensors")
    spec = ModelSpec(
        "tiny",
        BackboneSpec("modernbert", config, (tmp_path / "model.safetensors",), "model."),
        policy,
        max_input_tokens=4096,
        encoder=True,
    )
    accelerator = CPUAccelerator()
    return NativeEngine().load(
        spec,
        accelerator,
        accelerator.devices()[0],
        EngineOptions(exact_kernels_only=exact_kernels_only),
    )


@pytest.mark.parametrize(
    ("gpu", "cpu", "exact_only", "laid_out"),
    [
        (False, True, False, True),  # CPU consent, --dtype auto: zentorch
        (True, False, False, False),  # a GPU record's consent: the reference
        (False, True, True, False),  # an explicit --dtype: exact kernels only
        (False, False, False, False),  # no consent
    ],
)
def test_only_cpu_consent_lays_out_zentorch(
    tmp_path, gates, monkeypatch, gpu, cpu, exact_only, laid_out
):
    from vllm_srun.plugins.base import DtypePolicy

    monkeypatch.setenv("VLLM_SRUN_ZENTORCH", "1")
    policy = DtypePolicy(
        autocast=None,
        bf16_resident=False,
        approximate_kernels=gpu,
        approximate_cpu_kernels=cpu,
    )
    model = _load_tiny(tmp_path, policy, exact_only)
    modules = list(model.backbone.modules())
    assert any(isinstance(m, zentorch.ZentorchLinear) for m in modules) is laid_out
    assert any(isinstance(m, nn.Linear) for m in modules) is not laid_out
    assert model.approximate_kernels is laid_out


def test_lora_adapters_run_over_laid_out_factors(gates):
    """vela3's adapters: LoRALinear reads its dtype from packed factors, which hold no ``weight``."""
    from vllm_srun.engines.native.models.lora import LoRALinear
    from vllm_srun.engines.native.weights import lay_out_linears

    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(32, 24), rank=4, scaling=2.0)
    nn.init.normal_(layer.lora_B.weight)
    x = torch.randn(2, 7, 32)
    expected = layer(x)
    holder = nn.Sequential(layer)
    lay_out_linears(holder, zentorch.lay_out)
    assert isinstance(layer.lora_A, zentorch.ZentorchLinear)
    torch.testing.assert_close(layer(x), expected)


def test_laid_out_model_counts_its_weights(tmp_path, gates, monkeypatch):
    """The load check compares parameter counts with the pinned revision: packed weights still count."""
    from vllm_srun.plugins.base import DtypePolicy

    reference = _load_tiny(tmp_path, DtypePolicy(autocast=None, bf16_resident=False))
    monkeypatch.setenv("VLLM_SRUN_ZENTORCH", "1")
    laid = _load_tiny(
        tmp_path,
        DtypePolicy(autocast=None, bf16_resident=False, approximate_cpu_kernels=True),
    )
    assert any(isinstance(m, zentorch.ZentorchLinear) for m in laid.backbone.modules())
    assert laid.parameter_count() == reference.parameter_count()
    assert laid.memory_bytes() == reference.memory_bytes()


def test_layers_shared_by_branches_count_once(tmp_path, gates, monkeypatch):
    """vela3's adapter networks share the backbone's layers: each laid-out layer counts once."""
    from vllm_srun.plugins.base import DtypePolicy

    monkeypatch.setenv("VLLM_SRUN_ZENTORCH", "1")
    laid = _load_tiny(
        tmp_path,
        DtypePolicy(autocast=None, bf16_resident=False, approximate_cpu_kernels=True),
    )
    alone = laid.parameter_count()
    laid.branches["shared"] = laid.backbone
    assert laid.parameter_count() == alone


def test_linears_sharing_a_weight_are_laid_out_once(gates):
    """vela3's adapter branches are their own modules over the backbone's tensors: one packed copy."""
    from vllm_srun.engines.native.weights import lay_out_linears

    base = nn.Linear(8, 8)
    twin = nn.Linear(8, 8)
    twin.weight, twin.bias = base.weight, base.bias
    other = nn.Linear(8, 8)
    first, second = nn.Sequential(base), nn.Sequential(nn.Sequential(twin), other)
    memo: dict = {}
    lay_out_linears(first, zentorch.lay_out, memo)
    lay_out_linears(second, zentorch.lay_out, memo)
    assert first[0] is second[0][0]
    assert second[1] is not first[0]


def sources(slot):
    accelerator = CPUAccelerator()
    kernels = accelerator.kernels(accelerator.devices()[0])
    return [(k.source, k.exact, k.variant) for k in kernels.available[slot]]


def test_each_slot_registers_only_with_its_ops(gates, monkeypatch):
    monkeypatch.setenv("VLLM_SRUN_ZENTORCH", "1")
    assert sources("sdpa")[0] == ("zentorch", False, None)
    linear_only = fake_ops()
    del linear_only.zentorch_sdpa
    monkeypatch.setattr(zentorch, "_ops", lambda: linear_only)
    monkeypatch.setattr(zentorch, "_CACHE", {"available": True})
    assert ("zentorch", False, None) not in sources("sdpa")
    assert sources("linear")[0] == ("zentorch", False, None)


def _qkv(heads=8, kv_heads=8, rows=3, width=40, dim=32):
    torch.manual_seed(0)
    return (
        torch.randn(rows, heads, width, dim),
        torch.randn(rows, kv_heads, width, dim),
        torch.randn(rows, kv_heads, width, dim),
    )


@pytest.mark.parametrize("shape", ["key", "square"])
def test_sdpa_turns_a_boolean_mask_additive(gates, shape):
    query, key, value = _qkv()
    keep = torch.ones(3, 1, 1, 40, dtype=torch.bool)
    keep[1:, ..., 30:] = False
    mask = keep if shape == "key" else keep.expand(3, 1, 40, 40)
    expected = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=0.1
    )
    got = zentorch.sdpa(
        query, key, value, mask, scale=0.1, is_causal=False, enable_gqa=False
    )
    torch.testing.assert_close(got, expected)


def test_sdpa_takes_query_slices_and_grouped_heads(gates):
    query, key, value = _qkv(heads=8, kv_heads=2)
    part = query[:, :, 8:24]  # ultrabert's local attention passes a strided slice
    expected = F.scaled_dot_product_attention(
        part, key, value, scale=0.2, enable_gqa=True
    )
    got = zentorch.sdpa(
        part, key, value, None, scale=0.2, is_causal=False, enable_gqa=True
    )
    torch.testing.assert_close(got, expected)


def test_sdpa_falls_back_to_the_reference(gates, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("unsupported shape")

    monkeypatch.setattr(zentorch, "_ops", lambda: SimpleNamespace(zentorch_sdpa=broken))
    monkeypatch.setattr(zentorch, "_SDPA_FALLBACKS", set())
    query, key, value = _qkv()
    for _ in range(2):
        got = zentorch.sdpa(
            query, key, value, None, scale=0.1, is_causal=True, enable_gqa=False
        )
    expected = F.scaled_dot_product_attention(
        query, key, value, scale=0.1, is_causal=True
    )
    torch.testing.assert_close(got, expected)
    assert sum("fell back" in r.message for r in caplog.records) == 1


@pytest.mark.parametrize(
    ("consent", "linear", "sdpa"),
    [
        (frozenset({"linear"}), "zentorch", "torch-reference"),
        (frozenset({"linear", "sdpa"}), "zentorch", "zentorch"),
        (True, "zentorch", "zentorch"),
        (False, "torch-reference", "torch-reference"),
    ],
)
def test_a_family_consents_slot_by_slot(
    tmp_path, gates, monkeypatch, consent, linear, sdpa
):
    """A family gets zentorch in the slots it names; vela3 and Decision 3.0 name the linear."""
    from vllm_srun.plugins.base import DtypePolicy

    monkeypatch.setenv("VLLM_SRUN_ZENTORCH", "1")
    model = _load_tiny(
        tmp_path,
        DtypePolicy(
            autocast=None, bf16_resident=False, approximate_cpu_kernels=consent
        ),
    )
    assert model.kernels.select("linear").source == linear
    assert model.kernels.select("sdpa").source == sdpa
