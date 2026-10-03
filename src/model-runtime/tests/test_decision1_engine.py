"""The engine pieces Decision 1.0 brought: branched encoders, BF16-resident GPU weights, kernel variants, typed heads."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from vllm_sr_runtime.accel.cpu import CPUAccelerator
from vllm_sr_runtime.accel.kernels import (
    ADDITIVE_MASKS,
    AdditiveMaskSDPA,
    Kernel,
    reference_kernels,
    sdpa_ref,
)
from vllm_sr_runtime.engines.native import models
from vllm_sr_runtime.engines.native.engine import NativeEngine
from vllm_sr_runtime.engines.native.weights import cast_parameters, load_backbone
from vllm_sr_runtime.families.decision1 import package as pkg
from vllm_sr_runtime.families.decision1.family import Decision1Family
from vllm_sr_runtime.heads.typed import TypeHeadLayer
from vllm_sr_runtime.plugins.base import EncoderBatch, EngineOptions, PackageRef
from vllm_sr_runtime.testing.decision1 import write_package
from vllm_sr_runtime.testing.fixtures import modernbert_config, random_backbone, save


def test_variants_run_only_for_models_that_name_them():
    kernels = reference_kernels("cpu")
    assert kernels.select("sdpa").fn is sdpa_ref
    kernels.use_variants({"sdpa": ADDITIVE_MASKS, "causal_conv1d": "missing"})
    assert isinstance(kernels.select("sdpa").fn, AdditiveMaskSDPA)
    assert kernels.select("causal_conv1d").variant is None
    kernels.register(
        Kernel("sdpa", sdpa_ref, "approximate", exact=False, variant=ADDITIVE_MASKS)
    )
    assert isinstance(kernels.select("sdpa").fn, AdditiveMaskSDPA)
    kernels.allow_approximate = True
    assert kernels.select("sdpa").source == "approximate"
    assert kernels.has("sdpa") and not kernels.has("gdn_prep")


def test_additive_masks_reproduce_boolean_and_missing_masks():
    torch.manual_seed(0)
    query, key, value = (torch.randn(2, 3, 6, 8) for _ in range(3))
    allowed = torch.ones(2, 1, 6, 6, dtype=torch.bool)
    allowed[1, :, :, 4:] = False
    additive = torch.zeros(2, 1, 6, 6).masked_fill(
        ~allowed, torch.finfo(torch.float32).min
    )
    arguments = {"scale": 8**-0.5, "is_causal": False, "enable_gqa": False}
    sdpa = AdditiveMaskSDPA()
    for _ in range(2):
        assert torch.equal(
            sdpa(query, key, value, allowed, **arguments),
            sdpa_ref(query, key, value, additive, **arguments),
        )
        assert torch.equal(
            sdpa(query, key, value, None, **arguments),
            sdpa_ref(query, key, value, torch.zeros(2, 1, 6, 6), **arguments),
        )
    assert sdpa.additive(query, key, allowed) is sdpa.additive(query, key, allowed)
    other = allowed.clone()
    other[0, :, :, 5:] = False
    assert torch.equal(
        sdpa.additive(query, key, other)[0, 0, 0],
        torch.tensor([0.0] * 5 + [torch.finfo(torch.float32).min]),
    )


def test_cast_parameters_keeps_buffers():
    module = nn.Linear(4, 4)
    module.register_buffer("frequencies", torch.arange(3, dtype=torch.float32))
    cast_parameters(module, torch.bfloat16)
    assert (
        module.weight.dtype == torch.bfloat16
        and module.frequencies.dtype == torch.float32
    )


def test_load_backbone_renames_prefixes(tmp_path):
    save(
        {"block.weight": torch.ones(2, 2), "block.bias": torch.zeros(2)},
        tmp_path / "w.safetensors",
    )
    holder = nn.Module()
    with torch.device("meta"):
        holder.layer = nn.Linear(2, 2)
    load_backbone(holder, [tmp_path / "w.safetensors"], renames={"block.": "layer."})
    assert torch.equal(holder.layer.weight, torch.ones(2, 2))


def test_branches_share_the_embedding_and_run_their_own_stack(tmp_path):
    root = write_package(tmp_path / "vela", runtime=pkg.VELA, seed=5)
    family = Decision1Family()
    package = family.verify(PackageRef(root))
    spec = family.describe(package)
    accelerator = CPUAccelerator()
    engine = NativeEngine().load(
        spec, accelerator, accelerator.devices()[0], EngineOptions()
    )
    choice = engine.branches["choice"]
    assert choice.embeddings is engine.backbone.embeddings
    assert set(engine.branches) == {"choice", "score"}
    modules = (engine.backbone, *engine.branches.values())
    unique = {id(p): p for module in modules for p in module.parameters()}
    assert engine.parameter_count() == sum(p.numel() for p in unique.values())
    assert engine.parameter_count() < sum(
        p.numel() for m in modules for p in m.parameters()
    )
    config = modernbert_config(spec.backbone.config["vocab_size"])
    alone = models.build("modernbert", config)
    branch = random_backbone("modernbert", config, 5 + 1)
    base = random_backbone("modernbert", config, 5)
    alone.load_state_dict(
        {
            **{
                name: value.float()
                for name, value in base.items()
                if name.startswith("embeddings.")
            },
            **{
                name: value.float()
                for name, value in branch.items()
                if not name.startswith("embeddings.")
            },
        }
    )
    alone.kernels = engine.kernels
    ids = torch.tensor([[2, 5, 9, 11, 7, 1], [2, 8, 1, 0, 0, 0]])
    mask = ids != 0
    out = engine.encode(
        EncoderBatch(input_ids=ids, attention_mask=mask, branch="choice")
    )
    with torch.inference_mode():
        expected = alone(ids, mask)
    assert torch.equal(out.hidden[config["num_hidden_layers"]], expected)


def test_gpu_weights_leave_cpu_models_in_fp32(tmp_path):
    root = write_package(tmp_path / "qwen", seed=6)
    family = Decision1Family()
    package = family.verify(PackageRef(root))
    spec = family.describe(package)
    assert spec.dtype.gpu_weights == "bfloat16" and not spec.dtype.bf16_resident
    accelerator = CPUAccelerator()
    engine = NativeEngine().load(
        spec, accelerator, accelerator.devices()[0], EngineOptions()
    )
    assert {p.dtype for p in engine.backbone.parameters()} == {torch.float32}


def test_named_kernel_variants_per_runtime(tmp_path):
    family = Decision1Family()
    vela = family.describe(
        family.verify(PackageRef(write_package(tmp_path / "v", runtime=pkg.VELA)))
    )
    assert vela.kernel_variants == {"sdpa": ADDITIVE_MASKS} and vela.encoder
    eos = write_package(tmp_path / "e", model_name="Decision-1.0-Eos-0.8B")
    assert family.describe(family.verify(PackageRef(eos))).kernel_variants == {
        "causal_conv1d": "fp64_accumulate"
    }


def test_type_head_layer_is_the_reference_encoder_layer():
    torch.manual_seed(0)
    reference = nn.TransformerEncoderLayer(
        32, 4, 128, 0.1, activation="relu", batch_first=True, norm_first=True
    ).eval()
    layer = TypeHeadLayer(32, 4, 128).eval()
    layer.load_state_dict(reference.state_dict())
    hidden = torch.randn(3, 7, 32)
    padding = torch.zeros(3, 7, dtype=torch.bool)
    padding[1, 5:] = True
    padding[2, 2:] = True
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.inference_mode():
            expected = reference(hidden, src_key_padding_mask=padding)
            key_padding = torch.zeros(3, 7).masked_fill(padding, float("-inf"))
            assert torch.equal(layer(hidden, key_padding), expected)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def test_fp64_convolution_falls_back_off_its_shapes():
    pytest.importorskip("triton")
    from vllm_sr_runtime.accel.triton_fp64_conv import fp64_conv

    calls = []

    def default(hidden_states, weight, bias=None, activation=None):
        calls.append(hidden_states.shape)
        return hidden_states

    conv = fp64_conv(default)
    x = torch.randn(1, 8, 16)
    assert conv(x, torch.randn(8, 4), None, activation="silu") is x
    assert calls == [x.shape]
