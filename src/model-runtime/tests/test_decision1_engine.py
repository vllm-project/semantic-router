"""The engine pieces Decision 1.0 brought: branched encoders, BF16-resident GPU weights, kernel variants, typed heads."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.accel.kernels import (
    ADDITIVE_MASKS,
    AdditiveMaskSDPA,
    Kernel,
    reference_kernels,
    sdpa_ref,
)
from vllm_srun.engines.native import models
from vllm_srun.engines.native import reduced as copies
from vllm_srun.engines.native.engine import NativeEngine
from vllm_srun.engines.native.weights import cast_parameters, load_backbone
from vllm_srun.families.decision1 import package as pkg
from vllm_srun.families.decision1.family import Decision1Family
from vllm_srun.heads.typed import TypeHeadLayer, TypeReadout
from vllm_srun.plugins.base import EncoderBatch, EngineOptions, PackageRef
from vllm_srun.registry import builtin
from vllm_srun.testing.decision1 import write_package
from vllm_srun.testing.fixtures import modernbert_config, random_backbone, save


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


def test_loaded_weights_live_in_process_memory_not_the_checkpoint_mapping(tmp_path):
    maps = Path("/proc/self/maps")
    if not maps.exists():
        pytest.skip("needs /proc/self/maps")
    path = tmp_path / "w.safetensors"
    save({"block.weight": torch.ones(512, 512)}, path)
    holder = nn.Module()
    with torch.device("meta"):
        holder.layer = nn.Linear(512, 512, bias=False)
    load_backbone(holder, [path], renames={"block.": "layer."})
    pointer = holder.layer.weight.data_ptr()
    for line in maps.read_text().splitlines():
        start, end = (int(value, 16) for value in line.split()[0].split("-"))
        if start <= pointer < end:
            assert str(path) not in line


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


def test_reduced_copy_consent_comes_from_the_pinned_package(tmp_path, monkeypatch):
    family = Decision1Family()
    package = family.verify(PackageRef(write_package(tmp_path / "v", runtime=pkg.VELA)))
    dtype = family.describe(package).dtype
    assert (dtype.reduced_gpu, dtype.reduced_cpu) == (None, None)
    lex = builtin.lookup("Decision-1.0-Lex-0.6B")
    monkeypatch.setattr(
        builtin,
        "by_identity",
        lambda digest: lex if digest == package.model_sha256 else None,
    )
    dtype = family.describe(package).dtype
    assert (dtype.reduced_gpu, dtype.reduced_cpu) == (None, "float32-packed")


PACKED_IDS = torch.tensor([2, 5, 9, 11, 7, 1, 2, 8, 1])
PACKED_LENGTHS = [6, 3]


def vela_engine(tmp_path, reduced_cpu, device=None):
    """The tiny Vela-encoder package on CPU under ``max_speed``, consenting to ``reduced_cpu``."""
    family = Decision1Family()
    spec = family.describe(
        family.verify(PackageRef(write_package(tmp_path / "v", runtime=pkg.VELA)))
    )
    spec = replace(spec, dtype=replace(spec.dtype, reduced_cpu=reduced_cpu))
    accelerator = CPUAccelerator()
    options = EngineOptions(reduced_precision=True, exact_kernels_only=False)
    return NativeEngine().load(
        spec, accelerator, device or accelerator.devices()[0], options
    )


def packed_hidden(engine, branch, reduced):
    batch = EncoderBatch(
        PACKED_IDS, None, lengths=PACKED_LENGTHS, branch=branch, reduced=reduced
    )
    return engine.encode(batch).hidden[engine.backbone.num_layers]


def test_reduced_batches_run_the_copy_of_their_stack(tmp_path):
    if copies.unavailable("int8", torch.device("cpu")):
        pytest.skip(copies.unavailable("int8", torch.device("cpu")))
    engine = vela_engine(tmp_path, "int8")
    assert set(engine.reduced) == {None, "choice", "score"}
    assert engine.receipt()["reduced"]["kind"] == "int8"
    for branch, stack in engine.stacks().items():
        view = engine.reduced[branch]
        assert (
            view.embeddings.tok_embeddings.weight
            is stack.embeddings.tok_embeddings.weight
        )
        alone = copies.reduced_view(stack, "int8")
        alone.kernels = engine.kernels
        with torch.inference_mode():
            expected = alone.encode(
                PACKED_IDS, alone.packed(PACKED_LENGTHS, "cpu"), (alone.num_layers,)
            )[alone.num_layers]
        exact = packed_hidden(engine, branch, reduced=False)
        assert torch.equal(packed_hidden(engine, branch, reduced=True), expected)
        assert not torch.equal(expected, exact)
        assert torch.equal(packed_hidden(engine, branch, reduced=False), exact)


@pytest.mark.parametrize("kind", ["bfloat16", "float32-packed"])
def test_a_copy_the_cpu_cannot_run_is_skipped_not_fatal(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(copies.onednn, "available", lambda: False)
    device = replace(CPUAccelerator().devices()[0], bf16=False)
    engine = vela_engine(tmp_path, kind, device)
    assert engine.reduced == {} and engine.memory_bytes() > 0
    receipt = engine.receipt()["reduced"]
    assert receipt["kind"] == kind and receipt["skipped"]
    for branch in engine.stacks():
        assert torch.equal(
            packed_hidden(engine, branch, reduced=True),
            packed_hidden(engine, branch, reduced=False),
        )


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


def test_type_readout_runs_in_fp32_on_reduced_precision_hidden_states():
    torch.manual_seed(0)
    readout = TypeReadout(32, 4, 1).eval()
    hidden = torch.randn(2, 6, 32)
    padding = torch.zeros(2, 6, dtype=torch.bool)
    padding[1, 4:] = True
    markers = torch.tensor([[1, 3], [0, 2]])
    with torch.inference_mode():
        full = readout("choice", hidden, padding, markers)
        reduced = readout("choice", hidden.bfloat16(), padding, markers)
    assert full.dtype == reduced.dtype == torch.float32
    assert torch.allclose(full, reduced, atol=0.05)
