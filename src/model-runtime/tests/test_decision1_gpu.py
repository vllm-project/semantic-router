"""GPU: Decision 1.0's BF16-stream decoders run the fused layers and graphs bit for bit like their eager backbone.

Random Qwen3.5 backbones at Eos-0.8B's and Nox-4B's layer widths (few layers) hold every parameter
in BF16, as the engine holds a ``gpu_weights`` backbone (rotary buffers stay FP32). An untouched copy
is the reference; the other copy runs the fused layers through the graph runner (first use eager,
second captured, third replayed) and must equal it (``torch.equal``). The Eos case also selects its
released FP64 convolution variant, whose gated-delta blocks run eagerly inside the fused layers,
and which hands shapes it does not cover to the default kernel. The Vela encoders' three layer
stacks each replay their own bucket graphs.
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from vllm_sr_runtime.engines.native import fast, models  # noqa: E402
from vllm_sr_runtime.engines.native.engine import NativeEngine  # noqa: E402
from vllm_sr_runtime.engines.native.weights import cast_parameters  # noqa: E402
from vllm_sr_runtime.families.decision1 import package as pkg  # noqa: E402
from vllm_sr_runtime.families.decision1.family import Decision1Family  # noqa: E402
from vllm_sr_runtime.plugins.base import (  # noqa: E402
    EncoderBatch,
    EngineOptions,
    PackageRef,
)
from vllm_sr_runtime.testing.decision1 import write_package  # noqa: E402

from .test_gpu_fast_path import LENGTHS, _device, batch, qwen3_5  # noqa: E402

pytestmark = pytest.mark.gpu

CASES = {
    "eos-dims": (lambda: qwen3_5(4, 1024, 16, 8), {}),
    "nox-dims": (lambda: qwen3_5(4, 2560, 32, 16), {}),
    "eos-fp64-conv": (
        lambda: qwen3_5(4, 1024, 16, 8),
        {"causal_conv1d": "fp64_accumulate"},
    ),
}


def build(config: dict, variants: dict, accelerator, device):
    torch.manual_seed(7)
    module = models.build(config["model_type"], config).float()
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.endswith("A_log"):
                parameter.copy_(torch.rand_like(parameter) * 2)
            elif "norm" in name:
                parameter.copy_(torch.randn_like(parameter) * 0.1)
            else:
                parameter.copy_(torch.randn_like(parameter) * 0.02)
    cast_parameters(module, torch.bfloat16)
    module = module.to(accelerator.torch_device(device)).eval()
    module.kernels = accelerator.kernels(device)
    module.kernels.use_variants(variants)
    return module


@pytest.mark.parametrize("case", sorted(CASES))
def test_bf16_stream_fast_path_equals_eager(case):
    accelerator, device = _device()
    make, variants = CASES[case]
    reference = build(make(), variants, accelerator, device)
    assert reference.rotary_emb.inv_freq.dtype == torch.float32
    fused = copy.deepcopy(reference)
    fused.kernels = reference.kernels
    assert fast.fused_unavailable(fused, fused.kernels) is None
    assert fast.install_fused(fused) == len(fused.layers)
    graphs = fast.Graphs(fused, fast.Masks())
    generator = torch.Generator().manual_seed(11)
    torch_device = accelerator.torch_device(device)
    for lengths in LENGTHS:
        ids, mask = batch(lengths, generator, torch_device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            want = reference(ids, mask).clone()
            assert want.dtype == torch.bfloat16
            for _ in range(3):
                got = graphs(ids, mask, lengths).clone()
                assert torch.equal(want, got), (case, lengths)
    assert not any(layer._fused_failed for layer in fused.layers)
    stats = graphs.receipt()
    assert stats["failed"] == 0 and stats["replays"] > 0


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


def test_every_encoder_stack_replays_its_own_graphs(tmp_path):
    accelerator, device = _device()
    family = Decision1Family()
    spec = family.describe(
        family.verify(PackageRef(write_package(tmp_path / "v", runtime=pkg.VELA)))
    )
    engine = NativeEngine().load(spec, accelerator, device, EngineOptions())
    eager = NativeEngine().load(spec, accelerator, device, EngineOptions(graphs=False))
    ids, lengths = torch.tensor([2, 5, 9, 11, 7, 1, 2, 8, 1]), [6, 3]
    last = engine.backbone.num_layers
    stacks = {}
    for branch in engine.stacks():
        packed = EncoderBatch(ids, None, lengths=lengths, branch=branch)
        runs = [engine.encode(packed).hidden[last].clone() for _ in range(4)]
        assert all(torch.equal(runs[0], run) for run in runs[1:]), branch
        torch.testing.assert_close(
            runs[0], eager.encode(packed).hidden[last], rtol=0, atol=1e-4
        )
        stacks[branch] = runs[0]
    assert not torch.equal(stacks["choice"], stacks[None])
    receipt = engine.receipt()["encoder_graphs"]
    for stats in (receipt, *receipt["branches"].values()):
        assert (stats["captures"], stats["replays"], stats["failed"]) == (1, 3, 0)
    assert set(receipt["branches"]) == {"choice", "score"}
