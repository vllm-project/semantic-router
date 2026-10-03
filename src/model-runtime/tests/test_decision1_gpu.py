"""GPU: Decision 1.0's BF16-stream decoders run the fused layers and graphs bit for bit like their eager backbone.

Random Qwen3.5 backbones at Eos-0.8B's and Nox-4B's layer widths (few layers) hold every parameter
in BF16, as the engine holds a ``gpu_weights`` backbone (rotary buffers stay FP32). An untouched copy
is the reference; the other copy runs the fused layers through the graph runner (first use eager,
second captured, third replayed) and must equal it (``torch.equal``). The Eos case also selects its
released FP64 convolution variant, whose gated-delta blocks run eagerly inside the fused layers.
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from vllm_sr_runtime.engines.native import fast, models  # noqa: E402
from vllm_sr_runtime.engines.native.weights import cast_parameters  # noqa: E402

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
