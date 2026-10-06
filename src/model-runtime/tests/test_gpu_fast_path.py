"""GPU: the native engine's fast path reproduces its eager backbone bit for bit (ROCm gfx942).

Random backbones at the released sizes' layer dimensions (few layers), held as the engine holds them
(BF16-resident Linear weights, everything else FP32): Qwen3 dense at Kai-0.6B's dims, Qwen3.5 hybrids at
Eos-0.8B's and Nox-4B's dims, and a Qwen3.5 with an unmerged LoRA (rank 16, scaling 2) on every Linear, as
Vega-27B. An untouched copy is the reference. The fused layers (and the lean LoRA) are installed on the
other copy, and right-padded batches of many shapes run three times through the graph runner (first use
eager, second captured, third replayed); every output must equal the reference's (``torch.equal``).
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from vllm_srun.accel.rocm import ROCmAccelerator  # noqa: E402
from vllm_srun.engines.native import fast, models  # noqa: E402
from vllm_srun.engines.native.models.forest import ForestShape  # noqa: E402
from vllm_srun.engines.native.models.lora import attach  # noqa: E402
from vllm_srun.engines.native.weights import keep_linear_bf16  # noqa: E402

pytestmark = pytest.mark.gpu

LENGTHS = [
    [8],
    [13],
    [64],
    [347],
    [1000],
    [100, 100],
    [37, 200],
    [5, 17, 33],
    [64] * 8,
    [251, 249, 250, 7, 120],
    [512, 512, 300],
    [2100, 2050],
]


def _device():
    accelerator = ROCmAccelerator()
    if not accelerator.available():
        pytest.skip("needs a ROCm GPU")
    device = accelerator.devices()[0]
    if device.arch != "gfx942":
        pytest.skip("the fused kernels are validated on gfx942")
    return accelerator, device


def qwen3(layers: int) -> dict:
    return {
        "model_type": "qwen3",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": layers,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        "vocab_size": 1024,
        "hidden_act": "silu",
        "attention_bias": False,
    }


def qwen3_5(layers: int, hidden: int, value_heads: int, heads: int) -> dict:
    kinds = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    return {
        "model_type": "qwen3_5_text",
        "hidden_size": hidden,
        "intermediate_size": hidden * 3,
        "num_hidden_layers": layers,
        "layer_types": [kinds[i % 4] for i in range(layers)],
        "num_attention_heads": heads,
        "num_key_value_heads": max(1, heads // 4),
        "head_dim": 256,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": value_heads,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
        },
        "vocab_size": 1024,
        "hidden_act": "silu",
        "attention_bias": False,
    }


def build(config: dict, seed: int, lora: bool, accelerator, device):
    torch.manual_seed(seed)
    module = models.build(config["model_type"], config).float()
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.endswith("A_log"):
                parameter.copy_(torch.rand_like(parameter) * 2)
            elif "norm" in name:
                parameter.copy_(
                    torch.randn_like(parameter) * 0.1
                    + (0.0 if config["model_type"] != "qwen3" else 1.0)
                )
            else:
                parameter.copy_(torch.randn_like(parameter) * 0.02)
            if parameter.ndim == 2 and "embed" not in name:
                parameter.copy_(parameter.to(torch.bfloat16).float())
    if lora:
        targets = [
            name
            for name, layer in module.named_modules()
            if isinstance(layer, torch.nn.Linear)
        ]
        attach(module, targets, 16, 2.0)
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                if "lora_" in name:
                    parameter.copy_(torch.randn_like(parameter) * 0.02)
    keep_linear_bf16(module)
    module = module.to(accelerator.torch_device(device)).eval()
    module.kernels = accelerator.kernels(device)
    return module


def batch(lengths, generator, device):
    width = -(-max(lengths) // 8) * 8
    ids = torch.randint(0, 1024, (len(lengths), width), generator=generator)
    mask = torch.zeros_like(ids)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    return ids.to(device), mask.to(device)


CASES = {
    "qwen3-0.6b-dims": (lambda: qwen3(4), False),
    "qwen3_5-0.8b-dims": (lambda: qwen3_5(4, 1024, 16, 8), False),
    "qwen3_5-4b-dims": (lambda: qwen3_5(4, 2560, 32, 16), False),
    "qwen3_5-lora": (lambda: qwen3_5(4, 1024, 16, 8), True),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_fast_path_equals_eager(case):
    accelerator, device = _device()
    make, lora = CASES[case]
    reference = build(make(), 7, lora, accelerator, device)
    fused = copy.deepcopy(reference)
    fused.kernels = reference.kernels
    assert fast.fused_unavailable(fused, fused.kernels) is None
    assert fast.install_fused(fused) == len(fused.layers)
    if lora:
        assert fast.install_lean_lora(fused)["lean"] > 0
    graphs = fast.Graphs(fused, fast.Masks())
    generator = torch.Generator().manual_seed(11)
    torch_device = accelerator.torch_device(device)
    for lengths in LENGTHS:
        ids, mask = batch(lengths, generator, torch_device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            want = reference(ids, mask).clone()
            for _ in range(3):
                got = graphs(ids, mask, lengths).clone()
                assert torch.equal(want, got), (case, lengths)
    stats = graphs.receipt()
    assert stats["failed"] == 0 and stats["replays"] > 0


def test_full_graph_cache_runs_eager():
    """Past the cache limit new shapes run eagerly; the captured graphs stay and replay."""
    accelerator, device = _device()
    reference = build(qwen3_5(4, 1024, 16, 8), 7, False, accelerator, device)
    fused = copy.deepcopy(reference)
    fused.kernels = reference.kernels
    fast.install_fused(fused)
    graphs = fast.Graphs(fused, fast.Masks(), max_graphs=2)
    torch_device = accelerator.torch_device(device)
    shapes = [[8], [13], [37, 200], [64] * 8]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):
            for seed, lengths in enumerate(shapes):
                generator = torch.Generator().manual_seed(seed)
                ids, mask = batch(lengths, generator, torch_device)
                want = reference(ids, mask).clone()
                assert torch.equal(want, graphs(ids, mask, lengths)), lengths
    stats = graphs.receipt()
    assert stats["cached"] == 2 and stats["full"] > 0 and stats["replays"] > 0


def test_attention_prep_above_2gib():
    """A q projection above 2 GiB (9 x 16,384 tokens at Nox-4B's widths) runs fused, without the eager fallback."""
    accelerator, device = _device()
    config = qwen3_5(1, 2560, 32, 16)
    config["layer_types"] = ["full_attention"]
    reference = build(config, 7, False, accelerator, device)
    fused = copy.deepcopy(reference)
    fused.kernels = reference.kernels
    assert fast.install_fused(fused) == 1
    lengths = [16384] * 9
    assert (
        sum(lengths) * 2 * config["num_attention_heads"] * config["head_dim"] * 2
        > 2 << 30
    )
    ids, mask = batch(
        lengths, torch.Generator().manual_seed(11), accelerator.torch_device(device)
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        want = reference(ids, mask)
        got = fused(ids, mask, masks=fast.Masks().build(mask, False))
    assert not fused.layers[0]._fused_failed
    assert torch.equal(want, got)


FOREST_CASES = [
    # (prefix row lengths, block lengths, block owners)
    ([20], [7], [0]),
    ([347], [12, 180, 33], [0, 0, 0]),
    ([64, 401], [5, 260, 17, 90], [0, 1, 1, 0]),
    ([1000, 990, 12], [300, 300, 1, 64, 128], [2, 1, 0, 0, 1]),
]


def test_fused_forest_equals_eager():
    """The fused forest forward (prefix rows and blocks, ``models/forest.py``) equals the eager one."""
    accelerator, device = _device()
    reference = build(qwen3_5(4, 2560, 32, 16), 7, False, accelerator, device)
    fused = copy.deepcopy(reference)
    fused.kernels = reference.kernels
    assert fast.install_fused(fused) == len(fused.layers)
    generator = torch.Generator().manual_seed(11)
    torch_device = accelerator.torch_device(device)
    for prefix_lengths, block_lengths, owners in FOREST_CASES:
        prefix_ids, prefix_mask = batch(prefix_lengths, generator, torch_device)
        width = prefix_ids.shape[1]
        for row, length in enumerate(prefix_lengths):
            prefix_ids[row] = prefix_ids[row].roll(width - length)
            prefix_mask[row] = prefix_mask[row].roll(width - length)
        block_ids, block_mask = batch(block_lengths, generator, torch_device)
        owner = torch.tensor(owners, device=torch_device)
        shape = ForestShape(
            tuple(width - length for length in prefix_lengths),
            tuple(block_lengths),
            tuple(owners),
        )
        args = (prefix_ids, prefix_mask, block_ids, block_mask, owner, shape)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            want = reference.forward_forest(*args)
            got = fused.forward_forest(*args)
        assert not any(layer._fused_failed for layer in fused.layers)
        for a, b in zip(want, got, strict=True):
            assert torch.equal(a, b), (prefix_lengths, block_lengths)
