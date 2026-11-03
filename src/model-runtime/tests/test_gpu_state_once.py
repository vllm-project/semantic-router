"""GPU: the state-once branched forward on random backbones at d3-edge's dimensions (ROCm gfx942).

Eight layers of d3-edge's widths (hidden 1024, 256-wide attention heads, 128-wide gated-delta heads), held in
BF16 as Decision 3.0 holds them, for every state-once layout. Requests of several states (37 to 2,000 tokens)
and 1 to 64 question blocks run branched, through the variable-length kernels, the fused layers and FLA's
``cu_seqlens``, and per row under the same mask. The fused layers must reproduce the eager branched forward
bit for bit, and the branched answers must equal the per-row ones within BF16 rounding.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from vllm_srun.accel import triton_fp64_conv  # noqa: E402
from vllm_srun.accel.rocm import ROCmAccelerator  # noqa: E402
from vllm_srun.engines.native import fast, models  # noqa: E402
from vllm_srun.engines.native.models import branch as branched_module  # noqa: E402
from vllm_srun.engines.native.models.branch import (  # noqa: E402
    CAUSAL,
    LAYOUTS,
    PREFIX_BIDIR,
    Branches,
    StaticBranches,
    pack_layout,
    static_layout,
)
from vllm_srun.engines.native.weights import cast_parameters  # noqa: E402
from vllm_srun.families.decision3.family import CONV_VARIANT  # noqa: E402

pytestmark = pytest.mark.gpu

# (state tokens, question blocks of that state)
REQUESTS = [
    (400, [48, 61, 75, 90]),
    (37, [120]),
    (2000, [80] * 64),
]
OPTIONS = 8
BF16_TOLERANCE = 0.05


def _device():
    accelerator = ROCmAccelerator()
    if not accelerator.available():
        pytest.skip("needs a ROCm GPU")
    device = accelerator.devices()[0]
    if device.arch != "gfx942":
        pytest.skip("the fused kernels are validated on gfx942")
    return accelerator, device


def edge_config(layout: str) -> dict:
    kinds = ["linear_attention"] * 3 + ["full_attention"]
    return {
        "model_type": "qwen3_5_text",
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "num_hidden_layers": 8,
        "layer_types": kinds * 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 4096,
        "hidden_act": "silu",
        "attention_bias": False,
        "rope_parameters": {
            "rope_theta": 10000000,
            "rope_type": "default",
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
        "attention_mode": layout,
    }


def build(layout: str, accelerator, device, *, fused: bool, variants: bool):
    torch.manual_seed(7)
    backbone = models.build("qwen3_5_text", edge_config(layout))
    with torch.no_grad():
        for name, parameter in backbone.named_parameters():
            if name.endswith("A_log"):
                parameter.copy_(torch.log(torch.rand(parameter.shape) * 15 + 1))
            elif name.endswith("dt_bias") or name.endswith("linear_attn.norm.weight"):
                parameter.copy_(1 + 0.02 * torch.randn(parameter.shape))
            elif "norm" in name:
                parameter.copy_(0.02 * torch.randn(parameter.shape))
            else:
                parameter.copy_(0.02 * torch.randn(parameter.shape))
    cast_parameters(backbone, torch.bfloat16)
    backbone = backbone.to(accelerator.torch_device(device)).eval()
    kernels = accelerator.kernels(device)
    if variants:
        kernels.use_variants(CONV_VARIANT)
    backbone.kernels = kernels
    if fused:
        assert fast.fused_unavailable(backbone, kernels) is None
        fast.install_fused(backbone)
    return backbone


def tokens():
    generator = torch.Generator().manual_seed(11)
    states, blocks, owners = [], [], []
    for owner, (state, lengths) in enumerate(REQUESTS):
        states.append(torch.randint(10, 4000, (state,), generator=generator).tolist())
        for length in lengths:
            blocks.append(
                torch.randint(10, 4000, (length,), generator=generator).tolist()
            )
            owners.append(owner)
    return states, blocks, owners


def run_branched(backbone, states, blocks, owners) -> torch.Tensor:
    device = backbone.embed_tokens.weight.device
    ids = torch.tensor([t for s in (*states, *blocks) for t in s], device=device)
    positions = [torch.arange(len(s)) for s in states]
    positions += [
        len(states[o]) + torch.arange(len(b))
        for o, b in zip(owners, blocks, strict=True)
    ]
    layout = Branches(
        [len(s) for s in states],
        [len(b) for b in blocks],
        owners,
        backbone.layout,
        device,
    )
    with torch.inference_mode():
        embeds = backbone.embed(ids[None])
        place = torch.cat(positions)[None].expand(3, -1).to(device)
        hidden = backbone.forward_branches(embeds, place, layout)
        return hidden[0, layout.last].float()


def run_static(
    backbone,
    states,
    blocks,
    owners,
    *,
    joined=False,
    exact=False,
    gaps=False,
    merged=False,
):
    """The static layout's forward (``engine._branch_static`` without its graph): the blocks' last rows."""
    device = backbone.embed_tokens.weight.device
    window = max(
        layer.linear_attn.conv_kernel_size
        for layer in backbone.layers
        if layer.kind == "linear_attention"
    )
    shape, values = static_layout(
        states,
        blocks,
        owners,
        window,
        joined=joined,
        exact=exact,
        gaps=gaps,
        merged=merged,
    )
    ints32, ints64 = (t.to(device) for t in pack_layout(shape, values))

    def forward(flat32, flat64):
        view = StaticBranches(shape, backbone.layout, flat32, flat64, values)
        embeds = backbone.embed(view.ids[None])
        return backbone.forward_branches(
            embeds, view.positions.expand(3, -1), view, view.last
        )[0]

    return shape, (ints32, ints64), forward


def static_hidden(backbone, states, blocks, owners, **options) -> torch.Tensor:
    _, inputs, forward = run_static(backbone, states, blocks, owners, **options)
    with torch.inference_mode():
        return forward(*inputs)[: len(blocks)].float()


def single_questions():
    """Requests of one question each (a joined layout's batches)."""
    generator = torch.Generator().manual_seed(13)
    states = [
        torch.randint(10, 4000, (n,), generator=generator).tolist()
        for n in (37, 400, 1100)
    ]
    blocks = [
        torch.randint(10, 4000, (n,), generator=generator).tolist()
        for n in (48, 90, 61)
    ]
    return states, blocks, [0, 1, 2]


def run_rows(backbone, states, blocks, owners) -> torch.Tensor:
    device = backbone.embed_tokens.weight.device
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(blocks), 8):
            chunk = list(range(start, min(start + 8, len(blocks))))
            rows = [states[owners[i]] + blocks[i] for i in chunk]
            width = max(len(r) for r in rows)
            ids = torch.zeros((len(rows), width), dtype=torch.long)
            mask = torch.zeros((len(rows), width), dtype=torch.long)
            for r, row in enumerate(rows):
                ids[r, width - len(row) :] = torch.tensor(row)
                mask[r, width - len(row) :] = 1
            ids, mask = ids.to(device), mask.to(device)
            positions = (mask.cumsum(1) - 1).clamp(min=0)[None].expand(3, -1, -1)
            hidden = backbone.forward_rows(
                backbone.embed(ids),
                mask,
                [len(states[owners[i]]) for i in chunk],
                positions,
                any(len(r) != width for r in rows),
            )
            outputs.append(hidden[:, -1].float())
    return torch.cat(outputs)


def probabilities(hidden: torch.Tensor) -> torch.Tensor:
    readout = torch.randn(
        OPTIONS, hidden.shape[-1], generator=torch.Generator().manual_seed(3)
    ).to(hidden.device)
    return (hidden @ readout.T * 0.05).softmax(-1)


@pytest.mark.parametrize("variants", [False, True], ids=["default-conv", "d3-conv"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_fused_branches_reproduce_the_eager_branches(layout, variants):
    accelerator, device = _device()
    states, blocks, owners = tokens()
    eager = build(layout, accelerator, device, fused=False, variants=variants)
    fused = build(layout, accelerator, device, fused=True, variants=variants)
    assert torch.equal(
        run_branched(eager, states, blocks, owners),
        run_branched(fused, states, blocks, owners),
    )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_branches_match_rows_on_the_gpu(layout):
    accelerator, device = _device()
    states, blocks, owners = tokens()
    backbone = build(layout, accelerator, device, fused=True, variants=True)
    branched = probabilities(run_branched(backbone, states, blocks, owners))
    rows = probabilities(run_rows(backbone, states, blocks, owners))
    assert (branched - rows).abs().max() < BF16_TOLERANCE
    # A changed decision is a near-tie of the per-row answer.
    top = rows.topk(2, dim=-1).values
    changed = branched.argmax(-1) != rows.argmax(-1)
    assert bool(((top[:, 0] - top[:, 1])[changed] < 2 * BF16_TOLERANCE).all())


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_variable_length_kernels_match_the_reference_attention(layout, monkeypatch):
    accelerator, device = _device()
    if branched_module.VARLEN is None:
        pytest.skip("this PyTorch has no variable-length attention with log-sum-exps")
    states, blocks, owners = tokens()
    backbone = build(layout, accelerator, device, fused=True, variants=True)
    kernel = probabilities(run_branched(backbone, states, blocks, owners))
    monkeypatch.setattr(branched_module, "VARLEN", None)
    reference = probabilities(run_branched(backbone, states, blocks, owners))
    assert (kernel - reference).abs().max() < BF16_TOLERANCE


@pytest.mark.parametrize(
    ("joined", "exact", "merged"),
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, False),
        (False, False, True),
        (False, True, True),
    ],
    ids=[
        "branched",
        "branched-exact",
        "joined",
        "joined-exact",
        "merged",
        "merged-exact",
    ],
)
@pytest.mark.parametrize("layout", [PREFIX_BIDIR, CAUSAL])
def test_static_layouts_answer_as_the_branches(layout, joined, exact, merged):
    accelerator, device = _device()
    states, blocks, owners = single_questions() if joined else tokens()
    backbone = build(layout, accelerator, device, fused=True, variants=True)
    static = probabilities(
        static_hidden(
            backbone, states, blocks, owners, joined=joined, exact=exact, merged=merged
        )
    )
    packed = probabilities(run_branched(backbone, states, blocks, owners))
    assert (static - packed).abs().max() < BF16_TOLERANCE


def test_the_segmented_prep_and_the_shared_stages_compute_bit_for_bit(monkeypatch):
    accelerator, device = _device()
    states, blocks, owners = tokens()
    backbone = build(PREFIX_BIDIR, accelerator, device, fused=True, variants=True)
    segmented = static_hidden(backbone, states, blocks, owners)
    rule = backbone.kernels.select("chunk_gated_delta_rule_static").fn
    assert rule.regions is not None
    monkeypatch.setattr(rule, "regions", None)
    per_region = static_hidden(backbone, states, blocks, owners)
    monkeypatch.delattr(triton_fp64_conv.gdn_prep_fp64, "segments")
    gapped = static_hidden(backbone, states, blocks, owners, gaps=True)
    assert torch.equal(segmented, per_region) and torch.equal(per_region, gapped)


@pytest.mark.parametrize(
    "first_capture_tokens", [0, 4096], ids=["second-sighting", "first-sighting"]
)
def test_a_captured_layout_replays_every_batch_of_its_shape(first_capture_tokens):
    accelerator, device = _device()
    backbone = build(PREFIX_BIDIR, accelerator, device, fused=True, variants=True)
    graphs = fast.Graphs(
        backbone,
        None,
        max_tokens=4096,
        first_capture_tokens=first_capture_tokens,
    )
    states, blocks, owners = tokens()
    states, blocks, owners = states[:2], blocks[:5], owners[:5]
    other = (
        [s[::-1] for s in states],
        [b[::-1] for b in blocks[:-1]] + [blocks[-1][:-3]],
        owners,
    )
    shape, inputs, forward = run_static(backbone, states, blocks, owners)
    other_shape, other_inputs, other_forward = run_static(backbone, *other)
    assert other_shape == shape
    with torch.inference_mode():
        eager = [forward(*inputs).clone(), other_forward(*other_inputs).clone()]
        replayed = [
            graphs.run(("branch", shape), inputs, forward, shape.rows).clone()
            for _ in range(3)
        ]
        replayed.append(
            graphs.run(("branch", shape), other_inputs, forward, shape.rows).clone()
        )
    assert graphs.receipt()["captures"] == 1
    assert graphs.receipt()["eager"] == (0 if first_capture_tokens else 1)
    assert all(torch.equal(out, eager[0]) for out in replayed[:3])
    assert torch.equal(replayed[3], eager[1])


def test_the_readout_rows_are_the_rows_of_the_full_forward():
    accelerator, device = _device()
    states, blocks, owners = tokens()
    backbone = build(PREFIX_BIDIR, accelerator, device, fused=True, variants=True)
    _, inputs, _ = run_static(backbone, states, blocks, owners)
    shape, values = static_layout(states, blocks, owners, 4)
    with torch.inference_mode():
        view = StaticBranches(shape, backbone.layout, *inputs, values)
        embeds = backbone.embed(view.ids[None])
        positions = view.positions.expand(3, -1)
        full = backbone.forward_branches(embeds, positions, view)[0, view.last]
        read = backbone.forward_branches(embeds, positions, view, view.last)[0]
    assert (
        probabilities(full.float()) - probabilities(read.float())
    ).abs().max() < BF16_TOLERANCE
