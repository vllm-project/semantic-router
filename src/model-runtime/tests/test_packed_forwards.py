"""Packed merged forwards: rows back to back without padding answer as each request's own forwards do."""

import pytest
import torch
import torch.nn.functional as F
from vllm_srun.accel.gpu import _wrap_varlen
from vllm_srun.accel.kernels import reference_kernels, varlen_attention_ref
from vllm_srun.engines.native.models.packed import Packed, packed_gated_delta
from vllm_srun.engines.native.models.qwen3_5 import GatedDeltaNet
from vllm_srun.engines.native.models.qwen3_5_vision import Qwen3_5VisionBackbone

from .test_decision3_family import QUESTIONS

CPU = torch.device("cpu")


def test_the_gapped_layout_puts_zero_rows_before_every_row():
    packed = Packed([2, 3], CPU)
    x = torch.arange(1, 6, dtype=torch.float32).view(1, 5, 1)
    gapped = packed.gapped(x, 4)
    assert gapped.view(-1).tolist() == [0, 0, 0, 1, 2, 0, 0, 0, 3, 4, 5]
    assert torch.equal(packed.ungapped(gapped, 4), x)
    assert packed.cu_seqlens.tolist() == [0, 2, 5]
    assert packed.positions.tolist() == [0, 1, 0, 1, 2]
    assert packed.last.tolist() == [1, 4]
    with pytest.raises(ValueError):
        Packed([3, 0], CPU)


def test_variable_length_attention_attends_within_each_row():
    torch.manual_seed(0)
    lengths = [3, 7, 1]
    packed = Packed(lengths, CPU)
    total = sum(lengths)
    q, k, v = (
        torch.randn(total, 4, 8),
        torch.randn(total, 2, 8),
        torch.randn(total, 2, 8),
    )
    out = varlen_attention_ref(q, k, v, packed, scale=8**-0.5, groups=2)
    start = 0
    for n in lengths:
        rows = slice(start, start + n)
        qq, kk, vv = (x[rows].transpose(0, 1)[None] for x in (q, k, v))
        want = F.scaled_dot_product_attention(
            qq, kk.repeat_interleave(2, 1), vv.repeat_interleave(2, 1), scale=8**-0.5
        )[0].transpose(0, 1)
        torch.testing.assert_close(out[rows], want)
        start += n


def test_pytorch_varlen_attention_takes_16_bit_inputs_and_fp32_takes_the_reference():
    from vllm_srun.accel.gpu import _wrap_varlen

    packed = Packed([3, 7, 1], CPU)
    seen = []

    def flash(query, key, value, *args, scale, enable_gqa):
        seen.append(query.dtype)
        return query

    attend = _wrap_varlen(flash)
    q, k, v = torch.randn(11, 4, 8), torch.randn(11, 2, 8), torch.randn(11, 2, 8)
    out = attend(q, k, v, packed, scale=0.125, groups=2)
    torch.testing.assert_close(
        out, varlen_attention_ref(q, k, v, packed, scale=0.125, groups=2)
    )
    assert seen == []
    half = [x.to(torch.bfloat16) for x in (q, k, v)]
    assert attend(*half, packed, scale=0.125, groups=2) is half[0]
    assert seen == [torch.bfloat16]


@pytest.mark.parametrize("groups", [1, 2])
def test_legacy_varlen_api_keeps_default_scale_and_grouped_heads(groups):
    """The torch 2.10 interface infers GQA and accepts no scale/GQA keywords."""
    packed = Packed([3, 7, 1], CPU)
    q = torch.randn(11, 4, 8, dtype=torch.bfloat16)
    k, v = (torch.randn(11, 4 // groups, 8, dtype=torch.bfloat16) for _ in range(2))
    calls = []

    def legacy(query, key, value, cu_q, cu_k, max_q, max_k, is_causal=False):
        calls.append((query, key, value, cu_q, cu_k, max_q, max_k, is_causal))
        return query

    out = _wrap_varlen(legacy)(q, k, v, packed, scale=8**-0.5, groups=groups)
    assert out is q
    assert len(calls) == 1
    query, key, value, cu_q, cu_k, max_q, max_k, causal = calls[0]
    assert query is q and key is k and value is v
    assert cu_q is packed.cu_seqlens_int32 and cu_k is cu_q
    assert (max_q, max_k, causal) == (7, 7, False)


@pytest.mark.parametrize("groups", [1, 2])
def test_installed_varlen_python_api_accepts_runtime_arguments(monkeypatch, groups):
    """Check the real torch entrypoint on CPU, replacing only GPU dispatch."""
    varlen = pytest.importorskip("torch.nn.attention.varlen").varlen_attn
    monkeypatch.setattr(
        torch.ops.torch_attn, "_varlen_attn", lambda *args: (args[0], None, None)
    )
    packed = Packed([2, 5], CPU)
    q = torch.randn(7, 4, 8, dtype=torch.bfloat16)
    k, v = (torch.randn(7, 4 // groups, 8, dtype=torch.bfloat16) for _ in range(2))
    out = _wrap_varlen(varlen)(q, k, v, packed, scale=8**-0.5, groups=groups)
    assert out is q


def test_legacy_varlen_uses_reference_for_a_custom_scale():
    """A fixed-scale kernel must not silently change the caller's attention."""
    torch.manual_seed(3)
    packed = Packed([3, 7, 1], CPU)
    q, k, v = (torch.randn(11, n, 8, dtype=torch.bfloat16) for n in (4, 2, 2))

    def legacy(query, key, value, cu_q, cu_k, max_q, max_k):
        pytest.fail("the legacy kernel cannot represent this scale")

    out = _wrap_varlen(legacy)(q, k, v, packed, scale=0.125, groups=2)
    want = varlen_attention_ref(q, k, v, packed, scale=0.125, groups=2)
    torch.testing.assert_close(out, want)


@pytest.mark.parametrize("groups", [1, 2])
def test_modern_varlen_api_receives_scale_and_gqa(groups):
    """Newer torch interfaces retain their explicit scale and GQA controls."""
    packed = Packed([2, 5], CPU)
    q = torch.randn(7, 4, 8, dtype=torch.bfloat16)
    k, v = (torch.randn(7, 4 // groups, 8, dtype=torch.bfloat16) for _ in range(2))

    def modern(query, key, value, cu_q, cu_k, max_q, max_k, *, scale, enable_gqa):
        assert scale == 0.125
        assert enable_gqa is (groups > 1)
        return varlen_attention_ref(
            query, key, value, packed, scale=scale, groups=groups
        )

    out = _wrap_varlen(modern)(q, k, v, packed, scale=0.125, groups=groups)
    want = varlen_attention_ref(q, k, v, packed, scale=0.125, groups=groups)
    torch.testing.assert_close(out, want)


def test_a_packed_gated_delta_layer_answers_each_row_as_alone():
    torch.manual_seed(1)
    config = {
        "hidden_size": 32,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
    }
    layer = GatedDeltaNet(config).eval()
    with torch.no_grad():
        layer.A_log.normal_(0, 0.5)
        layer.norm.weight.normal_(1, 0.1)
    kernels = reference_kernels("cpu")
    lengths = [5, 1, 70, 12]
    rows = [torch.randn(1, n, 32) for n in lengths]
    with torch.inference_mode():
        packed = packed_gated_delta(
            layer, torch.cat(rows, dim=1), Packed(lengths, CPU), kernels
        )
        alone = torch.cat([layer(row, None, kernels) for row in rows], dim=1)
    torch.testing.assert_close(packed, alone, rtol=1e-4, atol=1e-5)


def test_the_vision_tower_attends_over_all_images_in_one_packed_call(d3_runtime):
    engine = d3_runtime.lookup(None).model.engine_model
    tower = engine.towers["vision"]
    assert isinstance(tower, Qwen3_5VisionBackbone)
    config = tower.config
    grids = [(1, 4, 6), (1, 8, 4), (1, 2, 2)]
    width = (
        config["in_channels"]
        * config["temporal_patch_size"]
        * config["patch_size"] ** 2
    )
    torch.manual_seed(2)
    pixels = torch.randn(sum(t * h * w for t, h, w in grids), width)
    with torch.inference_mode():
        loop = tower(pixels, grids)
        packed = tower(pixels, grids, packed=True)
    torch.testing.assert_close(packed, loop, rtol=1e-4, atol=1e-5)


def test_the_cpu_keeps_merged_forwards_padded(d3_runtime):
    model = d3_runtime.lookup(None).model
    assert not model.engine_model.supports_packed
    assert not model.packs_merged_rows
    assert model.engine_model.receipt()["packed"] == {
        "skipped": "no variable-length attention kernel on this device"
    }


@pytest.fixture()
def packed_model(d3_runtime, monkeypatch):
    """The d3 fixture with packed merged forwards forced on, over the reference kernels."""
    model = d3_runtime.lookup(None).model
    monkeypatch.setattr(model.engine_model, "packed_reason", None)
    monkeypatch.setattr(model, "packs_merged_rows", True)
    monkeypatch.setattr(
        model, "feature_cache", type(model.feature_cache)(model.feature_cache.max_bytes)
    )
    return model


def _max_gap(got, want):
    return max(
        abs(a - b)
        for got_row, want_row in zip(got, want, strict=True)
        for a, b in zip(got_row, want_row, strict=True)
    )


def test_packed_text_rows_answer_as_each_question_alone(packed_model):
    model = packed_model
    plans = [
        model.plan("Merge two sorted lists.", QUESTIONS),
        model.plan("Prove that there are infinitely many primes. " * 6, QUESTIONS),
        model.plan({"ticket": "refund", "lines": ["a", "b"]}, QUESTIONS),
    ]
    items = [item for plan in plans for item in plan.items]
    merged = model.run_approximate(items)
    # A packed row sees no padding and no neighbour: it answers as its question run alone.
    alone = [model.run([item])[0] for item in items]
    assert _max_gap(merged, alone) < 1e-5
    # Against a request's own left-padded forwards it moves by their padding's rounding only.
    exact = [value for plan in plans for value in model.run(list(plan.items))]
    assert _max_gap(merged, exact) < 0.05


def test_packed_image_rows_answer_as_each_request_alone(
    packed_model, png_url, monkeypatch
):
    model = packed_model
    plans = [
        model.plan_images("Shown?", QUESTIONS, [png_url(80, 60, 41)]),
        model.plan_images(
            "Shown?", QUESTIONS, [png_url(64, 96, 42), png_url(64, 64, 43)]
        ),
    ]
    exact = [value for plan in plans for value in model.run(list(plan.items))]
    calls = []
    tower = model.engine_model.tower_features

    def counted(name, pixel_values, grids, packed=False):
        calls.append((len(grids), packed))
        return tower(name, pixel_values, grids, packed=packed)

    monkeypatch.setattr(model.engine_model, "tower_features", counted)
    batches = []
    forward = model.engine_model.forward

    def recorded(batch):
        batches.append(batch)
        return forward(batch)

    monkeypatch.setattr(model.engine_model, "forward", recorded)
    items = [item for plan in plans for item in plan.items]
    merged = model.run_approximate(items)
    assert calls == [(3, True)]
    assert [(batch.packed, batch.input_ids.shape[0]) for batch in batches] == [
        (True, 1)
    ]
    assert batches[0].input_ids.shape[1] == sum(batches[0].lengths)
    monkeypatch.setattr(model.engine_model, "forward", forward)
    alone = [model.run([item])[0] for item in items]
    assert _max_gap(merged, alone) < 1e-3
    assert _max_gap(merged, exact) < 0.05


def test_rows_of_like_lengths_stay_padded(packed_model, monkeypatch):
    model = packed_model
    batches = []
    forward = model.engine_model.forward

    def recorded(batch):
        batches.append(batch.packed)
        return forward(batch)

    monkeypatch.setattr(model.engine_model, "forward", recorded)
    like = [
        item
        for state in ("Merge two sorted lists.", "Merge two sorted lists!")
        for item in model.plan(state, {"q": QUESTIONS["reasoning"]}).items
    ]
    varied = like + list(
        model.plan("Prove that there are infinitely many primes. " * 6, QUESTIONS).items
    )
    model.run_approximate(like)
    model.run_approximate(varied)
    # Padding rows of like lengths costs nothing worth packing them for.
    assert batches == [False, True]


def test_a_packed_batch_must_cover_its_rows(packed_model):
    from vllm_srun.plugins.base import ForwardBatch

    ids = torch.ones((1, 10), dtype=torch.long)
    with pytest.raises(ValueError, match="packed lengths do not cover"):
        packed_model.engine_model.forward(
            ForwardBatch(
                ids, ids, torch.tensor([[3]]), torch.tensor([3]), [4, 5], packed=True
            )
        )
