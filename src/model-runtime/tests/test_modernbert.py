"""The native ModernBERT backbone against the Transformers reference."""

import pytest
import torch
from vllm_srun.accel import onednn
from vllm_srun.accel.kernels import Kernel, reference_kernels
from vllm_srun.engines.native import encoder, models
from vllm_srun.engines.native.models import modernbert
from vllm_srun.engines.native.models.modernbert import (
    BAND_FROM,
    FULL,
    SLIDING,
    ModernBertBackbone,
    layer_types,
    length_groups,
    packed_layout,
    padded_layout,
    rope_frequencies,
    rope_parameters,
)
from vllm_srun.engines.native.weights import lay_out_linears, load_backbone
from vllm_srun.testing.fixtures import modernbert_config, random_backbone, save

VOCAB = 300
# Shorter and longer than the 4-token half-window and the 9-token band, padded and not.
LENGTHS = [[3], [17], [9, 3], [12, 12], [21, 5, 1, 16]]
DEFAULT_ROPE = {
    "global_rope_theta": 160000,
    "local_rope_theta": 10000,
    "rope_scaling": None,
}
NESTED_ROPE = {
    "global_rope_theta": None,
    "local_rope_theta": None,
    "rope_scaling": None,
    "rope_parameters": {
        FULL: {"rope_type": "default", "rope_theta": 160000.0},
        SLIDING: {"rope_type": "default", "rope_theta": 10000.0},
    },
}
CONFIGS = {
    "yarn": modernbert_config(VOCAB),
    "default": modernbert_config(VOCAB, **DEFAULT_ROPE),
    "nested": modernbert_config(VOCAB, **NESTED_ROPE, num_hidden_layers=5),
}


def native(config, seed=0):
    state = random_backbone("modernbert", config, seed)
    backbone = models.build("modernbert", config)
    backbone.load_state_dict({name: value.float() for name, value in state.items()})
    backbone.kernels = reference_kernels("cpu")
    return backbone.eval(), state


def reference(config, state):
    transformers = pytest.importorskip("transformers")
    hf_config = transformers.ModernBertConfig(
        **{key: value for key, value in config.items() if value is not None},
        attn_implementation="sdpa",
    )
    model = transformers.ModernBertModel(hf_config).eval()
    model.load_state_dict(
        {name: value.float() for name, value in state.items()}, strict=True
    )
    return model


def batch(lengths, seed=0):
    generator = torch.Generator().manual_seed(seed)
    width = max(lengths)
    ids = torch.zeros(len(lengths), width, dtype=torch.long)
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for row, length in enumerate(lengths):
        ids[row, :length] = torch.randint(3, VOCAB, (length,), generator=generator)
        mask[row, :length] = 1
    return ids, mask


@pytest.mark.reference
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_padded_rows_match_transformers_bit_for_bit(name):
    backbone, state = native(CONFIGS[name])
    model = reference(CONFIGS[name], state)
    for lengths in LENGTHS:
        ids, mask = batch(lengths)
        with torch.inference_mode():
            ours = backbone(ids, mask)
            theirs = model(input_ids=ids, attention_mask=mask).last_hidden_state
        for row, length in enumerate(lengths):
            assert torch.equal(ours[row, :length], theirs[row, :length]), (
                name,
                lengths,
            )


@pytest.mark.reference
@pytest.mark.parametrize("normalize", [False, True])
def test_layer_exits_follow_transformers_hidden_states(normalize):
    config = CONFIGS["yarn"]
    backbone, state = native(config)
    model = reference(config, state)
    ids, mask = batch([7, 11])
    layout = padded_layout(mask, 2, 11, backbone.window, "cpu")
    with torch.inference_mode():
        exits = backbone.encode(ids, layout, (0, 1, 2, 4), normalize_exits=normalize)
        hidden = model(
            input_ids=ids, attention_mask=mask, output_hidden_states=True
        ).hidden_states
        # Transformers' last captured state is already the final-normed output.
        expected = {
            layer: (
                model.final_norm(hidden[layer])
                if normalize and layer < 4
                else hidden[layer]
            )
            for layer in (0, 1, 2, 4)
        }
    assert sorted(exits) == [0, 1, 2, 4]
    for layer, value in exits.items():
        assert torch.equal(value[0, :7], expected[layer][0, :7])
        assert torch.equal(value[1], expected[layer][1])


def test_packed_tokens_match_each_sequence_alone():
    backbone, _ = native(CONFIGS["yarn"], seed=3)
    for lengths in LENGTHS:
        ids, mask = batch(lengths, seed=1)
        flat = ids[mask.bool()]
        with torch.inference_mode():
            packed = backbone.encode(
                flat, packed_layout(lengths, backbone.window, "cpu")
            )[backbone.num_layers]
            alone = [
                backbone(ids[row : row + 1, :length])[0]
                for row, length in enumerate(lengths)
            ]
        assert packed.shape == (sum(lengths), CONFIGS["yarn"]["hidden_size"])
        torch.testing.assert_close(packed, torch.cat(alone), rtol=0, atol=1e-5)


def test_one_unpadded_row_is_the_padded_path_bit_for_bit():
    backbone, _ = native(CONFIGS["yarn"], seed=4)
    ids, _ = batch([19], seed=2)
    with torch.inference_mode():
        packed = backbone.encode(ids[0], packed_layout([19], backbone.window, "cpu"))
        assert torch.equal(packed[backbone.num_layers], backbone(ids)[0])


def test_packed_rows_wider_than_every_sequence_keep_their_values():
    backbone, _ = native(CONFIGS["default"], seed=5)
    lengths = [6, 2]
    ids, mask = batch(lengths, seed=3)
    flat = ids[mask.bool()]
    with torch.inference_mode():
        tight = backbone.encode(flat, packed_layout(lengths, backbone.window, "cpu"))
        wide = backbone.encode(
            flat, packed_layout(lengths, backbone.window, "cpu", width=16)
        )
    torch.testing.assert_close(tight[4], wide[4], rtol=0, atol=1e-5)


@pytest.mark.reference
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_rotary_frequencies_match_transformers(name):
    transformers = pytest.importorskip("transformers")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    config = CONFIGS[name]
    hf_config = transformers.ModernBertConfig(
        **{key: value for key, value in config.items() if value is not None}
    )
    params = rope_parameters(config)
    dim = config["hidden_size"] // config["num_attention_heads"]
    for kind in (FULL, SLIDING):
        expected = hf_config.rope_parameters[kind]
        assert params[kind]["rope_type"] == expected["rope_type"]
        assert params[kind]["rope_theta"] == expected["rope_theta"]
        inv_freq, scaling = rope_frequencies(
            params[kind], dim, config["max_position_embeddings"]
        )
        if expected["rope_type"] == "default":
            reference_freq = 1.0 / (
                expected["rope_theta"]
                ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
            )
            reference_scaling = 1.0
        else:
            reference_freq, reference_scaling = ROPE_INIT_FUNCTIONS[
                expected["rope_type"]
            ](hf_config, None, layer_type=kind)
        assert torch.equal(inv_freq, reference_freq)
        assert scaling == reference_scaling


def test_rope_and_layer_types_from_either_config_generation():
    legacy = rope_parameters(CONFIGS["default"])
    nested = rope_parameters(CONFIGS["nested"])
    assert legacy == {
        kind: {"rope_type": "default", "rope_theta": value["rope_theta"]}
        for kind, value in nested.items()
    }
    yarn = rope_parameters(CONFIGS["yarn"])
    assert yarn[FULL]["rope_type"] == yarn[SLIDING]["rope_type"] == "yarn"
    assert yarn[SLIDING]["original_max_position_embeddings"] == 8192
    assert layer_types(CONFIGS["nested"]) == [FULL, SLIDING, SLIDING, FULL, SLIDING]
    explicit = modernbert_config(VOCAB, layer_types=[FULL] * 4)
    assert layer_types(explicit) == [FULL] * 4


def test_checkpoint_namespace_loads_by_exact_name(tmp_path):
    config = CONFIGS["yarn"]
    state = random_backbone("modernbert", config, seed=6)
    task_model = {f"model.{name}": value for name, value in state.items()}
    task_model["classifier.weight"] = torch.zeros(3, config["hidden_size"])
    save(task_model, tmp_path / "model.safetensors")
    with torch.device("meta"):
        backbone = ModernBertBackbone(config)
    backbone.rotary_emb = type(backbone.rotary_emb)(config)
    load_backbone(backbone, [tmp_path / "model.safetensors"], "model.")
    assert not any(parameter.is_meta for parameter in backbone.parameters())
    assert backbone.layers[0].attn_norm.__class__ is torch.nn.Identity
    assert backbone.layers[1].attn_norm.weight.shape == (config["hidden_size"],)


@pytest.mark.parametrize(
    "override,message",
    [
        ({"rope_scaling": {"rope_type": "dynamic", "factor": 2.0}}, "rope type"),
        ({"hidden_activation": "gelu_new"}, "activation"),
        ({"layer_types": [FULL, "chunked_attention", FULL, FULL]}, "layer types"),
    ],
)
def test_unsupported_configurations_are_refused(override, message):
    with pytest.raises(ValueError, match=message):
        ModernBertBackbone(modernbert_config(VOCAB, **override))


def test_layer_exits_outside_the_encoder_are_refused():
    backbone, _ = native(CONFIGS["yarn"])
    ids, _ = batch([5])
    layout = packed_layout([5], backbone.window, "cpu")
    with pytest.raises(ValueError, match="layer exits"):
        backbone.encode(ids[0], layout, exits=(-1,))
    with pytest.raises(ValueError, match="layer exits"):
        backbone.encode(ids[0], layout, exits=(5,))


def test_native_engine_encodes_packed_and_padded_batches(tmp_path):
    from vllm_srun.accel.cpu import CPUAccelerator
    from vllm_srun.engines.native.engine import NativeEngine
    from vllm_srun.plugins.base import (
        BackboneSpec,
        DtypePolicy,
        EncoderBatch,
        EngineOptions,
        ModelSpec,
    )

    config = CONFIGS["yarn"]
    state = random_backbone("modernbert", config, seed=7)
    save({f"model.{n}": v for n, v in state.items()}, tmp_path / "model.safetensors")
    backbone, _ = native(config, seed=7)
    spec = ModelSpec(
        "tiny",
        BackboneSpec("modernbert", config, (tmp_path / "model.safetensors",), "model."),
        DtypePolicy(autocast=None, bf16_resident=False),
        max_input_tokens=4096,
        encoder=True,
    )
    accelerator = CPUAccelerator()
    model = NativeEngine().load(
        spec, accelerator, accelerator.devices()[0], EngineOptions()
    )
    lengths = [9, 3, 14]
    ids, mask = batch(lengths, seed=4)
    padded = model.encode(EncoderBatch(ids, mask, layers=(2, 4))).hidden
    packed = model.encode(
        EncoderBatch(ids[mask.bool()], None, layers=(2, 4), lengths=lengths)
    ).hidden
    with torch.inference_mode():
        assert torch.equal(padded[4], backbone(ids, mask))
    start = 0
    for row, length in enumerate(lengths):
        for layer in (2, 4):
            torch.testing.assert_close(
                packed[layer][start : start + length],
                padded[layer][row, :length],
                rtol=0,
                atol=1e-5,
            )
        start += length
    with pytest.raises(ValueError, match="packed lengths"):
        model.encode(EncoderBatch(ids[0], None, lengths=[4]))


@pytest.mark.parametrize("lengths", [[40], [37, 9, 22], [64, 64]])
def test_local_layers_in_query_blocks_match_the_dense_band(lengths):
    backbone, _ = native(CONFIGS["yarn"], seed=10)
    ids, mask = batch(lengths, seed=7)
    flat = ids[mask.bool()]
    dense = packed_layout(lengths, backbone.window, "cpu", band_from=1 << 20)
    blocked = packed_layout(lengths, backbone.window, "cpu", band_from=1, block=8)
    assert dense.groups[0].band is None and blocked.groups[0].band is not None
    assert blocked.groups[0].masks[SLIDING] is None
    with torch.inference_mode():
        expected = backbone.encode(flat, dense)[backbone.num_layers]
        actual = backbone.encode(flat, blocked)[backbone.num_layers]
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize("variant", [None, "additive_masks"])
@pytest.mark.parametrize("lengths", [[40], [37, 9, 22], [64, 64], [61]])
def test_local_layers_read_query_blocks_a_few_at_a_time_bit_for_bit(
    monkeypatch, lengths, variant
):
    """CPU calls of ``BAND_CALL_TOKENS`` query tokens give the result of one call over every block."""
    backbone, _ = native(CONFIGS["yarn"], seed=13)
    backbone.kernels.use_variants({"sdpa": variant} if variant else {})
    ids, mask = batch(lengths, seed=10)
    flat = ids[mask.bool()]
    layout = packed_layout(lengths, backbone.window, "cpu", band_from=1, block=8)
    sdpa, calls = backbone.kernels.select("sdpa"), []

    def counted(query, *args, **kwargs):
        calls.append(query.shape[1])
        return sdpa.fn(query, *args, **kwargs)

    backbone.kernels.register(
        Kernel("sdpa", counted, "counted", exact=True, variant=sdpa.variant)
    )
    with torch.inference_mode():
        monkeypatch.setattr(modernbert, "BAND_CALL_TOKENS", 1 << 20)
        whole = backbone.encode(flat, layout, (1, 2, 4))
        monkeypatch.setattr(modernbert, "BAND_CALL_TOKENS", 16)
        calls.clear()
        chunked = backbone.encode(flat, layout, (1, 2, 4))
    heads = CONFIGS["yarn"]["num_attention_heads"]
    assert calls and max(calls) <= heads * 16 // 8
    for layer, value in whole.items():
        assert torch.equal(chunked[layer], value)


def test_length_groups_cut_where_a_grid_pads_too_much():
    assert length_groups([12, 3, 7]) == [[0, 2, 1]]
    assert length_groups([2000, 10, 12, 1900]) == [[0, 3], [2, 1]]
    assert length_groups([1500, 1500, 1500]) == [[0, 1, 2]]


@pytest.mark.parametrize("band_from", [BAND_FROM["cpu"], 1])
def test_rows_of_very_different_lengths_attend_in_separate_grids(
    monkeypatch, band_from
):
    monkeypatch.setattr(encoder, "LAUNCH_BOUND_TOKENS", 0)
    backbone, _ = native(CONFIGS["yarn"], seed=11)
    lengths = [40, 3, 37, 5]
    ids, mask = batch(lengths, seed=8)
    flat = ids[mask.bool()]
    layout = packed_layout(
        lengths, backbone.window, "cpu", band_from=band_from, block=8
    )
    assert [(group.rows, group.width) for group in layout.groups] == [(2, 40), (2, 5)]
    with torch.inference_mode():
        grouped = backbone.encode(flat, layout, (0, 2, 4))
        alone = [
            backbone.encode(
                ids[row, :length],
                packed_layout([length], backbone.window, "cpu"),
                (0, 2, 4),
            )
            for row, length in enumerate(lengths)
        ]
    for layer in (0, 2, 4):
        expected = torch.cat([outputs[layer] for outputs in alone])
        torch.testing.assert_close(grouped[layer], expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize("band_from", [BAND_FROM["cpu"], 1])
def test_packed_rows_are_bit_identical_alone_and_in_any_batch(band_from):
    if not onednn.available():
        pytest.skip("oneDNN's packed linear needs an x86 CPU")
    backbone, _ = native(CONFIGS["yarn"], seed=12)
    lay_out_linears(backbone, onednn.PackedLinear)
    lengths = [7, 30, 7, 19, 30, 3]
    ids, _ = batch(lengths, seed=9)
    rows = [ids[row, :length] for row, length in enumerate(lengths)]

    def run(selected):
        widths = [lengths[row] for row in selected]
        layout = packed_layout(
            widths, backbone.window, "cpu", band_from=band_from, block=8, uniform=True
        )
        out = backbone.encode(
            torch.cat([rows[row] for row in selected]), layout, (0, 2, 4)
        )
        starts = [0, *torch.tensor(widths).cumsum(0).tolist()]
        return [
            {layer: value[starts[i] : starts[i + 1]] for layer, value in out.items()}
            for i in range(len(selected))
        ]

    with torch.inference_mode():
        alone = [run([row])[0] for row in range(len(lengths))]
        for selected in ([0, 1, 2, 3, 4, 5], [5, 3, 1], [4, 0, 2]):
            for position, outputs in enumerate(run(selected)):
                for layer, value in outputs.items():
                    assert torch.equal(value, alone[selected[position]][layer])
