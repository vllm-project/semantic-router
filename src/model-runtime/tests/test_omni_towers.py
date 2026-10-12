"""The native Vela Omni towers (BERT, SigLIP vision, Whisper encoder, CLAP audio) against Transformers."""

from __future__ import annotations

import pytest
import torch
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.accel.kernels import reference_kernels
from vllm_srun.engines.native import models
from vllm_srun.engines.native.engine import NativeEngine
from vllm_srun.engines.native.models.clap import relative_position_index
from vllm_srun.plugins.base import (
    BackboneSpec,
    DeviceInfo,
    DtypePolicy,
    EncoderBatch,
    EngineOptions,
    ModelSpec,
)
from vllm_srun.testing import omni
from vllm_srun.testing.fixtures import save

CPU = DeviceInfo(accelerator="cpu", index=None, name="cpu")
WHISPER = {**omni.WHISPER_CONFIG, "max_source_positions": 40}
VISION = {**omni.vision_config(64), "patch_size": 16}
BERT = {**omni.text_config("nano", 60), "hidden_size": 32, "num_hidden_layers": 2}


def native(model_type, config, seed=0):
    torch.manual_seed(seed)
    module = models.build(model_type, config)
    generator = torch.Generator().manual_seed(seed)
    module.load_state_dict(omni._random_state(module, generator))
    module.kernels = reference_kernels("cpu")
    return module.eval()


def transformers_module(name):
    return getattr(pytest.importorskip("transformers"), name)


@pytest.mark.reference
def test_bert_rows_match_transformers_padded_and_packed():
    backbone = native("bert", BERT)
    config = transformers_module("BertConfig")(**BERT, attn_implementation="sdpa")
    reference = transformers_module("BertModel")(config, add_pooling_layer=False).eval()
    reference.load_state_dict(backbone.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(1)
    lengths = [7, 19, 3]
    ids = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, length in enumerate(lengths):
        ids[row, :length] = torch.randint(4, 60, (length,), generator=generator)
        mask[row, :length] = 1
    with torch.inference_mode():
        expected = reference(input_ids=ids, attention_mask=mask).last_hidden_state
        padded = backbone(ids, mask)
        packed = backbone.encode(
            torch.cat([ids[row, :n] for row, n in enumerate(lengths)]),
            backbone.packed(lengths, "cpu"),
        )[backbone.num_layers]
        alone = reference(input_ids=ids[1:2, :19]).last_hidden_state[0]
    for row, length in enumerate(lengths):
        torch.testing.assert_close(padded[row, :length], expected[row, :length])
    starts = [0, 7, 26]
    for row, length in enumerate(lengths):
        torch.testing.assert_close(
            packed[starts[row] : starts[row] + length], expected[row, :length]
        )
    assert torch.equal(
        backbone.encode(ids[1, :19], backbone.packed([19], "cpu"))[2], alone
    )


def test_bert_exits_and_layouts():
    backbone = native("bert", BERT)
    ids = torch.randint(4, 60, (11,))
    with torch.inference_mode():
        exits = backbone.encode(ids, backbone.packed([11], "cpu"), (0, 1, 2))
        padded = backbone.encode(ids[None], backbone.padded(None, 1, 11, "cpu"))[2]
    assert set(exits) == {0, 1, 2} and torch.equal(exits[2], padded[0])
    with pytest.raises(ValueError, match="layer exits"):
        backbone.encode(ids, backbone.packed([11], "cpu"), (3,))
    with pytest.raises(ValueError, match="absolute"):
        models.build("bert", {**BERT, "position_embedding_type": "relative_key"})


@pytest.mark.reference
def test_siglip_pooled_vector_matches_transformers():
    tower = native("siglip_vision_model", VISION)
    config = transformers_module("SiglipVisionConfig")(**VISION)
    config._attn_implementation = "sdpa"
    model = transformers_module("SiglipVisionModel")(config)
    # Transformers 5 folds the vision transformer into the model; 4.x nests it.
    reference = getattr(model, "vision_model", model).eval()
    reference.load_state_dict(tower.state_dict(), strict=True)
    pixels = (
        torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(2)) * 2 - 1
    )
    with torch.inference_mode():
        expected = reference(pixel_values=pixels).pooler_output
        pooled = tower(pixels)["pooled"]
    torch.testing.assert_close(pooled, expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="patches"):
        tower(torch.zeros(1, 3, 32, 32))


@pytest.mark.reference
def test_whisper_hidden_states_match_transformers():
    tower = native("whisper_encoder", WHISPER)
    encoder = pytest.importorskip("transformers.models.whisper.modeling_whisper")
    config = transformers_module("WhisperConfig")(**WHISPER)
    config._attn_implementation = "sdpa"
    reference = encoder.WhisperEncoder(config).eval()
    reference.load_state_dict(tower.state_dict(), strict=True)
    features = torch.randn(1, 80, 80, generator=torch.Generator().manual_seed(3))
    with torch.inference_mode():
        expected = reference(input_features=features).last_hidden_state
        hidden = tower(features)["hidden"]
    torch.testing.assert_close(hidden, expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="feature frames"):
        tower(torch.zeros(1, 80, 60))


@pytest.mark.reference
def test_clap_pooled_vector_matches_transformers():
    tower = native("clap_audio_model", omni.CLAP_CONFIG)
    config = transformers_module("ClapAudioConfig")(**omni.CLAP_CONFIG)
    reference = transformers_module("ClapAudioModel")(config).audio_encoder.eval()
    state = tower.state_dict()
    for name, buffer in reference.named_buffers():
        if name.endswith(("relative_position_index", "num_batches_tracked")):
            state[name] = buffer
    reference.load_state_dict(state, strict=True)
    windows = torch.randn(3, 1, 1001, 64, generator=torch.Generator().manual_seed(4))
    with torch.inference_mode():
        expected = reference(input_features=windows * 20 - 40).pooler_output
        pooled = tower(windows * 20 - 40)["pooled"]
    torch.testing.assert_close(pooled, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(
        relative_position_index(8),
        reference.layers[0].blocks[0].attention.self.relative_position_index,
    )


def tower_spec(tmp_path, extra=None):
    """A model whose backbone is a tiny BERT and whose towers live in the same checkpoint."""
    tensors = {}
    parts = {
        "text.": ("bert", BERT),
        "image.": ("siglip_vision_model", VISION),
        "speech.": ("whisper_encoder", WHISPER),
        "clap.": ("clap_audio_model", omni.CLAP_CONFIG),
    }
    for prefix, (model_type, config) in parts.items():
        module = native(model_type, config)
        tensors.update(
            {prefix + name: value for name, value in module.state_dict().items()}
        )
    tensors["clap.batch_norm.num_batches_tracked"] = torch.tensor(0)
    tensors.update(extra or {})
    path = tmp_path / "model.safetensors"
    save(tensors, path)
    files = (path,)
    return ModelSpec(
        name="towers",
        backbone=BackboneSpec("bert", BERT, files, "text."),
        dtype=DtypePolicy(autocast=None, bf16_resident=False),
        max_input_tokens=64,
        encoder=True,
        towers={
            name: BackboneSpec(model_type, config, files, prefix)
            for prefix, (model_type, config) in parts.items()
            if (name := prefix[:-1]) != "text"
        },
    )


def test_the_native_engine_loads_and_runs_towers(tmp_path):
    spec = tower_spec(tmp_path)
    engine = NativeEngine()
    assert engine.supports(spec, CPU) is None
    loaded = engine.load(spec, CPUAccelerator(), CPU, EngineOptions(threads=2))
    assert set(loaded.towers) == {"image", "speech", "clap"}
    assert loaded.parameter_count() == sum(
        sum(p.numel() for p in module.parameters())
        for module in (loaded.backbone, *loaded.towers.values())
    )
    batch = EncoderBatch(
        torch.zeros(0, dtype=torch.long),
        None,
        tower="clap",
        graph_inputs={"input_features": torch.zeros(1, 1, 1001, 64)},
    )
    pooled = loaded.encode(batch).outputs["pooled"]
    assert pooled.shape == (1, 16)
    reference = native("clap_audio_model", omni.CLAP_CONFIG)
    assert torch.equal(
        loaded.towers["clap"].batch_norm.running_var, reference.batch_norm.running_var
    )
    with pytest.raises(ValueError, match="no tower"):
        loaded.encode(
            EncoderBatch(torch.zeros(0, dtype=torch.long), None, tower="video")
        )


def test_a_tower_refuses_unknown_tensors_under_its_prefix(tmp_path):
    spec = tower_spec(tmp_path, {"speech.unexpected.weight": torch.zeros(2)})
    with pytest.raises(ValueError, match=r"unexpected\.weight"):
        NativeEngine().load(spec, CPUAccelerator(), CPU, EngineOptions(threads=1))


def test_an_unknown_tower_architecture_is_refused(tmp_path):
    spec = tower_spec(tmp_path)
    towers = {**spec.towers, "video": BackboneSpec("video_tower", {}, ())}
    from dataclasses import replace

    reason = NativeEngine().supports(replace(spec, towers=towers), CPU)
    assert reason == "no native 'video_tower' backbone"
