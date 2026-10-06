"""Vela Omni as published, on the native engine: verification, readouts and /v1/embeddings."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from dataclasses import replace

import numpy as np
import pytest
import torch
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.engines.native.engine import NativeEngine
from vllm_srun.errors import PackageError
from vllm_srun.families.multimodal_embedding import audio
from vllm_srun.families.multimodal_embedding import package as snapshots
from vllm_srun.families.multimodal_embedding.family import (
    MultimodalEmbeddingFamily,
    golden_audio,
    golden_image,
    identity,
)
from vllm_srun.families.multimodal_embedding.model import NativeOmniModel
from vllm_srun.plugins.base import (
    DeviceInfo,
    EngineOptions,
    PackageRef,
    SurfaceRequest,
)
from vllm_srun.registry.artifacts import safetensors_header
from vllm_srun.registry.tables import omni as pins
from vllm_srun.testing import omni
from vllm_srun.testing.fixtures import save

pytest.importorskip("PIL")

CPU = DeviceInfo(accelerator="cpu", index=None, name="cpu")
PICTURE = {
    "type": "image_url",
    "image_url": {
        "url": f"data:image/png;base64,{base64.b64encode(golden_image()).decode()}"
    },
}
SOUND = {
    "type": "input_audio",
    "input_audio": {"data": base64.b64encode(golden_audio()).decode(), "format": "wav"},
}


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    return omni.write_snapshot(tmp_path_factory.mktemp("omni") / "nano")


def load(root, threads=2):
    family = MultimodalEmbeddingFamily()
    package = family.verify(PackageRef(root))
    spec = family.describe(package)
    engine_model = NativeEngine().load(
        spec, CPUAccelerator(), CPU, EngineOptions(threads=threads)
    )
    return family.load(package, spec, engine_model)


@pytest.fixture(scope="module")
def model(snapshot):
    return load(snapshot)


def serve(model, body):
    request = SurfaceRequest("embeddings", body, None, "exact", True, time.monotonic())
    plan = model.plan_surface("embeddings", request)
    return plan, model.finish_surface(plan, model.run(plan.items))


def rewrite_json(root, name, change):
    path = root / name
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_the_pins_name_every_file_the_family_reads():
    for model in pins.MODELS:
        assert set(model.files) == set(snapshots.FILES)
        assert model.model_sha256 == identity(dict(model.files))
        assert model.engines == {"cpu": "native"}
        assert not any(name.endswith(".py") for name in model.files)


def test_a_local_snapshot_verifies_by_its_file_digests(snapshot):
    family = MultimodalEmbeddingFamily()
    assert family.detect(PackageRef(snapshot))
    package = family.verify(PackageRef(snapshot))
    assert package.details["verification"] == "local"
    assert package.model_sha256 == identity(package.details["files"])
    assert set(package.details["files"]) == set(snapshots.FILES)
    assert package.max_input_tokens == 512 and package.licence is None
    spec = family.describe(package)
    assert spec.backbone.model_type == "bert"
    assert {name: tower.model_type for name, tower in spec.towers.items()} == {
        "image": "siglip_vision_model",
        "speech": "whisper_encoder",
        "clap": "clap_audio_model",
    }
    assert spec.graphs == {} and spec.dtype.autocast is None


def test_a_pinned_revision_must_match_its_digests(snapshot, monkeypatch):
    pinned = replace(pins.MODELS[0], files=dict.fromkeys(snapshots.FILES, "0" * 64))
    monkeypatch.setattr(pins, "lookup", lambda repo: pinned)
    family = MultimodalEmbeddingFamily()
    ref = PackageRef(snapshot, pinned.repo_id, pinned.revision)
    with pytest.raises(PackageError, match="differ from the pinned digests"):
        family.verify(ref)


@pytest.mark.parametrize(
    "tamper, message",
    [
        (lambda root: (root / snapshots.CONFIG).unlink(), "missing"),
        (
            lambda root: rewrite_json(
                root, snapshots.CONFIG, lambda c: c.update(text_pooling="mean")
            ),
            "differs from the published nano model",
        ),
        (
            lambda root: rewrite_json(
                root, snapshots.CONFIG, lambda c: c.pop("parameter_count")
            ),
            "parameter count",
        ),
        (
            lambda root: rewrite_json(
                root,
                f"{snapshots.TEXT}/config.json",
                lambda c: c.update(hidden_size=64),
            ),
            "readout needs 384",
        ),
        (
            lambda root: rewrite_json(
                root,
                f"{snapshots.SPEECH}/preprocessor_config.json",
                lambda c: c.update(hop_length=320),
            ),
            "hop_length",
        ),
    ],
)
def test_snapshots_that_differ_from_the_published_model_are_refused(
    tmp_path, tamper, message
):
    root = omni.write_snapshot(tmp_path / "nano")
    tamper(root)
    with pytest.raises(PackageError, match=message):
        MultimodalEmbeddingFamily().verify(PackageRef(root))


def test_every_checkpoint_tensor_belongs_to_a_part_of_the_model(tmp_path):
    root = omni.write_snapshot(tmp_path / "nano")
    package = snapshots.read(root)
    parts = snapshots.weight_names(package)
    assert sum(map(len, parts.values())) == len(safetensors_header(package.weights))
    from safetensors.torch import load_file

    tensors = load_file(str(package.weights))
    save({**tensors, "fusion.layers.0.weight": torch.zeros(2)}, package.weights)
    with pytest.raises(PackageError, match="no part of the model reads"):
        snapshots.weight_names(package)
    del tensors["audio_residual.scale"]
    save(tensors, package.weights)
    with pytest.raises(PackageError, match="misses parts"):
        snapshots.weight_names(package)


def test_text_image_and_audio_share_one_unit_space(model):
    plan, body = serve(model, {"input": ["route this request", PICTURE, SOUND]})
    vectors = [np.asarray(entry["embedding"]) for entry in body["data"]]
    assert [len(v) for v in vectors] == [384, 384, 384]
    assert all(abs(np.linalg.norm(v) - 1) < 1e-6 for v in vectors)
    assert body["data"][0]["input"]["tokens"] == len(plan.items[0].ids)
    assert model.info.dtype == "fp32" and model.info.embedding.pooling == "cls"
    assert model.device_thread and model.batch_invariant
    keys = [item.cache_key for item in plan.items]
    assert len(set(keys)) == 3


def test_a_batch_answers_every_input_as_it_answers_alone(model):
    inputs = [
        "route this request",
        PICTURE,
        SOUND,
        "a second, somewhat longer request to embed than the first",
        "route this request",
    ]
    plan, _ = serve(model, {"input": inputs})
    assert model.run(plan.items) == [model.run([item])[0] for item in plan.items]


def test_texts_pack_only_where_packing_keeps_each_answer(snapshot, monkeypatch):
    packed = load(snapshot, threads=1)
    assert packed.packed_text
    calls = []
    texts = NativeOmniModel._texts

    def counting(self, sequences):
        calls.append(len(sequences))
        return texts(self, sequences)

    monkeypatch.setattr(NativeOmniModel, "_texts", counting)
    plan, _ = serve(packed, {"input": ["one text", "another text", "a third"]})
    assert calls == [3]
    packed.packed_text = False
    calls.clear()
    packed.run(plan.items)
    assert calls == [1, 1, 1]


def test_the_loaded_parameters_must_match_the_declared_count(tmp_path):
    root = omni.write_snapshot(tmp_path / "nano")
    rewrite_json(
        root,
        snapshots.CONFIG,
        lambda c: c.update(parameter_count=c["parameter_count"] + 1),
    )
    with pytest.raises(PackageError, match="parameters; the model declares"):
        load(root)


def test_invalid_residual_statistics_are_refused(tmp_path):
    from safetensors.torch import load_file

    root = omni.write_snapshot(tmp_path / "nano")
    tensors = load_file(str(root / snapshots.WEIGHTS))
    tensors["audio_residual.scale"][0] = 0.0
    save(tensors, root / snapshots.WEIGHTS)
    with pytest.raises(ValueError, match="residual statistics"):
        load(root)


def test_a_non_finite_readout_is_an_invalid_model_output(model, monkeypatch):
    monkeypatch.setattr(type(model.readout), "image", lambda self, pooled: None)
    _, body = serve(model, {"input": [PICTURE, "still fine"]})
    assert body["data"][0]["error"] == "invalid_model_output"
    assert "embedding" in body["data"][1]


def test_mini_reads_its_published_prefix_and_formats_queries(tmp_path):
    model = load(omni.write_snapshot(tmp_path / "mini", variant="mini"))
    assert model.info.embedding.dimensions == (768,)
    assert model.info.embedding.input_types == ("query", "document")
    assert model.info.limits["max_input_tokens"] == 32768
    _, body = serve(
        model, {"input": ["  hello  ", PICTURE, SOUND], "input_type": "query"}
    )
    assert [len(entry["embedding"]) for entry in body["data"]] == [768, 768, 768]
    plain = model.text.encode("  hello  ", 64)[0]
    query = model.text.encode("  hello  ", 64, "query")[0]
    assert len(query) > len(plain)


def test_the_native_model_runs_on_the_cpu_device_thread(snapshot, monkeypatch):
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.runtime import Runtime

    threads: list[str] = []
    run = NativeOmniModel.run

    def recording(self, items):
        threads.append(threading.current_thread().name)
        return run(self, items)

    monkeypatch.setattr(NativeOmniModel, "run", recording)
    served = ModelConfig(model=str(snapshot), name="omni", device="cpu")
    runtime = Runtime(ServeConfig(models=(served,), threads=2))
    runtime.start(background=False)
    try:
        card = runtime.lookup("omni").card([])
        threads.clear()
        status, _ = asyncio.run(
            runtime.call("embeddings", {"model": "omni", "input": ["hello", PICTURE]})
        )[:2]
    finally:
        runtime.stop()
    assert card["engine"] == "native" and status == 200
    assert threads and all(name.startswith("vllm-sr-cpu") for name in threads)


def test_the_golden_request_runs_every_tower(snapshot, model):
    family = MultimodalEmbeddingFamily()
    (golden,) = family.golden(family.verify(PackageRef(snapshot)))
    assert golden["expected"] == {}
    plan, body = serve(model, golden["body"])
    assert [item.modality for item in plan.items] == ["text", "image", "audio"]
    values = model.golden_values("embeddings", body)
    assert len(values) == 3 * 384 and all(np.isfinite(list(values.values())))


@pytest.mark.reference
def test_mel_filters_are_the_published_extractors():
    audio_utils = pytest.importorskip("transformers.audio_utils")
    whisper = audio.Spectrum.whisper(dict(audio.WHISPER_FEATURES))
    clap = audio.Spectrum.clap_window(dict(audio.CLAP_FEATURES))
    expected = {
        "whisper": audio_utils.mel_filter_bank(
            201, 80, 0.0, 8000.0, 16000, "slaney", "slaney"
        ),
        "clap": audio_utils.mel_filter_bank(
            513, 64, 50, 14000, 48000, "slaney", "slaney"
        ),
    }
    assert np.array_equal(whisper.mel_filters, expected["whisper"])
    assert np.array_equal(clap.mel_filters, expected["clap"])
    assert (whisper.n_frames, clap.n_frames, clap.n_samples) == (3000, 1001, 480000)
