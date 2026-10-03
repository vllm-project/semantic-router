"""The multimodal_embedding family: bundle verification, processors and /v1/embeddings on Omni."""

from __future__ import annotations

import base64
import io
import json
import time

import numpy as np
import pytest
from vllm_sr_runtime.accel.cpu import CPUAccelerator
from vllm_sr_runtime.errors import PackageError
from vllm_sr_runtime.families.multimodal_embedding import bundle as bundles
from vllm_sr_runtime.families.multimodal_embedding.family import (
    PREPARED_DIR_ENV,
    MultimodalEmbeddingFamily,
    golden_audio,
    golden_image,
)
from vllm_sr_runtime.families.multimodal_embedding.processors import (
    ImageProcessor,
    TextProcessor,
)
from vllm_sr_runtime.plugins.base import (
    DEADLINE,
    DeviceInfo,
    EngineOptions,
    PackageRef,
    SurfaceRequest,
)
from vllm_sr_runtime.registry.artifacts import inventory
from vllm_sr_runtime.testing import omni

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")
pytest.importorskip("PIL")

CPU = DeviceInfo(accelerator="cpu", index=None, name="cpu")


@pytest.fixture(scope="module")
def nano(tmp_path_factory):
    return omni.write_bundle(tmp_path_factory.mktemp("omni") / "vela-1.0-omni-nano")


@pytest.fixture(scope="module")
def model(nano):
    from vllm_sr_runtime.engines.onnxruntime.engine import OnnxRuntimeEngine

    family = MultimodalEmbeddingFamily()
    package = family.verify(PackageRef(nano))
    spec = family.describe(package)
    engine = OnnxRuntimeEngine()
    assert engine.supports(spec, CPU) is None
    loaded = engine.load(spec, CPUAccelerator(), CPU, EngineOptions(threads=1))
    return family.load(package, spec, loaded)


def request(body):
    return SurfaceRequest("embeddings", body, None, "exact", True, time.monotonic())


def serve(model, body):
    plan = model.plan_surface("embeddings", request(body))
    return plan, model.finish_surface(plan, model.run(plan.items))


def png(width: int, height: int, mode: str = "RGB") -> bytes:
    from PIL import Image

    rng = np.random.default_rng(width * height)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).convert(mode).save(
        buffer, format="PNG" if mode != "CMYK" else "JPEG"
    )
    return buffer.getvalue()


def rewrite(root, change):
    manifest = json.loads((root / bundles.MANIFEST).read_text())
    change(manifest)
    (root / bundles.MANIFEST).write_text(json.dumps(manifest))


def refresh_inventory(root):
    files = {
        name: digest
        for name, digest in inventory(root).items()
        if name != bundles.MANIFEST
    }
    rewrite(root, lambda manifest: manifest.update(files=files))


def test_bundle_verifies_and_identifies_its_inventory(nano):
    verified = bundles.load(nano)
    assert verified.variant == "nano" and verified.contract.dimension == 384
    assert verified.source == (omni.SOURCE["repo_id"], omni.SOURCE["revision"])
    assert (
        set(verified.graphs) == set(bundles.GRAPHS) and len(verified.model_sha256) == 64
    )


@pytest.mark.parametrize(
    "tamper",
    [
        lambda root: (root / "onnx/text.onnx").write_bytes(b"tampered"),
        lambda root: (root / "extra.txt").write_text("x"),
        lambda root: (root / bundles.PENDING_MANIFEST).write_text("{}"),
        lambda root: rewrite(root, lambda m: m["embedding"].update(dimension=768)),
        lambda root: rewrite(root, lambda m: m["processors"]["image"].update(size=256)),
        lambda root: rewrite(root, lambda m: m["files"].update({"model.py": "0" * 64})),
    ],
)
def test_bundles_that_differ_from_the_contract_are_refused(tmp_path, tamper):
    root = omni.write_bundle(tmp_path / "bundle")
    tamper(root)
    with pytest.raises(PackageError):
        bundles.load(root)


def test_a_failed_parity_receipt_is_refused(tmp_path):
    root = omni.write_bundle(tmp_path / "bundle")
    receipt = json.loads((root / "reference_parity.json").read_text())
    receipt["tests"] = receipt["tests"][1:]
    (root / "reference_parity.json").write_text(json.dumps(receipt))
    refresh_inventory(root)
    with pytest.raises(PackageError, match="parity receipt"):
        bundles.load(root)


def test_hub_ids_resolve_to_the_prepared_bundle(nano, monkeypatch, tmp_path):
    family = MultimodalEmbeddingFamily()
    ref = PackageRef(tmp_path, omni.SOURCE["repo_id"], omni.SOURCE["revision"])
    assert family.detect(ref) and not family.detect(
        PackageRef(tmp_path, "someone/else")
    )
    monkeypatch.setenv(PREPARED_DIR_ENV, str(nano.parent))
    assert family.fetch(ref).root == nano
    monkeypatch.setenv(PREPARED_DIR_ENV, str(tmp_path))
    with pytest.raises(PackageError, match="tools/models/vela_omni"):
        family.fetch(ref)
    with pytest.raises(PackageError, match="prepared from"):
        family.verify(PackageRef(nano, "vllm-sr/Vela-1.0-Omni-Mini", "1" * 40))


def test_model_card_descriptor(model):
    embedding = model.info.embedding
    assert embedding.dimensions == (384,) and embedding.layers == ()
    assert (
        embedding.modalities == ("text", "image", "audio")
        and embedding.pooling == "cls"
    )
    assert model.info.licence == "apache-2.0" and model.info.parameters > 0
    descriptor = MultimodalEmbeddingFamily.descriptor()
    assert descriptor["formats"] == ["vela-omni-bundle/1"] and descriptor[
        "surfaces"
    ] == ["embeddings"]


def test_text_image_and_audio_share_one_unit_space(model):
    image = base64.b64encode(golden_image()).decode()
    sound = base64.b64encode(golden_audio()).decode()
    plan, body = serve(
        model,
        {
            "input": [
                "route this request",
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image}"},
                },
                {
                    "type": "input_audio",
                    "input_audio": {"data": sound, "format": "wav"},
                },
            ]
        },
    )
    vectors = [np.asarray(entry["embedding"]) for entry in body["data"]]
    assert [len(v) for v in vectors] == [384, 384, 384]
    assert all(abs(np.linalg.norm(v) - 1) < 1e-5 for v in vectors)
    assert body["data"][0]["input"] == {
        "tokens": 5,
        "processed_tokens": 5,
        "truncated": False,
    }
    assert "input" not in body["data"][1]
    assert body["usage"] == {"prompt_tokens": 5, "total_tokens": 5}
    assert body["meta"]["representation"]["layer"] == 0
    keys = [item.cache_key for item in plan.items]
    assert len(set(keys)) == 3
    again, _ = serve(model, {"input": ["route this request"]})
    assert again.items[0].cache_key == keys[0]


def test_bad_inputs_fail_in_place(model):
    garbage = base64.b64encode(b"not an image").decode()
    _, body = serve(
        model,
        {
            "input": [
                "hello " * 600,
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{garbage}"},
                },
                {
                    "type": "input_audio",
                    "input_audio": {"data": garbage, "format": "mp3"},
                },
                "hello",
            ]
        },
    )
    errors = [entry.get("error") for entry in body["data"]]
    assert errors == ["max_length_exceeded", "invalid_input", "invalid_input", None]
    with pytest.raises(ValueError, match="rejects over-long"):
        model.plan_surface(
            "embeddings",
            request({"input": "hello", "options": {"overflow": "truncate"}}),
        )
    with pytest.raises(ValueError, match="no layer exits"):
        model.plan_surface("embeddings", request({"input": "hello", "layer": 3}))


def test_images_of_any_mode_become_normalized_channels_first_pixels(nano):
    processor = ImageProcessor(bundles.load(nano))
    for mode in ("RGB", "L", "RGBA", "P", "CMYK"):
        pixels = processor.pixels(png(37, 21, mode))
        assert pixels.shape == (1, 3, 512, 512) and pixels.dtype == np.float32
        assert pixels.min() >= -1.0 and pixels.max() <= 1.0
    assert processor.pixels(b"\x89PNG broken") == "invalid_input"


def test_deadlines_and_non_unit_outputs(tmp_path, model):
    plan = model.plan_surface("embeddings", request({"input": ["hello", "route"]}))
    expired = model.finish_surface(plan, DEADLINE)
    assert [entry["error"] for entry in expired["data"]] == ["deadline_exceeded"] * 2
    from vllm_sr_runtime.engines.onnxruntime.engine import OnnxRuntimeEngine

    root = omni.write_bundle(tmp_path / "raw", normalize=False)
    family = MultimodalEmbeddingFamily()
    package = family.verify(PackageRef(root))
    spec = family.describe(package)
    engine_model = OnnxRuntimeEngine().load(
        spec, CPUAccelerator(), CPU, EngineOptions(threads=1)
    )
    raw = family.load(package, spec, engine_model)
    _, body = serve(raw, {"input": ["hello"]})
    assert body["data"][0]["error"] == "invalid_model_output"


def test_golden_request_exercises_every_graph(nano, model):
    family = MultimodalEmbeddingFamily()
    (golden,) = family.golden(family.verify(PackageRef(nano)))
    assert golden["surface"] == "embeddings" and golden["expected"] == {}
    plan, body = serve(model, golden["body"])
    assert [item.modality for item in plan.items] == ["text", "image", "audio"]
    values = model.golden_values("embeddings", body)
    assert len(values) == 3 * 384 and all(np.isfinite(list(values.values())))


def test_mini_formats_queries_with_its_instruction(tmp_path):
    text = TextProcessor(
        bundles.load(omni.write_bundle(tmp_path / "mini", variant="mini"))
    )
    assert text.input_types == ("query", "document")
    plain, _ = text.encode("  hello  ", 64)
    document, _ = text.encode("  hello  ", 64, "document")
    query, _ = text.encode("  hello  ", 64, "query")
    assert plain == document == [2, 10, 3]
    assert len(query) > len(plain) and query[-2:] == [10, 3]
    assert (
        TextProcessor(bundles.load(omni.write_bundle(tmp_path / "nano"))).input_types
        == ()
    )
