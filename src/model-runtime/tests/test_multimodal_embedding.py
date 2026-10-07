"""The multimodal_embedding family's prepared bundles on onnxruntime: verification, processors, /v1/embeddings."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
from dataclasses import replace

import numpy as np
import pytest
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.errors import PackageError
from vllm_srun.families.multimodal_embedding import bundle as bundles
from vllm_srun.families.multimodal_embedding.family import (
    MultimodalEmbeddingFamily,
    golden_audio,
    golden_image,
)
from vllm_srun.families.multimodal_embedding.processors import (
    ImageProcessor,
    TextProcessor,
)
from vllm_srun.plugins.base import (
    DEADLINE,
    DeviceInfo,
    EngineOptions,
    PackageRef,
    SurfaceRequest,
)
from vllm_srun.registry.artifacts import inventory
from vllm_srun.testing import omni

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")
pytest.importorskip("PIL")

CPU = DeviceInfo(accelerator="cpu", index=None, name="cpu")


@pytest.fixture(scope="module")
def nano(tmp_path_factory):
    return omni.write_bundle(tmp_path_factory.mktemp("omni") / "vela-1.0-omni-nano")


def test_a_missing_extra_names_its_install(nano, monkeypatch):
    import importlib.util

    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *rest: None if name == "PIL" else find_spec(name, *rest),
    )
    family = MultimodalEmbeddingFamily()
    package = family.verify(PackageRef(nano))
    with pytest.raises(
        RuntimeError,
        match=r"Omni bundles need Pillow: pip install '\./src/model-runtime\[multimodal,onnx\]'",
    ):
        family.describe(package)


def test_nano_sizes_its_text_graph_pool(nano, tmp_path):
    family = MultimodalEmbeddingFamily()
    spec = family.describe(family.verify(PackageRef(nano)))
    assert spec.graph_threads == {"text": 12}
    assert spec.graph_spin_us == {"text": 10_000}
    mini = omni.write_bundle(tmp_path / "vela-1.0-omni-mini", variant="mini")
    mini_spec = family.describe(family.verify(PackageRef(mini)))
    assert mini_spec.graph_threads == {} and mini_spec.graph_spin_us == {}


def test_the_default_engine_serves_a_bundle_on_onnxruntime(tmp_path):
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.runtime import Runtime

    source = {"repo_id": "example/omni-fixture", "revision": "0" * 40}
    bundle = omni.write_bundle(tmp_path / "omni", source=source)
    served = ModelConfig(model=str(bundle), name="omni", device="cpu")
    runtime = Runtime(ServeConfig(models=(served,)))
    runtime.start(background=False)
    try:
        assert runtime.lookup("omni").card([])["engine"] == "onnxruntime"
    finally:
        runtime.stop()
    native = Runtime(ServeConfig(models=(replace(served, engine="native"),)))
    with pytest.raises(RuntimeError, match="no engine can run"):
        native.load()


def test_batches_never_run_on_the_cpu_device_thread(tmp_path, monkeypatch):
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.families.multimodal_embedding.model import GraphOmniModel
    from vllm_srun.runtime import Runtime

    threads: list[str] = []
    run = GraphOmniModel.run

    def recording_run(self, items):
        threads.append(threading.current_thread().name)
        return run(self, items)

    monkeypatch.setattr(GraphOmniModel, "run", recording_run)
    source = {"repo_id": "example/omni-fixture", "revision": "0" * 40}
    bundle = omni.write_bundle(tmp_path / "omni", source=source)
    served = ModelConfig(model=str(bundle), name="omni", device="cpu")
    runtime = Runtime(ServeConfig(models=(served,)))
    runtime.start(background=False)
    try:
        threads.clear()
        body = {"model": "omni", "input": "hello", "options": {"overflow": "reject"}}
        inline = asyncio.run(runtime.call("embeddings", body, 64))
        # The worker answers a job before it releases the model, and a caller
        # runs a group only on a released model.
        scheduler = runtime.lookup("omni").scheduler
        with scheduler._lock:
            assert scheduler._lock.wait_for(lambda: not scheduler._owned, timeout=30)
        body["input"] = "hello there"
        planned_off_loop = asyncio.run(runtime.call("embeddings", body))
    finally:
        runtime.stop()
    assert inline[0] == planned_off_loop[0] == 200
    assert threads[0] == "vllm-srun-worker"
    assert threads[1] not in ("vllm-srun-worker", "vllm-sr-cpu")
    assert threads[1] != threading.main_thread().name


@pytest.fixture(scope="module")
def model(nano):
    from vllm_srun.engines.onnxruntime.engine import OnnxRuntimeEngine

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


def test_a_bundle_names_the_revision_it_was_prepared_from(nano, tmp_path):
    family = MultimodalEmbeddingFamily()
    ref = PackageRef(tmp_path, omni.SOURCE["repo_id"], omni.SOURCE["revision"])
    assert family.detect(ref) and not family.detect(
        PackageRef(tmp_path, "someone/else")
    )
    assert family.fetch(PackageRef(nano)).root == nano
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
    assert descriptor["formats"] == ["vela-omni/1", "vela-omni-bundle/1"]
    assert descriptor["surfaces"] == ["embeddings"]
    assert descriptor["engines"] == ["native", "onnxruntime"]


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


def test_a_batch_answers_every_input_as_it_answers_alone(model):
    image = base64.b64encode(golden_image()).decode()
    sound = base64.b64encode(golden_audio()).decode()
    inputs = [
        "route this request",
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
        {"type": "input_audio", "input_audio": {"data": sound, "format": "wav"}},
        "a second, somewhat longer request to embed",
        "route this request",
    ]
    plan = model.plan_surface("embeddings", request({"input": inputs}))
    assert model.batch_invariant
    assert model.run(plan.items) == [model.run([item])[0] for item in plan.items]


def test_a_batch_runs_its_images_one_at_a_time_beside_text_and_audio(
    model, monkeypatch
):
    image = base64.b64encode(golden_image()).decode()
    sound = base64.b64encode(golden_audio()).decode()
    picture = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image}"},
    }
    inputs = [
        picture,
        "route this request",
        picture,
        {"type": "input_audio", "input_audio": {"data": sound, "format": "wav"}},
        "a second request",
        picture,
    ]
    plan = model.plan_surface("embeddings", request({"input": inputs}))
    lock = threading.Lock()
    images_running, most_images = [0], [0]
    threads: dict[str, set[str]] = {}
    embed = type(model)._embed

    def recording(self, item):
        with lock:
            threads.setdefault(item.modality, set()).add(
                threading.current_thread().name
            )
            if item.modality == "image":
                images_running[0] += 1
                most_images[0] = max(most_images[0], images_running[0])
        try:
            time.sleep(0.01)
            return embed(self, item)
        finally:
            with lock:
                if item.modality == "image":
                    images_running[0] -= 1

    monkeypatch.setattr(type(model), "_embed", recording)
    vectors = model.run(plan.items)
    monkeypatch.undo()
    assert vectors == [model.run([item])[0] for item in plan.items]
    assert most_images[0] == 1
    assert threads["image"] == {threading.current_thread().name}
    shared = threads["text"] | threads["audio"]
    assert all(name.startswith("vllm-sr-omni_") for name in shared)


@pytest.mark.parametrize("failing", ["image", "text"])
def test_a_failed_input_fails_its_batch_and_leaves_none_of_it_running(
    model, monkeypatch, failing
):
    image = base64.b64encode(golden_image()).decode()
    picture = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image}"},
    }
    inputs = [picture, *(f"request number {n}" for n in range(7))]
    plan = model.plan_surface("embeddings", request({"input": inputs}))
    target = next(item for item in plan.items if item.modality == failing)
    lock = threading.Lock()
    started, running = [], [0]

    def embed(self, item):
        with lock:
            started.append(item)
            running[0] += 1
        try:
            time.sleep(0.01 if item is target else 0.05)
            if item is target:
                raise RuntimeError(f"the {failing} graph failed")
            return [0.0]
        finally:
            with lock:
                running[0] -= 1

    monkeypatch.setattr(type(model), "_embed", embed)
    with pytest.raises(RuntimeError, match=f"the {failing} graph failed"):
        model.run(plan.items)
    with lock:
        left_running, started_by_then = running[0], len(started)
    time.sleep(0.2)
    assert left_running == 0
    assert len(started) == started_by_then
    if failing == "image":
        # Four texts were running when the image failed; the other three never start.
        assert started_by_then <= 5


def test_a_failed_image_fails_only_its_request_and_the_model_keeps_serving(
    tmp_path, monkeypatch
):
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.families.multimodal_embedding.model import GraphOmniModel
    from vllm_srun.runtime import Runtime

    source = {"repo_id": "example/omni-fixture", "revision": "0" * 40}
    bundle = omni.write_bundle(tmp_path / "omni", source=source)
    served = ModelConfig(model=str(bundle), name="omni", device="cpu")
    runtime = Runtime(ServeConfig(models=(served,), result_cache_entries=0))
    image = base64.b64encode(golden_image()).decode()
    picture = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image}"},
    }
    body = {"model": "omni", "input": ["route this request", picture, "a second one"]}
    embed = GraphOmniModel._embed

    def failing(self, item):
        if item.modality == "image":
            raise RuntimeError("the image graph failed")
        return embed(self, item)

    runtime.start(background=False)
    try:
        before = asyncio.run(runtime.call("embeddings", body))
        monkeypatch.setattr(GraphOmniModel, "_embed", failing)
        failed = asyncio.run(runtime.call("embeddings", body))
        monkeypatch.undo()
        after = asyncio.run(runtime.call("embeddings", body))
        state = runtime.lookup("omni").health.state
    finally:
        runtime.stop()
    assert failed[0] == 500
    assert failed[1]["error"]["code"] == "internal_error"
    assert "the image graph failed" in failed[1]["error"]["message"]
    assert state == "ready"
    assert before[0] == after[0] == 200 and after[1] == before[1]


def test_media_inputs_carry_a_scheduler_cost_and_text_counts_its_tokens(model):
    from vllm_srun.families.multimodal_embedding.family import MEDIA_COST
    from vllm_srun.scheduler.planner import cost

    image = base64.b64encode(golden_image()).decode()
    sound = base64.b64encode(golden_audio()).decode()
    inputs = [
        "route this request",
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
        {"type": "input_audio", "input_audio": {"data": sound, "format": "wav"}},
    ]
    text, picture, audio = model.plan_surface(
        "embeddings", request({"input": inputs})
    ).items
    assert cost(text) == len(text.ids)
    assert (cost(picture), cost(audio)) == (
        MEDIA_COST["nano"]["image"],
        MEDIA_COST["nano"]["audio"],
    )


def test_truncate_cuts_over_long_text_inside_the_special_tokens(model):
    text = "route this request to the model that answers it best " * 4
    image = base64.b64encode(golden_image()).decode()
    inputs = [
        text,
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
    ]
    options = {"max_tokens": 8}
    rejected = serve(model, {"input": inputs, "options": options})[1]
    assert rejected["data"][0]["error"] == "max_length_exceeded"
    plan, cut = serve(
        model, {"input": inputs, "options": {**options, "overflow": "truncate"}}
    )
    full = model.text.encode(text, 512)[0]
    assert plan.items[0].ids == [*full[:7], full[-1]]
    assert cut["data"][0]["input"]["processed_tokens"] == 8
    assert cut["data"][0]["input"]["truncated"] is True
    assert cut["data"][0]["input"]["tokens"] == len(full)
    assert cut["data"][1]["embedding"] == rejected["data"][1]["embedding"]


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
    with pytest.raises(ValueError, match="no layer exits"):
        model.plan_surface("embeddings", request({"input": "hello", "layer": 3}))


def test_images_of_any_mode_and_size_become_normalized_channels_first_pixels(nano):
    from PIL import Image

    processor = ImageProcessor.from_bundle(bundles.load(nano))
    photo = io.BytesIO()
    noise = np.random.default_rng(7).integers(0, 256, (97, 151, 3), dtype=np.uint8)
    Image.fromarray(noise).save(photo, format="JPEG", quality=90)
    cases = [png(37, 21, mode) for mode in ("RGB", "L", "RGBA", "P", "CMYK")]
    for data in [*cases, png(1, 143), png(1031, 777), photo.getvalue()]:
        pixels = processor.pixels(data)
        assert pixels.shape == (1, 3, 512, 512) and pixels.dtype == np.float32
        assert pixels.min() >= -1.0 and pixels.max() <= 1.0
        with Image.open(io.BytesIO(data)) as image:
            resized = image.convert("RGB").resize(
                (512, 512), resample=Image.Resampling.BICUBIC, reducing_gap=None
            )
        scaled = (np.asarray(resized, dtype=np.float64) * (1 / 255)).astype(np.float32)
        reference = ((scaled - processor.mean) / processor.std).transpose(2, 0, 1)
        assert np.array_equal(pixels[0], reference)
    assert processor.pixels(b"\x89PNG broken") == "invalid_input"
    processor.close()


def test_deadlines_and_non_unit_outputs(tmp_path, model):
    plan = model.plan_surface("embeddings", request({"input": ["hello", "route"]}))
    expired = model.finish_surface(plan, DEADLINE)
    assert [entry["error"] for entry in expired["data"]] == ["deadline_exceeded"] * 2
    from vllm_srun.engines.onnxruntime.engine import OnnxRuntimeEngine

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


def test_golden_request_exercises_every_graph(nano, model, tmp_path):
    family = MultimodalEmbeddingFamily()
    (golden,) = family.golden(family.verify(PackageRef(nano)))
    assert golden["surface"] == "embeddings"
    assert set(golden["expected"]) == {"cpu", "rocm"}
    assert all(len(values) == 3 * 384 for values in golden["expected"].values())
    unpinned = {"repo_id": "example/omni-fixture", "revision": "0" * 40}
    other = omni.write_bundle(tmp_path / "other", source=unpinned)
    assert family.golden(family.verify(PackageRef(other)))[0]["expected"] == {}
    plan, body = serve(model, golden["body"])
    assert [item.modality for item in plan.items] == ["text", "image", "audio"]
    values = model.golden_values("embeddings", body)
    assert len(values) == 3 * 384 and all(np.isfinite(list(values.values())))


def test_mini_formats_queries_with_its_instruction(tmp_path):
    text = TextProcessor.from_bundle(
        bundles.load(omni.write_bundle(tmp_path / "mini", variant="mini"))
    )
    assert text.input_types == ("query", "document")
    plain, _ = text.encode("  hello  ", 64)
    document, _ = text.encode("  hello  ", 64, "document")
    query, _ = text.encode("  hello  ", 64, "query")
    assert plain == document == [2, 10, 3]
    assert len(query) > len(plain) and query[-2:] == [10, 3]
    assert (
        TextProcessor.from_bundle(
            bundles.load(omni.write_bundle(tmp_path / "nano"))
        ).input_types
        == ()
    )
