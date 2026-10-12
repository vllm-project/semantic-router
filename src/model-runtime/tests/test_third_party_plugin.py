"""The out-of-tree example plugin, installed through entry points, in a process serving two models.

The example's wheel is built by its own build backend and installed into a
fresh directory, so discovery reads the entry points its ``pyproject.toml``
declares. The tests exercise every surface beyond decisions, bundles across
models, the per-model result cache, bundle fusion and per-model readiness.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from starlette.testclient import TestClient
from vllm_srun.api.app import create_app
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.errors import PlacementError, VerificationError
from vllm_srun.placement import auto_order
from vllm_srun.plugins import registry
from vllm_srun.runtime import Runtime

from .conftest import QUESTIONS, STATE
from .test_api_contract import check

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "third_party_plugin"
BUILD_WHEEL = (
    "import setuptools.build_meta as backend, sys; backend.build_wheel(sys.argv[1])"
)
KEYWORDS = {
    "format": "vllm-sr-example/1",
    "labels": ["billing", "shipping", "other"],
    "keywords": {
        "billing": ["refund", "invoice", "charge"],
        "shipping": ["parcel", "delivery"],
    },
}


@pytest.fixture(scope="module")
def example_plugin(tmp_path_factory):
    root = tmp_path_factory.mktemp("example-plugin")
    source = root / "source"
    shutil.copytree(
        EXAMPLE,
        source,
        ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__"),
    )
    subprocess.run(
        [sys.executable, "-c", BUILD_WHEEL, str(root)],
        cwd=source,
        check=True,
        capture_output=True,
    )
    (wheel,) = root.glob("*.whl")
    site = root / "site"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
    patch = pytest.MonkeyPatch()
    patch.syspath_prepend(str(site))
    registry.discover.cache_clear()
    yield
    patch.undo()
    registry.discover.cache_clear()


@pytest.fixture(scope="module")
def keyword_package(tmp_path_factory):
    root = tmp_path_factory.mktemp("keywords") / "keywords"
    root.mkdir()
    (root / "example_model.json").write_text(json.dumps(KEYWORDS))
    return root


@pytest.fixture(scope="module")
def runtime(example_plugin, keyword_package, qwen3_package):
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(
                    model=str(keyword_package),
                    name="keywords",
                    device="cpu",
                    engine="example_counts",
                ),
                ModelConfig(model=str(qwen3_package), name="kai", device="cpu"),
            )
        )
    )
    runtime.start(background=False)
    yield runtime
    runtime.stop()


@pytest.fixture(scope="module")
def client(runtime):
    return TestClient(create_app(runtime))


def keyword_model(runtime):
    return runtime.lookup("keywords").model


def test_models_describe_both_models_and_the_plugins(client):
    body = client.get("/v1/models").json()
    check("ModelList", body)
    cards = {card["id"]: card for card in body["data"]}
    assert set(cards) == {"keywords", "kai"} and all(
        card["ready"] for card in cards.values()
    )
    keywords = cards["keywords"]
    assert (
        keywords["family"] == "example_keywords"
        and keywords["engine"] == "example_counts"
    )
    assert keywords["heads"][0]["labels"] == KEYWORDS["labels"]
    assert keywords["embedding"]["dimensions"] == [3]
    assert keywords["rerank"]["default"] == {"layer": 1, "dimension": 3}
    plugins = {(p["group"], p["name"]): p for p in keywords["plugins"]}
    assert plugins[("vllm_srun.families", "example_keywords")]["capabilities"] == {
        "surfaces": ["classify", "embeddings", "rerank"],
        "formats": ["vllm-sr-example/1"],
    }
    engines = {
        name: entry["capabilities"]
        for (group, name), entry in plugins.items()
        if group == "vllm_srun.engines"
    }
    assert engines["example_counts"]["outputs"] == ["hidden"]
    assert {name: card["auto_priority"] for name, card in engines.items()} == {
        "example_counts": None,
        "native": 0,
        "onnxruntime": None,
    }
    health = client.get("/health").json()
    check("Health", health)
    assert health["status"] == "ready" and set(health["models"]) == {"keywords", "kai"}


def test_a_third_party_accelerator_and_profile_serve_a_model(
    example_plugin, keyword_package
):
    assert "example_host" in registry.names("accelerators")
    assert "example_host" not in auto_order()
    config = ModelConfig(
        model=str(keyword_package),
        name="keywords",
        device="example_host",
        engine="example_counts",
        profile="example_one_by_one",
    )
    runtime = Runtime(ServeConfig(models=(config,)))
    runtime.start(background=False)
    try:
        client = TestClient(create_app(runtime))
        (card,) = client.get("/v1/models").json()["data"]
        assert (
            card["accelerator"] == "example_host" and card["device"] == "example_host"
        )
        assert card["profile"] == "example_one_by_one"
        plugins = {(p["group"], p["name"]): p for p in card["plugins"]}
        assert plugins[("vllm_srun.accelerators", "example_host")]["capabilities"] == {
            "validated": False,
            "auto_priority": None,
        }
        assert (
            plugins[("vllm_srun.profiles", "example_one_by_one")]["capabilities"][
                "numerics"
            ]
            == "exact"
        )
        body = client.post(
            "/v1/classify",
            json={
                "input": ["Please refund the invoice", "Where is my parcel?"],
                "options": {"profile": "example_one_by_one", "return_meta": True},
            },
        ).json()
        assert [result["label"] for result in body["results"]] == [
            "billing",
            "shipping",
        ]
        assert body["meta"]["profile"] == "example_one_by_one"
        assert body["meta"]["accelerator"] == "example_host"
    finally:
        runtime.stop()


@pytest.mark.parametrize("device_thread", [True, False])
def test_batches_off_the_cpu_run_through_the_accelerators_execute(
    example_plugin, keyword_package, monkeypatch, device_thread
):
    from vllm_sr_example.accelerator import HostAccelerator
    from vllm_sr_example.family import KeywordModel

    executed = []
    execute = HostAccelerator.execute

    def recording(self, device, work):
        executed.append(device.label)
        return execute(self, device, work)

    monkeypatch.setattr(HostAccelerator, "execute", recording)
    monkeypatch.setattr(KeywordModel, "device_thread", device_thread)
    config = ModelConfig(
        model=str(keyword_package),
        name="keywords",
        device="example_host",
        engine="example_counts",
    )
    runtime = Runtime(ServeConfig(models=(config,), result_cache_entries=0))
    runtime.start(background=False)
    try:
        loaded = len(executed)
        client = TestClient(create_app(runtime))
        for text in ("Please refund the invoice", "Where is my parcel?"):
            answer = client.post("/v1/classify", json={"input": [text]})
            assert answer.status_code == 200
    finally:
        runtime.stop()
    assert executed[loaded:] == ["example_host", "example_host"]


def test_classify_embeddings_and_rerank_answer_by_contract(client):
    classified = client.post(
        "/v1/classify",
        json={
            "model": "keywords",
            "input": ["Please refund the invoice", "Where is my parcel?"],
            "options": {"return_meta": True},
        },
    )
    assert classified.status_code == 200
    body = classified.json()
    check("ClassifyResponse", body)
    assert [result["label"] for result in body["results"]] == ["billing", "shipping"]
    assert body["meta"]["engine"] == "example_counts"

    embedded = client.post(
        "/v1/embeddings",
        json={"model": "keywords", "input": ["refund", "parcel delivery"]},
    ).json()
    check("EmbeddingsResponse", embedded)
    vectors = [item["embedding"] for item in embedded["data"]]
    assert all(abs(sum(v * v for v in vector) - 1) < 1e-9 for vector in vectors)
    assert embedded["meta"] == {"representation": embedded["meta"]["representation"]}
    assert embedded["meta"]["representation"]["dimension"] == 3

    ranked = client.post(
        "/v1/rerank",
        json={
            "model": "keywords",
            "query": "refund my charge",
            "documents": ["parcel", "invoice refund"],
        },
    ).json()
    check("RerankResponse", ranked)
    assert [result["index"] for result in ranked["results"]] == [1, 0]


def test_requests_name_their_model_and_surface(client):
    missing = client.post("/v1/classify", json={"input": ["refund"]})
    assert (
        missing.status_code == 400
        and "model is required" in missing.json()["error"]["message"]
    )
    unknown = client.post("/v1/classify", json={"model": "nobody", "input": ["refund"]})
    assert unknown.status_code == 404
    unsupported = client.post(
        "/v1/rerank", json={"model": "kai", "query": "q", "documents": ["d"]}
    )
    assert unsupported.status_code == 422
    check("ErrorResponse", unsupported.json())
    assert unsupported.json()["error"]["code"] == "unsupported_surface"


def test_a_bundle_serves_every_surface_and_model_in_task_order(client):
    alone = client.post(
        "/v1/decisions", json={"model": "kai", "state": STATE, "questions": QUESTIONS}
    ).json()
    bundle = client.post(
        "/v1/bundle",
        json={
            "tasks": [
                {
                    "id": "route",
                    "decisions": {
                        "model": "kai",
                        "state": STATE,
                        "questions": QUESTIONS,
                    },
                },
                {
                    "id": "topic",
                    "classify": {"model": "keywords", "input": ["refund please"]},
                },
                {
                    "id": "vector",
                    "embeddings": {"model": "keywords", "input": "parcel"},
                },
                {
                    "id": "rank",
                    "rerank": {
                        "model": "keywords",
                        "query": "parcel",
                        "documents": ["parcel"],
                    },
                },
                {"id": "lost", "classify": {"model": "nobody", "input": ["x"]}},
            ]
        },
    )
    assert bundle.status_code == 200
    body = bundle.json()
    check("BundleResponse", body)
    results = body["results"]
    assert [result["id"] for result in results] == [
        "route",
        "topic",
        "vector",
        "rank",
        "lost",
    ]
    assert [result["status"] for result in results] == [200, 200, 200, 200, 404]
    assert results[0]["decisions"]["answers"] == alone["answers"]
    assert results[1]["classify"]["results"][0]["label"] == "billing"
    assert results[4]["error"]["code"] == "model_not_found"


@pytest.mark.parametrize(
    "fillers, bundle_options, task_deadlines",
    [
        (11, {}, (None, None)),
        (12, {"deadline_ms": 5000}, (None, None)),
        (13, {}, (4000, 5000)),
    ],
    ids=["no deadline", "bundle deadline", "task deadlines"],
)
def test_bundled_tasks_for_one_model_share_one_forward(
    client, runtime, fillers, bundle_options, task_deadlines
):
    model = keyword_model(runtime)
    before = model.forwards
    tasks = []
    for task_id, text, deadline in zip(
        ("a", "b"), ("one charge", "two parcels, delivery"), task_deadlines, strict=True
    ):
        # The example caches by token IDs, so each case differs in length.
        classify = {"model": "keywords", "input": [text + " fused" * fillers]}
        if deadline is not None:
            classify["options"] = {"deadline_ms": deadline}
        tasks.append({"id": task_id, "classify": classify})
    body = client.post(
        "/v1/bundle", json={"tasks": tasks, "options": bundle_options}
    ).json()
    assert [result["status"] for result in body["results"]] == [200, 200]
    assert model.forwards == before + 1


def test_a_forward_error_fails_only_its_request(client, runtime, monkeypatch):
    model = keyword_model(runtime)
    run = model.run
    failed = []

    def fail_once(items, **options):
        if not failed:
            failed.append(items)
            raise ValueError("a readout bug")
        return run(items, **options)

    monkeypatch.setattr(model, "run", fail_once)
    broken = client.post(
        "/v1/classify", json={"model": "keywords", "input": ["a refund that fails"]}
    )
    assert broken.status_code == 500
    assert broken.json()["error"]["code"] == "internal_error"
    served = client.post(
        "/v1/classify", json={"model": "keywords", "input": ["a refund that works"]}
    )
    assert served.status_code == 200
    assert client.get("/health").status_code == 200


def test_a_non_finite_answer_fails_only_its_task(client, runtime, monkeypatch):
    model = keyword_model(runtime)
    run = model.run
    poisoned = 9

    def poison(items, **options):
        rows = run(items, **options)
        return [
            [math.nan] * len(row) if len(item.ids) == poisoned else row
            for item, row in zip(items, rows, strict=True)
        ]

    monkeypatch.setattr(model, "run", poison)
    text = "a refund" + " nan" * (poisoned - 2)
    alone = client.post("/v1/classify", json={"model": "keywords", "input": [text]})
    assert alone.status_code == 500
    assert alone.json()["error"]["code"] == "internal_error"
    body = client.post(
        "/v1/bundle",
        json={
            "tasks": [
                {"id": "bad", "classify": {"model": "keywords", "input": [text]}},
                {
                    "id": "good",
                    "classify": {"model": "keywords", "input": ["a parcel delivered"]},
                },
            ]
        },
    )
    assert body.status_code == 200
    results = body.json()["results"]
    assert [result["status"] for result in results] == [500, 200]
    assert results[0]["error"]["code"] == "internal_error"


def test_repeated_items_are_answered_from_the_result_cache(client, runtime):
    model = keyword_model(runtime)
    request = {"model": "keywords", "input": ["an invoice for the cache"]}
    first = client.post("/v1/classify", json=request).json()
    forwards = model.forwards
    second = client.post("/v1/classify", json=request).json()
    assert model.forwards == forwards
    assert second["results"] == first["results"]
    assert (
        'vllm_srun_result_cache_total{model="keywords",outcome="hit"}'
        in client.get("/metrics").text
    )


def test_a_bundle_deadline_reaches_every_task(client):
    body = client.post(
        "/v1/bundle",
        json={
            "tasks": [
                {
                    "id": "late",
                    "classify": {
                        "model": "keywords",
                        "input": ["unique words, never cached"],
                    },
                },
                {
                    "id": "route",
                    "decisions": {
                        "model": "kai",
                        "state": STATE,
                        "questions": QUESTIONS,
                    },
                },
            ],
            "options": {"deadline_ms": 0.001},
        },
    ).json()
    check("BundleResponse", body)
    late, route = body["results"]
    assert late["classify"]["results"][0]["error"] == "deadline_exceeded"
    assert {answer["error"] for answer in route["decisions"]["answers"].values()} == {
        "deadline_exceeded"
    }


@pytest.mark.parametrize(
    "body,status",
    [
        ({"tasks": []}, 400),
        ({"tasks": [{"id": "a"}]}, 400),
        ({"tasks": [{"id": "a", "classify": {}, "rerank": {}}]}, 400),
        (
            {
                "tasks": [
                    {"id": "a", "classify": {"input": "x"}},
                    {"id": "a", "classify": {"input": "y"}},
                ]
            },
            400,
        ),
        (
            {"tasks": [{"id": str(i), "classify": {"input": "x"}} for i in range(65)]},
            413,
        ),
        (
            {
                "tasks": [{"id": "a", "classify": {"input": "x"}}],
                "options": {"deadline_ms": -1},
            },
            400,
        ),
    ],
)
def test_malformed_bundles_are_rejected(client, body, status):
    response = client.post("/v1/bundle", json=body)
    assert response.status_code == status
    check("ErrorResponse", response.json())


def test_a_model_that_fails_to_load_leaves_the_others_serving(
    example_plugin, keyword_package, tmp_path
):
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(
                    model=str(keyword_package),
                    name="keywords",
                    device="cpu",
                    engine="example_counts",
                ),
                ModelConfig(
                    model=str(tmp_path / "missing"), name="broken", device="cpu"
                ),
            )
        )
    )
    runtime.start(background=True)
    runtime.wait(timeout=60)
    try:
        client = TestClient(create_app(runtime))
        health = client.get("/health")
        assert health.status_code == 503 and health.json()["status"] == "degraded"
        assert health.json()["models"]["broken"]["status"] == "failed"
        served = client.post(
            "/v1/classify", json={"model": "keywords", "input": ["refund"]}
        )
        assert served.status_code == 200
        broken = client.post(
            "/v1/classify", json={"model": "broken", "input": ["refund"]}
        )
        assert broken.status_code == 503
        cards = {card["id"]: card for card in client.get("/v1/models").json()["data"]}
        assert cards["broken"]["status"] == "failed" and not cards["broken"]["ready"]
    finally:
        runtime.stop()


@pytest.mark.parametrize(
    ("failure", "loads", "state"),
    [
        (PlacementError("rocm:0: 1.0 GiB free"), 3, "ready"),
        (VerificationError("golden mismatch"), 1, "failed"),
    ],
)
def test_a_failed_load_is_retried_unless_the_answers_are_wrong(
    example_plugin, keyword_package, monkeypatch, failure, loads, state
):
    import vllm_srun.runtime as runtime_module

    load = runtime_module.ServedModel.load
    seen = []

    def flaky(self):
        if self.label == "flaky":
            seen.append((self.health.state, self.health.reason))
            if len(seen) < 3:
                raise failure
        return load(self)

    monkeypatch.setattr(runtime_module.ServedModel, "load", flaky)
    models = tuple(
        ModelConfig(
            model=str(keyword_package), name=name, device="cpu", engine="example_counts"
        )
        for name in ("keywords", "flaky")
    )
    runtime = Runtime(ServeConfig(models=models, load_retry_seconds=0.01))
    runtime.start(background=True)
    try:
        runtime.wait(timeout=60)
        assert len(seen) == loads
        assert runtime.lookup("keywords").health.ready
        flaky_health = runtime.lookup("flaky").health
        assert flaky_health.state == state
        if state == "ready":
            assert seen[1] == (
                "loading",
                "retrying after PlacementError: rocm:0: 1.0 GiB free (attempt 1 of 5)",
            )
        else:
            assert flaky_health.reason == "VerificationError: golden mismatch"
    finally:
        runtime.stop()


@pytest.mark.parametrize("background", [False, True])
def test_loading_freezes_the_heap(example_plugin, keyword_package, background):
    import gc

    models = (
        ModelConfig(
            model=str(keyword_package),
            name="keywords",
            device="cpu",
            engine="example_counts",
        ),
    )
    gc.unfreeze()
    runtime = Runtime(ServeConfig(models=models))
    try:
        runtime.start(background=background)
        assert runtime.wait(timeout=60)
        assert gc.get_freeze_count() > 0
    finally:
        runtime.stop()
        gc.unfreeze()
