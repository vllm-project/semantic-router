"""The task_heads family served end to end on tiny HF ModernBERT task packages."""

from __future__ import annotations

import hashlib

import pytest
import torch
from starlette.testclient import TestClient
from vllm_srun.accel import onednn
from vllm_srun.api.app import create_app
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.errors import PackageError
from vllm_srun.families.task_heads.family import (
    TaskHeadsFamily,
    batch_invariant,
)
from vllm_srun.heads.task import identical
from vllm_srun.plugins.base import PackageRef, SurfaceRequest
from vllm_srun.registry import builtin
from vllm_srun.registry.tables.common import BuiltinModel
from vllm_srun.runtime import Runtime
from vllm_srun.testing.fixtures import write_fixture

from .test_api_contract import check

TEXT = (
    "My name is Tom Baker and my email is tom.baker@example.com; call +1 415 555 0100."
)
LONG = TEXT * 12
GROUNDED = {
    "context": "The tower was completed in 1889.",
    "question": "When was the tower completed?",
    "answer": "It was completed in 1889 in Rome.",
}


@pytest.fixture(scope="module")
def packages(tmp_path_factory):
    root = tmp_path_factory.mktemp("task-heads")
    return {
        variant: write_fixture(
            root / variant, family="task_heads", variant=variant, seed=index
        )
        for index, variant in enumerate(
            ["sequence", "safety", "scores", "token", "grounded"]
        )
    }


@pytest.fixture(scope="module")
def runtime(packages):
    models = tuple(
        ModelConfig(model=str(path), name=name, device="cpu")
        for name, path in packages.items()
    )
    runtime = Runtime(ServeConfig(models=models, result_cache_entries=0))
    runtime.start(background=False)
    yield runtime
    runtime.stop()


@pytest.fixture(scope="module")
def client(runtime):
    return TestClient(create_app(runtime))


def classify(client, model, inputs, **options):
    body = {"model": model, "input": inputs}
    if options:
        body["options"] = options
    response = client.post("/v1/classify", json=body)
    assert response.status_code == 200, response.text
    check("ClassifyResponse", response.json())
    return response.json()


def served(runtime, name):
    return next(s for s in runtime.served if s.config.name == name)


def test_cards_list_each_head(client):
    models = client.get("/v1/models").json()
    check("ModelList", models)
    cards = {card["id"]: card for card in models["data"]}
    assert cards["sequence"]["heads"][0]["kind"] == "sequence"
    assert cards["sequence"]["heads"][0]["window"] is None
    assert cards["scores"]["heads"][0]["window"] == {"tokens": 64, "overlap": 16}
    assert len(cards["scores"]["heads"][0]["thresholds"]) == 12
    assert cards["grounded"]["heads"][0]["inputs"] == ["grounded"]
    assert all(card["status"] == "ready" for card in cards.values())


def test_cards_pin_the_operating_point_each_head_applies(client, packages):
    cards = {card["id"]: card for card in client.get("/v1/models").json()["data"]}
    for name in ("scores", "grounded"):
        policy = (packages[name] / "operating_point.json").read_bytes()
        assert (
            cards[name]["heads"][0]["operating_point_sha256"]
            == hashlib.sha256(policy).hexdigest()
        )
    assert cards["sequence"]["heads"][0]["operating_point_sha256"] is None


@pytest.mark.reference
@pytest.mark.parametrize("name", ["sequence", "safety", "scores", "token"])
def test_heads_match_the_transformers_task_models(client, packages, name):
    transformers = pytest.importorskip("transformers")
    kind = "Token" if name == "token" else "Sequence"
    reference = (
        getattr(transformers, f"ModernBertFor{kind}Classification")
        .from_pretrained(
            str(packages[name]), dtype=torch.float32, attn_implementation="sdpa"
        )
        .eval()
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(packages[name] / "tokenizer.json")
    )
    body = classify(client, name, [TEXT], return_tokens=True)
    ids = torch.tensor([tokenizer.encode(TEXT)])
    with torch.inference_mode():
        logits = reference(input_ids=ids).logits[0]
    if name == "token":
        expected = torch.softmax(logits, -1)[1:-1]
        ours = torch.tensor(
            [row["probabilities"] for row in body["results"][0]["tokens"]]
        )
    else:
        expected = (
            torch.sigmoid(logits) if name == "scores" else torch.softmax(logits, -1)
        )
        key = "scores" if name == "scores" else "probabilities"
        ours = torch.tensor(body["results"][0][key])
    torch.testing.assert_close(ours, expected, rtol=0, atol=1e-6)


def test_sequence_overflow_policies(client):
    whole = classify(client, "sequence", [LONG])["results"][0]
    tokens = whole["input"]["tokens"]
    rejected = classify(client, "sequence", [LONG], max_tokens=32)["results"][0]
    assert rejected == {"index": 0, "error": "max_length_exceeded"}
    cut = classify(client, "sequence", [LONG], max_tokens=32, overflow="truncate")[
        "results"
    ][0]
    assert cut["input"] == {"tokens": tokens, "processed_tokens": 32, "truncated": True}
    windowed = classify(
        client,
        "sequence",
        [LONG],
        overflow="window",
        window={"tokens": 48, "overlap": 8},
    )["results"][0]
    assert windowed["input"]["windows"] == len(windowed["windows"]) > 1
    starts = [window["start"] for window in windowed["windows"]]
    assert starts == list(range(0, starts[-1] + 1, 38))
    assert windowed["windows"][-1]["end"] == tokens - 2
    for position, value in enumerate(windowed["probabilities"]):
        assert value == max(w["probabilities"][position] for w in windowed["windows"])
    assert (
        windowed["label"]
        == classify(client, "sequence", [LONG])["labels"][
            windowed["probabilities"].index(max(windowed["probabilities"]))
        ]
    )


def test_scores_read_their_operating_point_windows(client, packages):
    result = classify(client, "scores", [LONG])["results"][0]
    assert result["input"]["windows"] == len(result["windows"]) > 1
    assert all(w["end"] - w["start"] <= 62 for w in result["windows"])
    import json

    point = json.loads((packages["scores"] / "operating_point.json").read_text())
    expected = [
        label
        for label, score, threshold in zip(
            point["labels"], result["scores"], point["thresholds"], strict=True
        )
        if score >= threshold
    ]
    assert result["selected"] == expected


def test_token_windows_decode_spans_once(client):
    whole = classify(client, "token", [LONG], return_tokens=True)["results"][0]
    one = classify(
        client,
        "token",
        [LONG],
        overflow="window",
        window={"tokens": 4096, "overlap": 0},
    )
    assert one["results"][0]["spans"] == whole["spans"]
    windowed = classify(
        client, "token", [LONG], overflow="window", window={"tokens": 24, "overlap": 8}
    )["results"][0]
    assert windowed["input"]["windows"] > 3
    for span in windowed["spans"]:
        assert LONG[span["start"] : span["end"]] == span["text"]
        assert (
            span["label"] in {"AGE", "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"}
            or span["label"].isupper()
        )
    cut = classify(client, "token", [LONG], max_tokens=24, overflow="truncate")[
        "results"
    ][0]
    assert cut["input"]["truncated"] and cut["input"]["processed_tokens"] <= 24
    assert all(span["end"] <= len(LONG) for span in cut["spans"])
    strict = classify(client, "token", [LONG], threshold=1.0)["results"][0]
    assert strict["spans"] == []


def test_grounded_spans_stay_on_the_answer(client):
    result = classify(client, "grounded", [GROUNDED], return_tokens=True)["results"][0]
    answer = GROUNDED["answer"]
    for span in result["spans"]:
        assert answer[span["start"] : span["end"]] == span["text"]
        assert span["label"] == "hallucinated" and span["probability"] > 0.5
    assert len(result["tokens"]) > 3
    none = classify(client, "grounded", [GROUNDED], threshold=1.0)["results"][0]
    assert none["spans"] == []
    long = {**GROUNDED, "context": GROUNDED["context"] * 300}
    assert (
        classify(client, "grounded", [long])["results"][0]["error"]
        == "max_length_exceeded"
    )
    kept = classify(client, "grounded", [long], overflow="truncate")["results"][0]
    assert kept["input"]["truncated"] and kept["input"]["processed_tokens"] == 512
    assert all(answer[s["start"] : s["end"]] == s["text"] for s in kept["spans"])
    too_long = {**GROUNDED, "answer": GROUNDED["answer"] * 300}
    error = classify(client, "grounded", [too_long], overflow="truncate")["results"][0]
    assert error["error"] == "max_length_exceeded"


def test_inputs_fail_in_place_and_requests_fail_whole(client):
    body = classify(client, "sequence", ["fine", "", {"text_pair": "x"}])
    assert [r.get("error") for r in body["results"]] == [
        None,
        "invalid_input",
        "invalid_input",
    ]
    for options in (
        {"overflow": "window"},
        {"max_tokens": 1 << 20},
        {"overflow": "sideways"},
        {"threshold": 2},
    ):
        response = client.post(
            "/v1/classify",
            json={"model": "sequence", "input": ["x"], "options": options},
        )
        assert response.status_code == 400, options
    unknown = client.post(
        "/v1/classify", json={"model": "sequence", "input": ["x"], "head": "nope"}
    )
    assert unknown.status_code == 400
    wrong = client.post(
        "/v1/classify", json={"model": "grounded", "input": ["just text"]}
    )
    assert wrong.json()["results"][0]["error"] == "invalid_input"


def count_forwards(monkeypatch, runtime, name):
    model = served(runtime, name).model
    calls = []
    original = model.engine_model.encode

    def encode(batch):
        calls.append(list(batch.lengths))
        return original(batch)

    monkeypatch.setattr(model.engine_model, "encode", encode)
    return calls


def test_one_forward_reads_every_repeat_and_bundled_task(monkeypatch, client, runtime):
    calls = count_forwards(monkeypatch, runtime, "sequence")
    body = classify(client, "sequence", ["same", "same", "other"])
    assert len(calls) == 1 and len(calls[0]) == 2
    assert body["results"][0]["probabilities"] == body["results"][1]["probabilities"]
    calls.clear()
    bundle = client.post(
        "/v1/bundle",
        json={
            "tasks": [
                {"id": "a", "classify": {"model": "sequence", "input": ["same"]}},
                {
                    "id": "b",
                    "classify": {"model": "sequence", "input": ["same", "third"]},
                },
            ]
        },
    ).json()
    assert [r["status"] for r in bundle["results"]] == [200, 200]
    assert len(calls) == 1 and len(calls[0]) == 2


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "auto",
            marks=pytest.mark.skipif(
                torch.cuda.is_available(), reason="auto must land on the CPU"
            ),
        ),
    ],
)
def test_engines_learn_the_engines_of_their_processs_other_cpu_models(
    packages, monkeypatch, device
):
    import vllm_srun.runtime as runtime_module

    seen = []
    choose = runtime_module.choose_engine

    def recording(name, spec, device, preferred=None):
        chosen, engine = choose(name, spec, device, preferred)
        load = engine.load

        def load_and_record(spec, accelerator, device, options):
            seen.append(options.cpu_neighbors)
            return load(spec, accelerator, device, options)

        engine.load = load_and_record
        return chosen, engine

    monkeypatch.setattr(runtime_module, "choose_engine", recording)
    (name, path), (other, other_path) = list(packages.items())[:2]
    alone = ModelConfig(model=str(path), name=name, device=device)
    native = ModelConfig(
        model=str(other_path), name=other, device=device, engine="native"
    )
    for models, expected in (
        ((alone,), [frozenset()]),
        ((alone, native), [frozenset({"native"}), frozenset({"auto"})]),
    ):
        runtime = Runtime(ServeConfig(models=models, result_cache_entries=0))
        runtime.start(background=False)
        runtime.stop()
        assert seen == expected
        seen.clear()


def test_planned_engines_need_nothing_loaded():
    from vllm_srun.runtime import planned_engine

    assert (
        planned_engine(ModelConfig(model="/x", engine="onnxruntime")) == "onnxruntime"
    )
    assert planned_engine(ModelConfig(model="/no/such/package")) == "auto"
    assert planned_engine(ModelConfig(model="vllm-sr/Vela-1.0-Omni-Nano")) == "native"
    assert (
        planned_engine(ModelConfig(model="vllm-sr/Vela-1.0-Encoder-307M-Domain"))
        == "auto"
    )


@pytest.mark.parametrize("profile", ["shared_context", "batching", "max_speed"])
def test_approximate_profiles_serve_every_head(packages, profile):
    models = tuple(
        ModelConfig(model=str(path), name=name, device="cpu", profile=profile)
        for name, path in packages.items()
    )
    runtime = Runtime(ServeConfig(models=models, result_cache_entries=0))
    runtime.start(background=False)
    try:
        client = TestClient(create_app(runtime))
        for name in packages:
            inputs = [GROUNDED] if name == "grounded" else [TEXT, LONG[:200]]
            exact = classify(client, name, inputs, profile="exact")["results"]
            approximate = classify(client, name, inputs)["results"]
            for left, right in zip(exact, approximate, strict=True):
                values = "scores" if name == "scores" else "probabilities"
                if values in left:
                    torch.testing.assert_close(
                        torch.tensor(left[values]),
                        torch.tensor(right[values]),
                        rtol=0,
                        atol=0.05,
                    )
    finally:
        runtime.stop()


def test_cpu_rows_are_bit_identical_alone_and_inside_other_requests_batches(runtime):
    if not onednn.available():
        pytest.skip("oneDNN's packed linear needs an x86 CPU")
    texts = ["Hi", TEXT, LONG[:300], LONG[:120], LONG]
    for name in ("sequence", "safety", "scores", "token", "grounded"):
        model = served(runtime, name).model
        cpu_build = torch.version.hip is None and torch.version.cuda is None
        if cpu_build and torch.__version__.startswith("2.10."):
            # The images' CPU PyTorch; other builds may legitimately probe variant.
            assert model.batch_invariant, name
        if not model.batch_invariant:
            continue
        inputs = (
            [GROUNDED, {**GROUNDED, "answer": "It opened in 1889."}]
            if name == "grounded"
            else texts
        )
        plans = [
            model.plan_surface(
                "classify",
                SurfaceRequest(
                    "classify",
                    {"model": name, "input": [value]},
                    None,
                    "exact",
                    False,
                    0.0,
                ),
            )
            for value in inputs
        ]
        alone = [model.run(plan.items) for plan in plans]
        mixed = [item for plan in reversed(plans) for item in plan.items]
        together = model.run(mixed)
        position = 0
        for expected in reversed(alone):
            for value in expected:
                assert identical(together[position], value), name
                position += 1


def test_a_model_whose_rows_change_with_the_batch_is_not_batch_invariant(
    runtime, monkeypatch
):
    model = served(runtime, "sequence").model
    head = model.heads[model.primary]
    readout = head.readout
    monkeypatch.setattr(
        head,
        "readout",
        lambda rows, sequences: [
            tuple(value + 1e-7 * len(sequences) for value in values)
            for values in readout(rows, sequences)
        ],
    )
    vocab = model.engine_model.backbone.config["vocab_size"]
    assert not batch_invariant(model, vocab)


def test_the_result_cache_answers_repeated_inputs(monkeypatch, packages):
    runtime = Runtime(
        ServeConfig(
            models=(ModelConfig(model=str(packages["sequence"]), device="cpu"),)
        )
    )
    runtime.start(background=False)
    try:
        client = TestClient(create_app(runtime))
        first = classify(client, None, [TEXT])
        calls = []
        model = runtime.lookup(None).model
        original = model.engine_model.encode
        monkeypatch.setattr(
            model.engine_model,
            "encode",
            lambda b: calls.append(b) or original(b),
        )
        assert classify(client, None, [TEXT])["results"] == first["results"]
        assert calls == []
    finally:
        runtime.stop()


def test_verification_is_local_unless_a_pin_matches(monkeypatch, packages, tmp_path):
    family = TaskHeadsFamily()
    verified = family.verify(PackageRef(root=packages["token"]))
    assert verified.details["verification"] == "local" and verified.licence is None
    assert set(verified.details["files"]) == {
        "config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    }  # fmt: skip
    pinned = BuiltinModel(
        repo_id="vllm-sr/Tiny-Token",
        revision="a" * 40,
        family="task_heads",
        model_sha256=verified.model_sha256,
        manifest_sha256="",
        loaded_parameters=verified.loaded_parameters,
        backbone="modernbert",
        min_device_memory_gib=1,
        files=dict(verified.details["files"]),
    )
    monkeypatch.setattr(
        builtin, "lookup", lambda repo: pinned if repo == pinned.repo_id else None
    )
    ref = PackageRef(
        root=packages["token"], repo_id=pinned.repo_id, revision=pinned.revision
    )
    assert family.verify(ref).details["verification"] == "builtin"
    tampered = dict(pinned.files, **{"config.json": "0" * 64})
    monkeypatch.setattr(
        builtin,
        "lookup",
        lambda repo: BuiltinModel(**{**pinned.__dict__, "files": tampered}),
    )
    with pytest.raises(PackageError, match=r"config\.json"):
        family.verify(ref)


def test_unservable_checkpoints_are_refused(packages, tmp_path):
    import json
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(packages["sequence"], copy)
    config = json.loads((copy / "config.json").read_text())
    config["problem_type"] = "regression"
    (copy / "config.json").write_text(json.dumps(config))
    with pytest.raises(PackageError, match="problem_type"):
        TaskHeadsFamily().verify(PackageRef(root=copy))
    assert TaskHeadsFamily().detect(PackageRef(root=packages["grounded"]))
    assert not TaskHeadsFamily().detect(PackageRef(root=tmp_path))
