"""Every response is validated against the checked-in OpenAPI contract."""

import asyncio
import copy
import json
from pathlib import Path

import jsonschema
import pytest
import yaml
from starlette.testclient import TestClient
from vllm_srun.api.app import API_VERSION, OPENAPI_PATH, create_app
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.plugins.base import PackageRef
from vllm_srun.runtime import Runtime

from .conftest import QUESTIONS, STATE

SPEC = yaml.safe_load(Path(OPENAPI_PATH).read_text())


def schema(name):
    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(
                    SPEC["components"]["schemas"][node["$ref"].rsplit("/", 1)[1]]
                )
            node = {key: resolve(value) for key, value in node.items()}
            if node.pop("nullable", False) and "type" in node:
                node["type"] = [node["type"], "null"]
                if "enum" in node:
                    node["enum"] = [*node["enum"], None]
            return node
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(copy.deepcopy(SPEC["components"]["schemas"][name]))


def check(name, body):
    jsonschema.Draft4Validator(schema(name)).validate(body)


@pytest.fixture(scope="module")
def client(qwen3_runtime):
    return TestClient(create_app(qwen3_runtime))


def post(client, body, path="/v1/decisions"):
    return client.post(path, json=body)


def test_decisions_response_matches_the_contract(client):
    response = post(client, {"state": STATE, "questions": QUESTIONS})
    assert response.status_code == 200
    body = response.json()
    check("DecisionResponse", body)
    assert list(body["answers"]) == list(QUESTIONS)
    assert body["answers"]["domain"]["choice"] in {"code", "math", "other"}
    assert list(body["answers"]["domain"]["probabilities"]) == ["code", "math", "other"]
    assert 0 <= body["answers"]["reasoning"]["noul"] <= 1
    assert body["answers"]["difficulty"]["legend"] == {
        "0": "Trivial",
        "1": "Moderate",
        "2": "Hard",
    }
    assert body["usage"]["input_tokens"] > 0 and body["usage"]["output_tokens"] == 0
    assert "meta" not in body
    meta = post(
        client,
        {"state": STATE, "questions": QUESTIONS, "options": {"return_meta": True}},
    ).json()["meta"]
    assert meta["profile"] == "exact" and meta["numerics"] == "exact"


@pytest.mark.parametrize("repo_id", [None, "fixture-owner/Decision-2.0-Tiny-Qwen3"])
def test_metadata_keeps_loaded_identity_when_model_has_a_public_alias(
    qwen3_package, monkeypatch, repo_id
):
    revision = "a" * 40 if repo_id else None
    if repo_id:
        # Resolve the repository to tiny local weights; package verification and
        # the family's ModelInfo construction still use the real loading path.
        def resolve_fixture(model, **options):
            assert model == repo_id and options["revision"] == revision
            return PackageRef(qwen3_package, repo_id, revision)

        monkeypatch.setattr("vllm_srun.runtime.resolve", resolve_fixture)
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(
                    model=repo_id or str(qwen3_package),
                    name="public-alias",
                    revision=revision,
                    device="cpu",
                ),
            )
        )
    )
    runtime.start(background=False)
    try:
        request = {
            "model": "public-alias",
            "state": STATE,
            "questions": QUESTIONS,
            "states": {"second": {"state": STATE, "questions": QUESTIONS}},
            "options": {"return_meta": True},
        }
        client = TestClient(create_app(runtime))
        response = post(client, request, "/v1/systemone")
        assert response.status_code == 200
        body = response.json()
        check("DecisionResponse", body)
        info = runtime.lookup("public-alias").model.info
        assert info.id == "Decision-2.0-Tiny-Qwen3" and info.repo == repo_id
        expected_id = repo_id or info.id
        for envelope in (body, body["states"]["second"]):
            assert envelope["model"] == "public-alias"
            assert envelope["meta"]["model_id"] == expected_id
            assert envelope["meta"]["revision"] == revision
            assert envelope["meta"]["model_sha256"] == info.model_sha256
        request["options"] = {"return_meta": False}
        response = post(client, request, "/v1/systemone")
        hidden = response.json()
        assert response.status_code == 200 and "meta" not in hidden
        assert "meta" not in hidden["states"]["second"]
        assert hidden["answers"] == body["answers"]
        assert (
            hidden["states"]["second"]["answers"] == body["states"]["second"]["answers"]
        )
    finally:
        runtime.stop()


def test_systemone_alias_answers_identically(client):
    first = post(client, {"state": STATE, "questions": QUESTIONS}).json()
    second = post(
        client, {"state": STATE, "questions": QUESTIONS}, "/v1/systemone"
    ).json()
    assert first["answers"] == second["answers"] and first["usage"] == second["usage"]


def test_system_one_requests_are_accepted_unchanged(client):
    body = {
        "state": {"user": STATE, "turn": 2},
        "questions": {"q": {"type": "noul", "instructions": ["Is it", "code?"]}},
    }
    response = post(client, body)
    assert response.status_code == 200
    assert set(response.json()) >= {"model", "answers", "usage"}


def test_per_question_errors_do_not_fail_siblings(client):
    questions = dict(
        QUESTIONS,
        bad={"type": "set", "instructions": "x"},
        tiny={"type": "choice", "instructions": "x", "criteria": {"a": "A"}},
    )
    body = post(client, {"state": STATE, "questions": questions}).json()
    check("DecisionResponse", body)
    bad, tiny = body["answers"]["bad"], body["answers"]["tiny"]
    assert (bad["type"], bad["error"]) == ("set", "invalid_question")
    assert (tiny["type"], tiny["error"]) == ("choice", "invalid_question")
    # Each failed question says why, naming the field.
    assert bad["message"] and "criteria" in tiny["message"]
    assert "error" not in body["answers"]["domain"]


def test_a_request_without_a_valid_question_is_refused(client):
    response = post(
        client,
        {
            "state": STATE,
            "questions": {
                "bad": {"type": "noul", "instructions": "x", "colour": "blue"},
                "worse": {"type": "rank", "instructions": "x"},
            },
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert "bad: noul questions do not take ['colour']" in error["message"]
    assert "worse: type must be one of" in error["message"]


CHOICES = [{"key": "a"}, {"key": "b"}]
SYSTEM_ONE_CASES = {
    "choice via choices": ("ok", {"type": "choice", "choices": CHOICES}),
    "choice, null description": (
        "ok",
        {"type": "choice", "criteria": {"a": None, "b": "B"}},
    ),
    "score via levels": ("ok", {"type": "score", "levels": ["low", "mid", "high"]}),
    "noul, bare": ("ok", {"type": "noul"}),
    "noul, false and true": (
        "ok",
        {"type": "noul", "criteria": {"false": "n", "true": "y"}},
    ),
    "noul, null description": ("ok", {"type": "noul", "criteria": {"true": None}}),
    "instructions as object": ("ok", {"type": "noul", "instructions": {"q": "Code?"}}),
    "choice, one option": (
        "invalid_question",
        {"type": "choice", "criteria": {"a": "A"}},
    ),
    "choice, duplicate keys": (
        "invalid_question",
        {"type": "choice", "choices": [{"key": "a"}] * 2},
    ),
    "choice, blank key": (
        "invalid_question",
        {"type": "choice", "criteria": {" ": "A", "b": "B"}},
    ),
    "choice, empty description": (
        "invalid_question",
        {"type": "choice", "criteria": {"a": "", "b": "B"}},
    ),
    "choice, criteria and choices": (
        "invalid_question",
        {"type": "choice", "criteria": {"a": "A", "b": "B"}, "choices": CHOICES},
    ),
    "choice, levels": ("invalid_question", {"type": "choice", "levels": ["a", "b"]}),
    "score, eleven levels": (
        "invalid_question",
        {"type": "score", "criteria": [str(i) for i in range(11)]},
    ),
    "score, null level": (
        "invalid_question",
        {"type": "score", "criteria": ["low", None]},
    ),
    "noul, other key": (
        "invalid_question",
        {"type": "noul", "criteria": {"maybe": "?"}},
    ),
    "blank instructions": ("invalid_question", {"type": "noul", "instructions": "   "}),
    "no instructions": ("invalid_question", {"type": "noul", "instructions": None}),
    "unknown field": ("invalid_question", {"type": "noul", "colour": "blue"}),
    "no type": ("invalid_question", {}),
}


@pytest.fixture(scope="module")
def decision_runtimes(tmp_path_factory, qwen3_runtime):
    from vllm_srun.testing import decision1
    from vllm_srun.testing.vela2 import write_encoder_package

    root = tmp_path_factory.mktemp("systemone")
    packages = {
        "decision1-vela": decision1.write_fixture(root / "vela", "vela-encoder", 0),
        "decision1-qwen": decision1.write_fixture(root / "qwen", None, 0),
        "vela2": write_encoder_package(root / "vela2"),
    }
    started = {"decision2": qwen3_runtime}
    for name, package in packages.items():
        runtime = Runtime(
            ServeConfig(models=(ModelConfig(model=str(package), device="cpu"),))
        )
        runtime.start(background=False)
        started[name] = runtime
    yield started
    for name, runtime in started.items():
        if name != "decision2":
            runtime.stop()


def test_every_family_validates_system_one_questions_alike(decision_runtimes):
    questions = {
        case: {"instructions": "Pick one", **question}
        for case, (_, question) in SYSTEM_ONE_CASES.items()
    }
    expected = {case: outcome for case, (outcome, _) in SYSTEM_ONE_CASES.items()}
    for name, runtime in decision_runtimes.items():
        status, body = asyncio.run(
            runtime.call("decisions", {"state": STATE, "questions": questions})
        )
        assert status == 200, name
        check("DecisionResponse", body)
        outcomes = {
            case: answer.get("error", "ok") for case, answer in body["answers"].items()
        }
        assert outcomes == expected, name


def test_every_family_refuses_blank_question_ids(decision_runtimes):
    questions = {"   ": {"type": "noul", "instructions": "Pick one"}}
    for name, runtime in decision_runtimes.items():
        status, body = asyncio.run(
            runtime.call("decisions", {"state": STATE, "questions": questions})
        )
        assert (status, body["error"]["code"]) == (400, "invalid_request"), name


def test_the_contract_lists_every_question_field():
    question = schema("Question")
    assert question["additionalProperties"] is False
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft4Validator(question).validate({"type": "noul", "colour": "x"})


def test_overlong_question_is_rejected_not_truncated(client):
    body = post(
        client, {"state": "word " * 5000, "questions": {"q": QUESTIONS["reasoning"]}}
    ).json()
    answer = body["answers"]["q"]
    assert (answer["type"], answer["error"]) == ("noul", "max_length_exceeded")
    assert "exceeds max_length" in answer["message"]


def test_deadline_exceeded_is_reported_per_question(client):
    body = post(
        client,
        {"state": STATE, "questions": QUESTIONS, "options": {"deadline_ms": 0.001}},
    ).json()
    check("DecisionResponse", body)
    assert {answer["error"] for answer in body["answers"].values()} == {
        "deadline_exceeded"
    }


@pytest.mark.parametrize(
    "body,status,code",
    [
        ({"questions": QUESTIONS}, 400, "invalid_request"),
        ({"state": STATE, "questions": {}}, 400, "invalid_request"),
        ({"state": 3, "questions": QUESTIONS}, 400, "invalid_request"),
        ({"state": STATE, "questions": QUESTIONS, "extra": 1}, 400, "invalid_request"),
        (
            {"state": STATE, "questions": QUESTIONS, "options": {"deadline_ms": -1}},
            400,
            "invalid_request",
        ),
        (
            {
                "state": STATE,
                "questions": QUESTIONS,
                "options": {"profile": "max_speed"},
            },
            400,
            "invalid_request",
        ),
        (
            {"state": STATE, "questions": QUESTIONS, "options": {"stream": True}},
            400,
            "invalid_request",
        ),
        (
            {"state": STATE, "questions": QUESTIONS, "model": "someone/else"},
            404,
            "model_not_found",
        ),
    ],
)
def test_request_errors(client, body, status, code):
    response = post(client, body)
    assert response.status_code == status
    check("ErrorResponse", response.json())
    assert response.json()["error"]["code"] == code


def test_model_may_be_named(client, qwen3_runtime):
    response = post(
        client,
        {
            "model": qwen3_runtime.lookup(None).served_id,
            "state": STATE,
            "questions": QUESTIONS,
        },
    )
    assert response.status_code == 200


def test_body_that_is_not_json(client):
    response = client.post(
        "/v1/decisions", content=b"{", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400


@pytest.mark.parametrize(
    "field,old,surrogate",
    [
        ("state", '"Write a', b"\\ud800"),
        ("instructions", '"Which domain', b"\\ud800"),
        ("question id", '"domain"', b"\\ud800"),
        ("criteria key", '"code"', b"\\ud800"),
        ("state as raw bytes", '"Write a', b"\xed\xa0\x80"),
    ],
)
def test_unpaired_surrogate_is_an_invalid_request(client, field, old, surrogate):
    raw = json.dumps({"state": STATE, "questions": QUESTIONS}).encode()
    old = old.encode()
    assert raw.count(old) == 1, field
    response = client.post(
        "/v1/decisions",
        content=raw.replace(old, old[:3] + surrogate + old[3:]),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    check("ErrorResponse", response.json())
    assert response.json()["error"]["code"] == "invalid_request"
    assert client.get("/health").json()["status"] == "ready"


def test_escaped_surrogate_pair_is_accepted(client):
    raw = json.dumps({"state": STATE + " \U0001f600", "questions": QUESTIONS})
    assert "\\ud83d\\ude00" in raw
    response = client.post(
        "/v1/decisions",
        content=raw.encode(),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    "name, body",
    [
        ("ClassifyRequest", {"input": "one text"}),
        ("ClassifyRequest", {"input": ["a", "b"]}),
        ("ClassifyRequest", {"input": [{"text": "query", "text_pair": "document"}]}),
        (
            "ClassifyRequest",
            {"input": [{"context": "c", "question": "q", "answer": "a"}]},
        ),
        ("EmbeddingsRequest", {"input": "one text"}),
        ("EmbeddingsRequest", {"input": ["a", "b"]}),
        (
            "EmbeddingsRequest",
            {
                "input": [
                    {"type": "text", "text": "a"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "AA==", "format": "wav"},
                    },
                ]
            },
        ),
        ("Embedding", {"object": "embedding", "index": 0, "embedding": [0.5, -0.5]}),
        ("Embedding", {"object": "embedding", "index": 0, "embedding": "AAAAPw=="}),
    ],
)
def test_surface_payloads_match_the_contract(name, body):
    check(name, body)


@pytest.mark.parametrize(
    "name, body",
    [
        ("ClassifyRequest", {"input": []}),
        ("ClassifyRequest", {"input": [1]}),
        ("ClassifyRequest", {"input": [{"txt": "a"}]}),
        ("EmbeddingsRequest", {"input": [{"type": "video_url"}]}),
        ("EmbeddingsRequest", {"input": [{"type": "image_url", "image_url": {}}]}),
        ("Embedding", {"object": "embedding", "index": 0, "embedding": ["a"]}),
        (
            "RerankRequest",
            {"query": "q", "documents": ["d"], "options": {"profile": "Fast!"}},
        ),
    ],
)
def test_malformed_surface_payloads_fail_the_contract(name, body):
    with pytest.raises(jsonschema.ValidationError):
        check(name, body)


def test_models_health_metrics_and_openapi(client):
    models = client.get("/v1/models").json()
    check("ModelList", models)
    card = models["data"][0]
    assert (
        card["ready"] and card["family"] == "decision2" and card["accelerator"] == "cpu"
    )
    assert card["limits"]["max_options"] == 255 and card["limits"]["max_levels"] == 10
    assert {p["group"] for p in card["plugins"]} >= {
        "vllm_srun.families",
        "vllm_srun.engines",
    }
    health = client.get("/health")
    assert health.status_code == 200
    check("Health", health.json())
    live = client.get("/health/live").json()
    check("Liveness", live)
    assert live == {"api_version": API_VERSION, "status": "alive"}
    assert (
        API_VERSION
        == SPEC["info"]["version"]
        == models["api_version"]
        == health.json()["api_version"]
        == live["api_version"]
    )
    metrics = client.get("/metrics").text
    assert "vllm_srun_requests_total" in metrics and "vllm_srun_ready 1.0" in metrics
    memory = [
        line
        for line in metrics.splitlines()
        if line.startswith("vllm_srun_model_memory_bytes{")
    ]
    assert len(memory) == 1 and float(memory[0].rsplit(" ", 1)[1]) > 0
    assert client.get("/openapi.yaml").text == Path(OPENAPI_PATH).read_text()


def test_unready_runtime_answers_503(qwen3_package):
    runtime = Runtime(
        ServeConfig(models=(ModelConfig(model=str(qwen3_package), device="cpu"),))
    )
    unready = TestClient(create_app(runtime))
    response = unready.post(
        "/v1/decisions", json={"state": STATE, "questions": QUESTIONS}
    )
    assert (
        response.status_code == 503 and response.json()["error"]["code"] == "not_ready"
    )
    health = unready.get("/health")
    assert health.status_code == 503
    check("Health", health.json())
    live = unready.get("/health/live")
    assert live.status_code == 200
    check("Liveness", live.json())
    check("ModelList", unready.get("/v1/models").json())


@pytest.mark.parametrize(
    "name, status", [("Health", "alive"), ("Liveness", "ready"), ("Liveness", "failed")]
)
def test_liveness_and_readiness_states_do_not_mix(name, status):
    with pytest.raises(jsonschema.ValidationError):
        check(name, {"api_version": API_VERSION, "status": status})


def test_request_size_limit(qwen3_runtime):
    original = qwen3_runtime.config
    object.__setattr__(
        qwen3_runtime,
        "config",
        original.__class__(**{**original.__dict__, "max_request_bytes": 64}),
    )
    try:
        response = TestClient(create_app(qwen3_runtime)).post(
            "/v1/decisions", json={"state": STATE, "questions": QUESTIONS}
        )
        assert response.status_code == 413
    finally:
        object.__setattr__(qwen3_runtime, "config", original)


def test_models_report_the_process_limits_a_bundle_must_fit(client, qwen3_runtime):
    config = qwen3_runtime.config
    limits = client.get("/v1/models").json()["limits"]
    assert limits == {
        "max_bundle_tasks": config.max_bundle_tasks,
        "max_request_bytes": config.max_request_bytes,
    }
    task = {"decisions": {"state": STATE, "questions": QUESTIONS}}
    tasks = [{"id": str(i), **task} for i in range(limits["max_bundle_tasks"] + 1)]
    response = client.post("/v1/bundle", json={"tasks": tasks})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


@pytest.mark.parametrize("path", ["/v1/classify", "/v1/bundle"])
def test_a_client_that_disconnects_cancels_its_call(path):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    cancelled = []

    async def scenario():
        started = asyncio.Event()

        async def work(*args):
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.append(path)
                raise

        runtime = SimpleNamespace(
            config=SimpleNamespace(max_request_bytes=1024),
            metrics=MagicMock(),
            call=work,
            bundle=work,
        )
        messages = [{"type": "http.request", "body": b"{}", "more_body": False}]

        async def receive():
            if messages:
                return messages.pop()
            await started.wait()
            return {"type": "http.disconnect"}

        sent = []

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "query_string": b"",
            "root_path": "",
        }
        await create_app(runtime)(scope, receive, send)
        return runtime, sent

    runtime, sent = asyncio.run(scenario())
    assert cancelled == [path]
    assert sent[0]["status"] == 499
    runtime.metrics.requests.labels.assert_called_with(endpoint=path, status="499")
