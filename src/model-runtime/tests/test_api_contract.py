"""Every response is validated against the checked-in OpenAPI contract."""

import copy
from pathlib import Path

import jsonschema
import pytest
import yaml
from starlette.testclient import TestClient
from vllm_sr_runtime.api.app import API_VERSION, OPENAPI_PATH, create_app
from vllm_sr_runtime.config import ModelConfig, ServeConfig
from vllm_sr_runtime.runtime import Runtime

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
    assert body["meta"]["profile"] == "exact" and body["meta"]["numerics"] == "exact"


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
    assert body["answers"]["bad"] == {"type": "set", "error": "invalid_question"}
    assert body["answers"]["tiny"] == {"type": "choice", "error": "invalid_question"}
    assert "error" not in body["answers"]["domain"]


def test_overlong_question_is_rejected_not_truncated(client):
    body = post(
        client, {"state": "word " * 5000, "questions": {"q": QUESTIONS["reasoning"]}}
    ).json()
    assert body["answers"]["q"] == {"type": "noul", "error": "max_length_exceeded"}


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


def test_models_health_metrics_and_openapi(client):
    models = client.get("/v1/models").json()
    check("ModelList", models)
    card = models["data"][0]
    assert (
        card["ready"] and card["family"] == "decision2" and card["accelerator"] == "cpu"
    )
    assert card["limits"]["max_options"] == 255 and card["limits"]["max_levels"] == 10
    assert {p["group"] for p in card["plugins"]} >= {
        "vllm_sr_runtime.families",
        "vllm_sr_runtime.engines",
    }
    health = client.get("/health")
    assert health.status_code == 200
    check("Health", health.json())
    live = client.get("/health/live").json()
    check("Health", live)
    assert live["status"] == "alive"
    assert (
        API_VERSION
        == SPEC["info"]["version"]
        == models["api_version"]
        == health.json()["api_version"]
        == live["api_version"]
    )
    metrics = client.get("/metrics").text
    assert (
        "vllm_sr_runtime_requests_total" in metrics
        and "vllm_sr_runtime_ready 1.0" in metrics
    )
    memory = [
        line
        for line in metrics.splitlines()
        if line.startswith("vllm_sr_runtime_model_memory_bytes{")
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
    check("ModelList", unready.get("/v1/models").json())


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
