"""Raw transport and expanded logical input admission tests."""

import asyncio
import json
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime.api import create_app  # noqa: E402
from decision_runtime.backend import ModelDescriptor  # noqa: E402
from decision_runtime.engine import DecisionEngine  # noqa: E402
from decision_runtime.fake_backend import FakeDecisionBackend  # noqa: E402
from decision_runtime.model_inputs import (  # noqa: E402
    QWEN_DEFAULT_NO,
    QWEN_DEFAULT_YES,
    build_model_input,
    qwen_segments,
)
from decision_runtime.request_limits import (  # noqa: E402
    BATCH_MAX_REQUEST_BYTES,
    MAX_EXPANDED_INPUT_BYTES,
    SINGLE_MAX_REQUEST_BYTES,
    BoundedJSONBodyMiddleware,
    expanded_input_bytes,
)

MODEL = ModelDescriptor(
    name="decision-test",
    description="Test model",
    release_date="2026-09-23",
)


def _app():
    return create_app(DecisionEngine(FakeDecisionBackend([MODEL])))


def _payload(state="state"):
    return {
        "model": MODEL.name,
        "state": state,
        "questions": {
            "choice": {
                "type": "choice",
                "instructions": "Choose.",
                "criteria": {"left": None, "right": None},
            }
        },
    }


def _scope(*, path="/v1/systemone", root_path=""):
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": root_path,
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }


async def _invoke_asgi(app, messages, *, scope=None):
    queued = iter(messages)
    sent = []

    async def receive():
        return next(queued, {"type": "http.disconnect"})

    async def send(message):
        sent.append(message)

    await app(scope or _scope(), receive, send)
    return sent


def test_inference_endpoints_require_uncompressed_json():
    async def scenario():
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            missing = await client.post("/v1/systemone", content=b"{}")
            assert missing.status_code == 415
            compressed = await client.post(
                "/v1/systemone",
                content=b"{}",
                headers={
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                },
            )
            assert compressed.status_code == 415

    asyncio.run(scenario())


def test_single_raw_body_limit_is_enforced_before_json_parsing():
    async def scenario():
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/systemone",
                content=b"{" + b" " * SINGLE_MAX_REQUEST_BYTES + b"}",
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 413

    asyncio.run(scenario())


def test_chunked_body_cannot_bypass_the_single_raw_limit():
    async def chunks():
        yield b"{" + b" " * (SINGLE_MAX_REQUEST_BYTES // 2)
        yield b" " * (SINGLE_MAX_REQUEST_BYTES // 2 + 1) + b"}"

    async def scenario():
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/systemone",
                content=chunks(),
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 413

    asyncio.run(scenario())


def test_oversized_chunk_is_rejected_before_buffer_growth(monkeypatch):
    import decision_runtime.request_limits as limits  # noqa: PLC0415

    class GuardedBody(bytearray):
        def extend(self, chunk):
            assert len(self) + len(chunk) <= SINGLE_MAX_REQUEST_BYTES
            super().extend(chunk)

    monkeypatch.setattr(limits, "bytearray", GuardedBody, raising=False)

    async def downstream(_scope, _receive, _send):
        raise AssertionError("oversized bodies must not reach the application")

    async def scenario():
        sent = await _invoke_asgi(
            BoundedJSONBodyMiddleware(downstream),
            [
                {
                    "type": "http.request",
                    "body": b"x" * (SINGLE_MAX_REQUEST_BYTES + 1),
                    "more_body": False,
                }
            ],
        )
        assert sent[0]["type"] == "http.response.start"
        assert sent[0]["status"] == 413

    asyncio.run(scenario())


def test_disconnect_during_body_collection_aborts_before_inference():
    class CountingBackend(FakeDecisionBackend):
        def __init__(self):
            super().__init__([MODEL])
            self.calls = 0

        async def infer(self, request):
            self.calls += 1
            return await super().infer(request)

    async def scenario():
        backend = CountingBackend()
        app = create_app(DecisionEngine(backend))
        sent = await _invoke_asgi(
            app,
            [
                {
                    "type": "http.request",
                    "body": json.dumps(_payload()).encode(),
                    "more_body": True,
                },
                {"type": "http.disconnect"},
            ],
        )
        assert backend.calls == 0
        assert sent == []

    asyncio.run(scenario())


def test_receive_after_body_replay_delegates_to_original_channel():
    observed = []

    async def downstream(_scope, receive, _send):
        observed.append(await receive())
        observed.append(await receive())

    async def scenario():
        await _invoke_asgi(
            BoundedJSONBodyMiddleware(downstream),
            [
                {
                    "type": "http.request",
                    "body": b'{"state":"buffered"}',
                    "more_body": False,
                },
                {"type": "http.disconnect"},
            ],
        )

    asyncio.run(scenario())
    assert observed == [
        {
            "type": "http.request",
            "body": b'{"state":"buffered"}',
            "more_body": False,
        },
        {"type": "http.disconnect"},
    ]


def test_raw_body_limit_is_enforced_when_path_includes_asgi_root_path():
    async def scenario():
        transport = httpx.ASGITransport(app=_app(), root_path="/gateway")
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/gateway/v1/systemone",
                content=b"{" + b" " * SINGLE_MAX_REQUEST_BYTES + b"}",
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 413

    asyncio.run(scenario())


def test_batch_has_a_larger_but_still_bounded_raw_transport_limit():
    assert BATCH_MAX_REQUEST_BYTES > SINGLE_MAX_REQUEST_BYTES

    async def scenario():
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/decision/batches",
                content=b"{" + b" " * BATCH_MAX_REQUEST_BYTES + b"}",
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 413

    asyncio.run(scenario())


def test_expanded_size_accounts_for_state_question_cartesian_product():
    single = _payload(state="x" * 10)
    batch = {
        "model": MODEL.name,
        "states": [
            {"id": "first", "state": single["state"]},
            {"id": "second", "state": single["state"]},
        ],
        "questions": single["questions"],
    }
    from decision_runtime.contracts import (  # noqa: PLC0415
        SystemOneBatchRequest,
        SystemOneRequest,
    )

    single_size = expanded_input_bytes(SystemOneRequest.model_validate(single))
    batch_size = expanded_input_bytes(SystemOneBatchRequest.model_validate(batch))
    assert batch_size > single_size
    assert batch_size < MAX_EXPANDED_INPUT_BYTES


def test_many_choice_candidates_cannot_bypass_expanded_prompt_limit():
    from decision_runtime.contracts import SystemOneBatchRequest  # noqa: PLC0415

    question = {
        "type": "choice",
        "instructions": "Choose the matching option.",
        "criteria": {f"o{index:03d}": "x" * 40 for index in range(255)},
    }
    payload = {
        "model": MODEL.name,
        "states": [
            {"id": f"state-{index}", "state": "A short state."} for index in range(32)
        ],
        "questions": {f"q{index}": question for index in range(32)},
    }
    assert len(json.dumps(payload).encode()) < BATCH_MAX_REQUEST_BYTES
    request = SystemOneBatchRequest.model_validate(payload)
    one_rendering = qwen_segments(
        build_model_input(
            question_id="q0",
            state="A short state.",
            question=request.questions["q0"],
            choice_null_description="render_key",
            noul_default_false=QWEN_DEFAULT_NO,
            noul_default_true=QWEN_DEFAULT_YES,
            noul_explicit_null="use_default",
        )
    ).rendered
    actual_bytes = len(one_rendering.encode()) * 1024
    assert actual_bytes > MAX_EXPANDED_INPUT_BYTES
    assert expanded_input_bytes(request) >= actual_bytes

    async def scenario():
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post("/v1/decision/batches", json=payload)
            assert response.status_code == 413

    asyncio.run(scenario())


def test_invalid_empty_state_and_too_many_single_questions_are_not_retryable():
    async def scenario():
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            empty = await client.post("/v1/systemone", json=_payload(state=""))
            assert empty.status_code == 422

            body = _payload()
            body["questions"] = {
                f"q{index}": {
                    "type": "noul",
                    "instructions": "Is this relevant?",
                }
                for index in range(1025)
            }
            assert len(json.dumps(body).encode()) < SINGLE_MAX_REQUEST_BYTES
            too_many = await client.post("/v1/systemone", json=body)
            assert too_many.status_code == 422

    asyncio.run(scenario())


def test_expanded_limit_rejects_small_raw_batch_with_large_logical_product(monkeypatch):
    import decision_runtime.request_limits as limits  # noqa: PLC0415

    monkeypatch.setattr(limits, "MAX_EXPANDED_INPUT_BYTES", 1)

    async def scenario():
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            payload = _payload()
            response = await client.post(
                "/v1/systemone",
                content=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 413
            assert "Expanded Decision input" in response.json()["detail"]

    asyncio.run(scenario())
