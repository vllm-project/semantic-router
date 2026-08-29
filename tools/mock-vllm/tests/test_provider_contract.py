from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path
from typing import Any

import httpx
import pytest
from app import app
from chat_request import ChatRequest
from provider_contract import request_field_inventory

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CAPABILITY_FIXTURES = (
    REPOSITORY_ROOT
    / "src"
    / "semantic-router"
    / "pkg"
    / "protocolcodec"
    / "testdata"
    / "golden"
    / "capability"
)


def official_field_cases(name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with (CAPABILITY_FIXTURES / name).open(encoding="utf-8") as fixture_file:
        fixture = json.load(fixture_file)
    return fixture["base"], fixture["cases"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "protocol", "fixture"),
    [
        (
            "/v1/chat/completions",
            "openai_chat_completions",
            "013-chat-official-request-fields-in.json",
        ),
        (
            "/v1/responses",
            "openai_responses",
            "014-responses-official-request-fields-in.json",
        ),
    ],
)
async def test_simulator_accepts_every_published_request_field(
    path: str, protocol: str, fixture: str
) -> None:
    base, cases = official_field_cases(fixture)
    assert {case["name"] for case in cases} == request_field_inventory(protocol)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://simulator"
    ) as client:
        for case in cases:
            response = await client.post(path, json={**base, **case["patch"]})
            assert response.status_code == HTTPStatus.OK, (
                case["name"],
                response.text,
            )


def test_chat_parser_preserves_nested_official_fields() -> None:
    body = {
        "model": "provider-model",
        "messages": [
            {
                "role": "assistant",
                "content": "done",
                "name": "worker",
                "refusal": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            }
        ],
        "metadata": {"trace": "schema-contract"},
        "max_tokens": 64,
    }
    parsed = ChatRequest.model_validate(body).model_dump(exclude_unset=True)
    assert parsed == body


@pytest.mark.asyncio
async def test_debug_endpoint_preserves_the_native_provider_request() -> None:
    session_id = "openai-contract-session"
    body = {
        "model": "provider-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,aGVsbG8=",
                            "detail": "high",
                        },
                    }
                ],
            }
        ],
        "metadata": {"trace": "schema-contract"},
        "max_tokens": 64,
    }
    headers = {"x-vsr-test-session-id": session_id}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://simulator"
    ) as client:
        response = await client.post("/v1/chat/completions", json=body, headers=headers)
        assert response.status_code == HTTPStatus.OK
        observed = await client.get("/debug/last-request", headers=headers)
    assert observed.status_code == HTTPStatus.OK
    assert observed.json()["body"] == body


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
async def test_simulator_rejects_unknown_provider_fields(path: str) -> None:
    body: dict[str, Any]
    if path.endswith("chat/completions"):
        body = {
            "model": "provider-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
    else:
        body = {"model": "provider-model", "input": "hello"}
    body["silently_swallowed"] = True
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://simulator"
    ) as client:
        response = await client.post(path, json=body)
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["error"]["param"] == "silently_swallowed"
