from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path
from typing import Any

import httpx
import pytest
from app import app
from chat_request import ChatRequest
from provider_contract import (
    protocol_request_field_inventory,
    provider_request_field_inventory,
)

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
    assert {case["name"] for case in cases} == protocol_request_field_inventory(
        protocol
    )
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


PROVIDER_FIELD_CASES: dict[str, dict[str, Any]] = {
    "openai_chat_completions": {
        "add_generation_prompt": True,
        "add_special_tokens": True,
        "allowed_token_ids": [7, 11],
        "bad_words": ["blocked"],
        "cache_salt": "schema-contract",
        "chat_template": "{{ messages }}",
        "chat_template_kwargs": {"enable_thinking": True},
        "continue_final_message": True,
        "documents": [{"title": "Contract", "text": "Provider fields"}],
        "echo": True,
        "ec_transfer_params": {"connector": "test"},
        "ignore_eos": True,
        "include_reasoning": False,
        "include_stop_str_in_output": True,
        "kv_transfer_params": {"connector": "test"},
        "length_penalty": 0.8,
        "logprob_token_ids": [1, 2],
        "media_io_kwargs": {"image": {"num_frames": 1}},
        "min_p": 0.05,
        "min_tokens": 2,
        "mm_processor_kwargs": {"max_pixels": 1024},
        "priority": 1,
        "prompt_logprobs": 2,
        "repetition_detection": {"max_pattern_size": 8, "min_count": 2},
        "repetition_penalty": 1.1,
        "request_id": "request-contract",
        "return_assistant_tokens_mask": True,
        "return_prompt_text": True,
        "return_token_ids": True,
        "return_token_offsets": True,
        "return_tokens_as_token_ids": True,
        "routed_experts_prompt_start": 1,
        "session_id": "session-contract",
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "stop_token_ids": [2],
        "stream_interval": 2,
        "structured_outputs": {"choice": ["yes", "no"]},
        "thinking_token_budget": 1024,
        "top_k": 20,
        "truncate_prompt_tokens": 256,
        "truncation_side": "left",
        "use_beam_search": True,
        "vllm_xargs": {"trace": "contract"},
    },
    "openai_responses": {
        "cache_salt": "schema-contract",
        "chat_template_kwargs": {"enable_thinking": True},
        "ec_transfer_params": {"connector": "test"},
        "enable_response_messages": True,
        "frequency_penalty": 0.1,
        "ignore_eos": True,
        "include_reasoning": False,
        "include_stop_str_in_output": True,
        "kv_transfer_params": {"connector": "test"},
        "logit_bias": {"42": 1.0},
        "media_io_kwargs": {"image": {"num_frames": 1}},
        "mm_processor_kwargs": {"max_pixels": 1024},
        "presence_penalty": 0.1,
        "previous_input_messages": [{"role": "user", "content": "hello"}],
        "priority": 1,
        "repetition_penalty": 1.1,
        "request_id": "response-contract",
        "seed": 7,
        "session_id": "session-contract",
        "skip_special_tokens": False,
        "stop": ["DONE"],
        "structured_outputs": {"choice": ["yes", "no"]},
        "top_k": 20,
        "vllm_xargs": {"trace": "contract"},
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "protocol", "base"),
    [
        (
            "/v1/chat/completions",
            "openai_chat_completions",
            {
                "model": "provider-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        ),
        (
            "/v1/responses",
            "openai_responses",
            {"model": "provider-model", "input": "hello"},
        ),
    ],
)
async def test_simulator_accepts_every_pinned_provider_request_field(
    path: str, protocol: str, base: dict[str, Any]
) -> None:
    cases = PROVIDER_FIELD_CASES[protocol]
    assert set(cases) == provider_request_field_inventory(protocol)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://simulator"
    ) as client:
        for field, value in cases.items():
            response = await client.post(path, json={**base, field: value})
            assert response.status_code == HTTPStatus.OK, (field, response.text)


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
        "chat_template_kwargs": {"enable_thinking": True},
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


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
async def test_simulator_rejects_invalid_provider_extension_type(path: str) -> None:
    if path.endswith("chat/completions"):
        body: dict[str, Any] = {
            "model": "provider-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
    else:
        body = {"model": "provider-model", "input": "hello"}
    body["chat_template_kwargs"] = "enable_thinking=true"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://simulator"
    ) as client:
        response = await client.post(path, json=body)
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["error"]["param"] == "chat_template_kwargs"
