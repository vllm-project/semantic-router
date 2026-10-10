"""The provider check flags exactly the fields the router rejects."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load(name: str):
    """Load a sibling script by path, so the tests run from any directory."""
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


problems = _load("check_provider").problems

# The codec's own fixtures are the contract this check has to agree with.
CODEC_TESTDATA = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "semantic-router"
    / "pkg"
    / "protocolcodec"
    / "testdata"
)


def codec_fixture(*parts: str) -> dict:
    return json.loads(CODEC_TESTDATA.joinpath(*parts).read_text())


OPENAI_REPLY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4.1-mini",
    "choices": [
        {
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "ok",
                "refusal": None,
                "annotations": [],
            },
        }
    ],
    "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 1,
        "total_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    },
    "service_tier": "default",
    "system_fingerprint": "fp_1",
}


def test_openai_shaped_reply_passes() -> None:
    assert problems(OPENAI_REPLY) == []


def test_xai_and_groq_extras_pass_with_the_router_fix() -> None:
    reply = {
        **OPENAI_REPLY,
        "x_groq": {"id": "req_1"},
        "usage": {
            **OPENAI_REPLY["usage"],
            "cost_in_usd_ticks": 100,
            "num_sources_used": 0,
            "queue_time": 0.03,
            "prompt_time": 0.001,
            "completion_time": 0.4,
            "total_time": 0.41,
        },
    }
    assert problems(reply) == []


def test_unknown_fields_bad_tier_and_empty_fingerprint_are_flagged() -> None:
    reply = {
        **OPENAI_REPLY,
        "service_tier": "future",
        "system_fingerprint": "",
        "invented_by_the_provider": "someone",
    }
    assert problems(reply) == [
        "response.invented_by_the_provider is a field the router does not accept",
        "response.service_tier 'future' is not one the router accepts",
        "response.system_fingerprint must be 1 to 256 characters when present",
    ]


@pytest.mark.parametrize(
    "fixture",
    [
        ("contracts", "openrouter-chat-response-in.json"),
        ("providers", "ollama-chat-tool-call-out.json"),
        ("providers", "vllm-chat-long-stop-sequence-out.json"),
    ],
)
def test_replies_the_codec_decodes_are_not_rejected(fixture: tuple[str, ...]) -> None:
    """These are the codec's own fixtures, so a FAIL here is a false alarm."""
    assert problems(codec_fixture(*fixture)) == []


def test_cloudflare_reply_is_flagged_because_it_needs_its_vendor() -> None:
    """Canonical decoding rejects usage.neurons; only the Cloudflare vendor drops it."""
    reply = codec_fixture("providers", "cloudflare-workers-ai-chat-out.json")
    assert problems(reply) == [
        "response.usage.neurons is a field the router does not accept"
    ]


def test_unknown_fields_inside_nested_provider_objects_are_flagged() -> None:
    reply = {
        **OPENAI_REPLY,
        "usage": {
            **OPENAI_REPLY["usage"],
            "cost": 0.0001,
            "cost_details": {"upstream_inference_cost": 0.0001, "invented": True},
            "server_tool_use": {"web_search_requests": 0, "also_invented": 1},
        },
    }
    assert problems(reply) == [
        "response.usage.cost_details.invented is a field the router does not accept",
        "response.usage.server_tool_use.also_invented is a field the router does not accept",
    ]


def test_openrouter_metadata_and_ollama_tool_call_index_pass() -> None:
    reply = {
        **OPENAI_REPLY,
        "provider": "OpenAI",
        "choices": [
            {
                **OPENAI_REPLY["choices"][0],
                "native_finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "index": 0,
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
    }
    assert problems(reply) == []


def reply_with_stop_reason(reason: object) -> dict:
    choice = {**OPENAI_REPLY["choices"][0], "stop_reason": reason}
    return {**OPENAI_REPLY, "choices": [choice]}


@pytest.mark.parametrize(
    "reason",
    [
        1,
        0,
        -1,
        9223372036854775807,
        -9223372036854775808,
        "stop",
        "a" * 128,
        "a" * 129,
        # 43 CJK characters are 129 UTF-8 bytes, over the 128-byte cap the codec dropped (#4516).
        "停" * 43,
    ],
)
def test_stop_reasons_the_codec_accepts_pass(reason: object) -> None:
    assert problems(reply_with_stop_reason(reason)) == []


@pytest.mark.parametrize(
    "reason",
    [
        # Go decodes the number into int64, so either side of its range fails.
        9223372036854775808,
        -9223372036854775809,
        1.5,
        True,
        "",
        [],
    ],
)
def test_stop_reasons_the_codec_rejects_are_flagged(reason: object) -> None:
    assert problems(reply_with_stop_reason(reason)) == [
        "choices[0].stop_reason must be a signed 64-bit integer or a non-empty string"
    ]
