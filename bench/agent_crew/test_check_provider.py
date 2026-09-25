"""The provider check flags exactly the fields the router rejects."""

import importlib.util
import sys
from pathlib import Path


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
        "provider": "someone",
    }
    assert problems(reply) == [
        "response.provider is a field the router does not accept",
        "response.service_tier 'future' is not one the router accepts",
        "response.system_fingerprint must be 1 to 256 characters when present",
    ]
