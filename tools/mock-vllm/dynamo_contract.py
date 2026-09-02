"""Deterministic NVIDIA Dynamo wire-contract fixtures for the mock provider."""

from typing import Any

from chat_request import ChatRequest


def build_dynamo_response_nvext(req: ChatRequest) -> dict[str, Any] | None:
    """Return only response metadata selected by request nvext.extra_fields."""
    if not isinstance(req.nvext, dict):
        return None
    requested = req.nvext.get("extra_fields")
    if not isinstance(requested, list):
        return None
    fixtures: dict[str, Any] = {
        "worker_id": {
            "prefill_worker_id": 11,
            "prefill_dp_rank": 1,
            "decode_worker_id": 22,
            "decode_dp_rank": 2,
        },
        "timing": {
            "request_received_ms": 1000,
            "prefill_wait_time_ms": 1.25,
            "prefill_time_ms": 2.5,
            "ttft_ms": 3.75,
            "total_time_ms": 8.0,
        },
        "routed_experts": [[1, 3]],
        "engine_data": {"mock": "dynamo"},
        "stop_reason": "stop",
        "completion_token_ids": [101, 102],
        "prompt_logprobs": [
            {"101": {"logprob": -0.25, "rank": 1, "decoded_token": "mock"}}
        ],
    }
    response = {
        field: fixtures[field]
        for field in requested
        if isinstance(field, str) and field in fixtures
    }
    return response or None
