"""Chat Completions response construction for the mock provider."""

from typing import Any

from chat_request import ChatRequest
from dynamo_contract import build_dynamo_response_nvext


def build_chat_response(
    req: ChatRequest, content: str, usage: dict[str, Any], created_ts: int
) -> dict[str, Any]:
    response = {
        "id": "cmpl-mock-123",
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "system_fingerprint": "mock-vllm",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "logprobs": build_chat_logprobs(req, content),
            }
        ],
        "usage": usage,
    }
    nvext = build_dynamo_response_nvext(req)
    if nvext is not None:
        response["nvext"] = nvext
    return response


def build_chat_logprobs(req: ChatRequest, content: str) -> dict[str, Any] | None:
    if not req.logprobs:
        return None
    token = content[:8] or "mock"
    requested = max(1, min(req.top_logprobs or 1, 5))
    alternatives = [
        {"token": token, "logprob": -1.5, "bytes": list(token.encode())},
        {"token": "other", "logprob": -1.6, "bytes": list(b"other")},
    ]
    while len(alternatives) < requested:
        index = len(alternatives)
        alternative = f"alt-{index}"
        alternatives.append(
            {
                "token": alternative,
                "logprob": -1.6 - index,
                "bytes": list(alternative.encode()),
            }
        )
    return {
        "content": [
            {
                "token": token,
                "logprob": -1.5,
                "bytes": list(token.encode()),
                "top_logprobs": alternatives[:requested],
            }
        ],
        "refusal": [],
    }
