"""Deterministic ensemble outcomes and dispatch observations."""

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()
RESPONSES = {
    "ratings-model-a": ("answer-from-ratings-model-a", 3, 2),
    "ratings-model-b": ("answer-from-ratings-model-b", 7, 4),
    "confidence-model-low": ("private-low-confidence-candidate", 5, 2),
    "confidence-model-high": ("selected-high-confidence-answer", 7, 3),
    "fusion-panel-valid": ("usable-fusion-panel-answer", 11, 3),
    "fusion-fallback-target": ("fusion-quorum-fallback-answer", 13, 5),
    "fusion-fallback-tiny-window": ("fusion-tiny-window-dispatched", 13, 5),
}


@router.get("/test/calls")
async def calls(request: Request):
    return {"calls": dict(request.app.state.dispatch_counts)}


@router.post("/test/reset")
async def reset(request: Request):
    request.app.state.dispatch_counts.clear()
    return {"status": "reset"}


def judge_content(prompt: str) -> str:
    if "return only valid JSON" in prompt:
        return json.dumps(
            {
                "consensus": ["unexpected analysis"],
                "contradictions": [],
                "partial_coverage": [],
                "unique_insights": [],
                "blind_spots": [],
            }
        )
    for marker, answer in (
        ("Compare the panel responses", "fusion-one-call-answer"),
        ("Synthesize a final answer directly", "fusion-none-answer"),
        ("Structured analysis:", "fusion-separate-answer"),
    ):
        if marker in prompt:
            return answer
    return "unexpected-fusion-synthesized-answer"


async def respond(request: Request, req):
    model = req.model
    counts = request.app.state.dispatch_counts
    counts[model] = counts.get(model, 0) + 1
    if model in {"fusion-panel-fail", "fusion-fallback-broken"}:
        detail = "panel" if model == "fusion-panel-fail" else "fallback"
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"synthetic {detail} failure"}},
        )
    if model == "fusion-panel-slow":
        await asyncio.sleep(6)
        content, prompt_tokens, completion_tokens = "too late", 9, 2
    elif model == "fusion-judge":
        prompt = req.messages[-1].content if req.messages else ""
        content, prompt_tokens, completion_tokens = judge_content(prompt), 17, 5
    elif model == "fusion-panel-empty":
        content, prompt_tokens, completion_tokens = "", 13, 0
    elif model in RESPONSES:
        content, prompt_tokens, completion_tokens = RESPONSES[model]
    else:
        return JSONResponse(
            status_code=400, content={"error": {"message": "unknown model"}}
        )
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": content},
        "finish_reason": "stop",
    }
    if model.startswith("confidence-model-"):
        choice["logprobs"] = {
            "content": [
                {
                    "token": "answer",
                    "logprob": -2.0 if model.endswith("low") else -0.1,
                    "top_logprobs": [],
                }
            ]
        }
    return {
        "id": "chatcmpl-looper-e2e-" + model,
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [] if model == "fusion-panel-empty" else [choice],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
