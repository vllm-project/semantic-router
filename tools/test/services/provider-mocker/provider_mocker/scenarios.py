"""Named deterministic behaviors, orthogonal to the native wire codecs."""

from fastapi.responses import StreamingResponse

from . import looper
from .chat_wire import build_chat_response, build_chat_usage, generate_chat_stream
from .demo_fixtures import hallucination_text, toolcall_message
from .memory import MemoryScenario


def scenario_response(req, content, created, usage=None):
    usage = usage or build_chat_usage(req, content)
    response = build_chat_response(req, content, usage, created)
    if req.stream:
        return StreamingResponse(
            generate_chat_stream(req, response, content, usage, created),
            media_type="text/event-stream",
        )
    return response


async def respond_to_scenario(request, req, created):
    settings = request.app.state.settings
    if settings.scenario == "default":
        return None
    if settings.scenario == "looper":
        return await looper.respond(request, req)
    if settings.scenario == "memory":
        content = MemoryScenario().content([m.model_dump() for m in req.messages])
        count = len(content.split())
        usage = {
            "prompt_tokens": count,
            "completion_tokens": count,
            "total_tokens": count * 2,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
        return scenario_response(req, content, created, usage)
    if settings.scenario == "cli":
        return scenario_response(req, "ok", created)
    if settings.scenario == "hallucination":
        return scenario_response(
            req,
            hallucination_text(req.messages),
            created,
            {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
        )
    if settings.scenario == "toolcall":
        message, finish, (prompt, completion) = toolcall_message(req)
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
        response = build_chat_response(
            req, message.get("content") or "", usage, created
        )
        response["choices"][0].update(message=message, finish_reason=finish)
        return response
    raise ValueError(f"unknown fixture scenario: {settings.scenario}")
