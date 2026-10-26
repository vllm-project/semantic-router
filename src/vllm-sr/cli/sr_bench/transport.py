"""Bounded OpenAI-compatible transport with final-channel-only extraction."""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from http import HTTPStatus
from typing import Any

import requests

from . import native_output
from .activity import CHECKPOINT_SECONDS

MAX_USAGE_RECEIPT_BYTES = 12288
MAX_RECEIPT_CALLS = 256
SESSION_PHASE_HEADER = "x-vsr-session-phase"
USER_TURN_PHASE = "user_turn"
TOOL_LOOP_PHASE = "tool_loop"
PROVIDER_STATE_PHASE = "provider_state"
UNKNOWN_PHASE = "unknown"
SESSION_PHASES = frozenset({USER_TURN_PHASE, TOOL_LOOP_PHASE, PROVIDER_STATE_PHASE})
SESSION_ID_HEADER = "x-session-id"
CACHE_OBSERVATION_VERSION: str = "provider-cache-observation-v1"
CACHE_REPORTING_STATES: tuple[str, ...] = ("missing", "reported_zero", "positive")
# Native precedence: https://github.com/vllm-project/semantic-router/blob/a6d23eaff862a9d24dc10ccb778ad0618a839ef7/bench/cache_token_probe.py#L26-L37
CACHE_READ_PATHS: tuple[tuple[str, ...], ...] = (
    ("prompt_tokens_details", "cached_tokens"),
    ("input_tokens_details", "cached_tokens"),
    ("cached_tokens",),
    ("prompt_cache_hit_tokens",),
)
CACHE_WRITE_CONFLICT: str = "Conflicting cache-write token usage"


# Shared internal transport/harness contract; preserving its exception identity.
class CallFailure(RuntimeError):  # noqa: N818
    def __init__(self, message, partial=None):
        super().__init__(message)
        self.partial = partial or {}


def final_content(content):
    # Explicit reasoning tags may be emitted inside content by self-hosted models.
    if "<think>" in content:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
        if "<think>" in content:
            return ""
    return content.strip()


def _is_tool_result_message(message: dict) -> bool:
    if message.get("role") in {"tool", "function"}:
        return True
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(part, dict) and part.get("type") == "tool_result" for part in content
    )


def request_phase(messages: list[dict]) -> str:
    """Apply the Router's tool-loop rule to the request's latest messages."""
    if not messages or not isinstance(messages[-1], dict):
        return USER_TURN_PHASE
    latest = messages[-1]
    role = latest.get("role")
    if _is_tool_result_message(latest) or (
        role == "assistant" and latest.get("tool_calls")
    ):
        return TOOL_LOOP_PHASE
    if role == "user" and len(messages) > 1:
        previous = messages[-2]
        if isinstance(previous, dict) and _is_tool_result_message(previous):
            return TOOL_LOOP_PHASE
    return USER_TURN_PHASE


def session_phase(
    messages: list[dict], response_headers: Mapping[str, str] | None = None
) -> tuple[str, str]:
    """Prefer the Router phase while retaining a bounded request fallback."""
    value = (response_headers or {}).get(SESSION_PHASE_HEADER)
    if isinstance(value, str) and value.strip():
        phase = value.strip()
        return (phase if phase in SESSION_PHASES else UNKNOWN_PHASE), "router"
    return request_phase(messages), "request"


def usage_presence(raw: object) -> tuple[bool, bool]:
    """Return whether cache-read and cache-write fields were actually present."""
    if not isinstance(raw, dict):
        return False, False
    details = raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}
    if not isinstance(details, dict):
        return False, False
    read_reported = (
        details.get("cached_tokens") is not None
        or raw.get("cache_read_input_tokens") is not None
    )
    write_reported = (
        details.get("cache_creation_tokens") is not None
        or details.get("created_cache_tokens") is not None
        or details.get("cache_write_tokens") is not None
        or raw.get("cache_creation_input_tokens") is not None
    )
    return read_reported, write_reported


def normalize_usage(raw):
    if not isinstance(raw, dict):
        raise CallFailure("Invalid token usage object")
    details = raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}
    if not isinstance(details, dict):
        raise CallFailure("Invalid token usage details")
    total = raw.get("prompt_tokens", raw.get("input_tokens"))
    output = raw.get("completion_tokens", raw.get("output_tokens"))
    cached = details.get("cached_tokens", raw.get("cache_read_input_tokens", 0))
    writes = [
        value
        for value in (
            details.get("cache_write_tokens"),
            details.get("cache_creation_tokens"),
            details.get("created_cache_tokens"),
            raw.get("cache_creation_input_tokens"),
        )
        if value is not None
    ]
    if any(value != writes[0] for value in writes):
        raise CallFailure(CACHE_WRITE_CONFLICT)
    written = writes[0] if writes else 0
    if total is None or output is None:
        return None
    values = (total, output, cached, written)
    if (
        any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values)
        or cached + written > total
    ):
        raise CallFailure("Invalid token usage buckets")
    return {
        "input_tokens": total - cached - written,
        "cached_input_tokens": cached,
        "cache_write_tokens": written,
        "output_tokens": output,
    }


def observe_cache_usage(raw: object) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "version": CACHE_OBSERVATION_VERSION,
        "scope": "response_usage",
        "field_present": False,
        "source": None,
        "sources": [],
        "values": {},
        "qualification": "missing",
        "state": "missing",
        "cached_tokens": None,
        "usage": None,
        "reasons": [],
    }
    if raw is None:
        return {**observation, "reasons": ["usage_missing"]}
    if not isinstance(raw, dict):
        return {
            **observation,
            "qualification": "malformed",
            "state": None,
            "reasons": ["invalid_usage_object"],
        }
    fields: dict[str, Any] = {}
    malformed_details = False
    for path in CACHE_READ_PATHS:
        value: Any = raw
        for part in path:
            if not isinstance(value, dict):
                malformed_details |= value is not None
                value = None
                break
            value = value.get(part)
        if value is not None:
            fields["usage." + ".".join(path)] = value
    observation.update(
        field_present=bool(fields),
        sources=list(fields),
        values={
            source: value if type(value) is int else None
            for source, value in fields.items()
        },
    )
    total = raw.get("prompt_tokens", raw.get("input_tokens"))
    output = raw.get("completion_tokens", raw.get("output_tokens"))
    missed = raw.get("prompt_cache_miss_tokens")
    if any(
        value is not None and (type(value) is not int or value < 0)
        for value in (total, output, missed)
    ):
        return {
            **observation,
            "qualification": "malformed",
            "state": None,
            "reasons": ["invalid_reported_token_count"],
        }
    if malformed_details or any(
        type(value) is not int or value < 0 for value in fields.values()
    ):
        return {
            **observation,
            "qualification": "malformed",
            "state": None,
            "reasons": ["invalid_cache_field"],
        }
    if not fields:
        return {**observation, "reasons": ["cache_field_missing"]}
    if len(set(fields.values())) != 1:
        return {
            **observation,
            "qualification": "conflicting",
            "state": None,
            "reasons": ["cache_read_aliases_disagree"],
        }
    verified = {
        source: value
        for source, value in fields.items()
        if source != "usage.cached_tokens"
    }
    if not verified:
        return {
            **observation,
            "qualification": "unqualified",
            "state": None,
            "reasons": ["unverified_legacy_cache_field"],
        }
    source, cached = next(iter(verified.items()))
    if total is not None and cached > total:
        return {
            **observation,
            "qualification": "malformed",
            "state": None,
            "reasons": ["cache_read_exceeds_prompt"],
        }
    if total is not None and missed is not None and cached + missed != total:
        return {
            **observation,
            "qualification": "conflicting",
            "state": None,
            "reasons": ["cache_hit_miss_total_disagree"],
        }
    # DeepSeek's hit field duplicates the nested count; this projection is not billing.
    # https://api-docs.deepseek.com/api/create-chat-completion/
    details = raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}
    projection = {**raw, "prompt_tokens_details": {**details, "cached_tokens": cached}}
    try:
        normalized = normalize_usage(projection)
    except CallFailure as exc:
        return {
            **observation,
            "qualification": (
                "conflicting" if str(exc) == CACHE_WRITE_CONFLICT else "malformed"
            ),
            "state": None,
            "reasons": [str(exc)],
        }
    observation.update(
        source=source,
        qualification="qualified" if normalized is not None else "incomplete",
        state="positive" if cached else "reported_zero",
        cached_tokens=cached,
        usage=normalized,
        reasons=[] if normalized is not None else ["incomplete_usage"],
    )
    return observation


def cost_for(usage, model, prices):
    p = prices.get(model)
    if usage is None or p is None:
        return None
    return (
        sum(
            usage[k] * p[b]
            for k, b in (
                ("input_tokens", "input"),
                ("cached_input_tokens", "cached_input"),
                ("cache_write_tokens", "cache_write"),
                ("output_tokens", "output"),
            )
        )
        / 1_000_000
    )


def mom_usage(
    response_headers,
    response_usage,
    prices,
    *,
    cache_read_reported: bool | None = None,
    cache_write_reported: bool | None = None,
):
    """Account every traced child call; never price MoM tokens as its final model."""
    raw = response_headers.get("x-vsr-model-usage")
    if raw:
        if len(raw.encode()) > MAX_USAGE_RECEIPT_BYTES:
            raise CallFailure("MoM usage receipt exceeds bounded contract")
        try:
            receipt = json.loads(raw)
        except ValueError as exc:
            raise CallFailure("Invalid MoM usage receipt") from exc
        if (
            not isinstance(receipt, dict)
            or type(receipt.get("version")) is not int
            or receipt["version"] != 1
            or not isinstance(receipt.get("complete"), bool)
            or not isinstance(receipt.get("calls"), list)
        ):
            raise CallFailure("Invalid MoM usage receipt contract")
        calls = receipt["calls"]
        if not calls and not receipt["complete"]:
            # The Router emits this sentinel when no complete call trace is available.
            return {
                "usage": None,
                "cost_usd": None,
                "model_usage": [],
                "inference_call_count": None,
                "usage_complete": False,
                "cache_read_reported": None,
                "cache_write_reported": None,
            }
        if not 1 <= len(calls) <= MAX_RECEIPT_CALLS:
            raise CallFailure("Invalid MoM inference call count")
        breakdown = []
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("model"), str):
                raise CallFailure("Invalid MoM child identity")
            raw_usage = call.get("usage") or {}
            normalized_usage = {**raw_usage}
            if "cached_input_tokens" in raw_usage:
                normalized_usage["cache_read_input_tokens"] = raw_usage[
                    "cached_input_tokens"
                ]
            if "cache_write_tokens" in raw_usage:
                normalized_usage["cache_creation_input_tokens"] = raw_usage[
                    "cache_write_tokens"
                ]
            usage = normalize_usage(normalized_usage)
            breakdown.append(
                {
                    **call,
                    "usage": usage,
                    "cache_read_reported": None,
                    "cache_write_reported": None,
                    "cost_usd": cost_for(usage, call["model"], prices),
                }
            )
        complete = receipt["complete"] and all(
            c["usage"] is not None for c in breakdown
        )
        usage = (
            {
                k: sum(c["usage"][k] for c in breakdown)
                for k in (
                    "input_tokens",
                    "cached_input_tokens",
                    "cache_write_tokens",
                    "output_tokens",
                )
            }
            if complete
            else None
        )
        cost = (
            sum(c["cost_usd"] for c in breakdown)
            if complete and all(c["cost_usd"] is not None for c in breakdown)
            else None
        )
        return {
            "usage": usage,
            "cost_usd": cost,
            "model_usage": breakdown,
            "inference_call_count": len(calls),
            "usage_complete": complete,
            "cache_read_reported": None,
            "cache_write_reported": None,
        }
    if response_headers.get("x-vsr-inference-call-count") == "1":
        selected = response_headers.get("x-vsr-selected-model")
        if selected:
            return {
                "usage": response_usage,
                "cost_usd": cost_for(response_usage, selected, prices),
                "inference_call_count": 1,
                "usage_complete": response_usage is not None,
                "cache_read_reported": cache_read_reported,
                "cache_write_reported": cache_write_reported,
            }
    return {
        "usage": None,
        "cost_usd": None,
        "inference_call_count": None,
        "usage_complete": False,
        "cache_read_reported": cache_read_reported,
        "cache_write_reported": cache_write_reported,
    }


def effective_request(target, messages, sampling, extra_body=None):
    extras = extra_body or {}
    if set(extras) - {
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "functions",
        "function_call",
        "metadata",
    }:
        raise CallFailure("Adapter attempted to override frozen request parameters")
    return {
        **sampling,
        **target.get("request_params", {}),
        **extras,
        "model": target["model"],
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def chat(
    target,
    messages,
    sampling,
    limits,
    cancelled,
    extra_body=None,
    stream_path=None,
    output_policy="bounded",
    activity=None,
    session_id: str | None = None,
):
    started = time.monotonic()
    endpoint = target["base_url"].rstrip("/") + "/chat/completions"
    body = effective_request(target, messages, sampling, extra_body)
    if output_policy not in {"bounded", "native"}:
        raise CallFailure("Unknown output policy")
    if output_policy == "native" and "max_tokens" in body:
        raise CallFailure("Native output forbids an explicit output cap")
    if body.get("max_tokens", 0) > limits["max_output_tokens"]:
        raise CallFailure("Requested tokens exceed frozen cap")
    headers = {"Content-Type": "application/json"}
    if target["kind"] == "mom":
        headers["X-VSR-Debug"] = "true"
    if session_id is not None:
        headers[SESSION_ID_HEADER] = session_id
    if target.get("api_key_env"):
        key = os.environ.get(target["api_key_env"])
        if not key:
            raise CallFailure("Target credential environment variable is not set")
        headers["Authorization"] = "Bearer " + key
    if target.get("config_hash"):
        headers["X-SR-Bench-Expected-Config-Hash"] = target["config_hash"]
    if target.get("max_inference_calls"):
        headers["X-SR-Bench-Max-Inference-Calls"] = str(target["max_inference_calls"])
    content = ""
    reasoning = ""
    usage = None
    model = target["model"]
    provider_model_observed = False
    finish = None
    ttft = None
    tool_calls = {}
    raw_usage = {}
    cache_read_usage: dict[str, int] | None = None
    cache_read_reported: bool | None = None
    cache_write_reported: bool | None = None
    cache_observation = observe_cache_usage(raw=None)
    cache_observation_issues: dict[str, int] = {}
    usage_event_index = 0
    observed_session_phase = request_phase(messages)
    observed_phase_source = "request"
    done = False
    response = None
    stop = threading.Event()
    guard_error = []
    # The finalizer fsyncs partial evidence before closing on every exit path.
    stream_file = None
    native_evidence = None
    wire_bytes = 0
    last_received_at = None
    next_checkpoint = 0
    if stream_path:
        stream_file = open(stream_path, "xb", buffering=8192)  # noqa: SIM115

    def partial():
        return {
            "final": final_content(content),
            "reasoning": reasoning,
            "model": model,
            "usage": usage,
            "latency_s": time.monotonic() - started,
            "ttft_s": ttft,
            "finish_reason": finish,
            "raw_usage": raw_usage,
            "response_status": response.status_code if response is not None else None,
            "provider_cache_observation": {
                **cache_observation,
                "event_issues": cache_observation_issues,
            },
            "cache_read_usage": cache_read_usage,
            "tool_calls": list(tool_calls.values()),
            "phase": observed_session_phase,
            "phase_source": observed_phase_source,
            "cache_read_reported": cache_read_reported,
            "cache_write_reported": cache_write_reported,
            "cost_usd": (
                cost_for(usage, model, target.get("prices", {}))
                if target["kind"] == "single"
                else None
            ),
            **(
                {
                    "native_output": {
                        **native_evidence,
                        "provider_model_observed": provider_model_observed,
                    }
                }
                if native_evidence is not None
                else {}
            ),
        }

    try:
        if output_policy == "native" and target["kind"] == "single":
            body, native_evidence = native_output.resolve_single(
                target, body, headers, limits, cancelled, started
            )
        if cancelled() or time.monotonic() - started > limits["total_timeout_s"]:
            raise CallFailure("cancelled or total request deadline exceeded")
        if activity is not None:
            activity.waiting()
        response = requests.post(
            endpoint,
            json=body,
            headers=headers,
            stream=True,
            timeout=(
                min(10, limits["total_timeout_s"]),
                min(limits["idle_timeout_s"], limits["total_timeout_s"]),
            ),
        )
        if response.status_code >= HTTPStatus.BAD_REQUEST:
            raise CallFailure(f"Target HTTP {response.status_code}")
        if "text/event-stream" not in response.headers.get("content-type", ""):
            raise CallFailure("Target did not return a streaming response")
        observed_session_phase, observed_phase_source = session_phase(
            messages, response.headers
        )
        if output_policy == "native" and target["kind"] == "mom":
            native_evidence = native_output.from_headers(target, response.headers)

        def watch():
            while not stop.wait(0.1):
                why = (
                    "cancelled"
                    if cancelled()
                    else (
                        "total request deadline exceeded"
                        if time.monotonic() - started > limits["total_timeout_s"]
                        else None
                    )
                )
                if why:
                    guard_error.append(why)
                    with suppress(AttributeError, OSError):
                        response.raw._fp.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                    return

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        buffer = bytearray()
        event_lines = []

        def consume(lines):
            nonlocal content, reasoning, usage, model, finish, ttft, raw_usage, done
            nonlocal provider_model_observed
            nonlocal cache_read_usage, cache_read_reported, cache_write_reported
            nonlocal cache_observation, usage_event_index
            data = "\n".join(x[5:].lstrip() for x in lines if x.startswith("data:"))
            if not data:
                return
            if data == "[DONE]":
                done = True
                return
            obj = json.loads(data)
            if obj.get("error"):
                raise CallFailure("Target returned an error event")
            if obj.get("model") is not None:
                if native_evidence is not None:
                    native_output.validate_response(native_evidence, obj["model"], None)
                if isinstance(obj["model"], str) and obj["model"]:
                    provider_model_observed = True
            model = obj.get("model") or model
            if "usage" in obj:
                observed = observe_cache_usage(raw=obj["usage"])
                qualification = observed["qualification"]
                if qualification in {"malformed", "conflicting", "unqualified"}:
                    cache_observation_issues[qualification] = (
                        cache_observation_issues.get(qualification, 0) + 1
                    )
                if (
                    observed["field_present"]
                    or observed["qualification"] in {"malformed", "conflicting"}
                    or not cache_observation["field_present"]
                ):
                    cache_observation = {
                        **observed,
                        "usage_event_index": usage_event_index,
                    }
                usage_event_index += 1
            if obj.get("usage"):
                raw_usage = obj["usage"]
                usage = normalize_usage(raw_usage)
                chunk_cache_read_reported, chunk_cache_write_reported = usage_presence(
                    raw_usage
                )
                cache_read_reported = (
                    cache_read_reported is True or chunk_cache_read_reported
                )
                cache_write_reported = (
                    cache_write_reported is True or chunk_cache_write_reported
                )
                if chunk_cache_read_reported:
                    cache_read_usage = usage
            for choice in obj.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta") or choice.get("message") or {}
                piece = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if not isinstance(piece, str) or not isinstance(thought, str):
                    raise CallFailure("Unsupported response content shape")
                if (piece or thought) and ttft is None:
                    ttft = time.monotonic() - started
                content += piece
                reasoning += thought
                for call in delta.get("tool_calls", []):
                    idx = call.get("index", 0)
                    entry = tool_calls.setdefault(
                        idx,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if call.get("id"):
                        entry["id"] = call["id"]
                    fn = call.get("function") or {}
                    if fn.get("name"):
                        entry["function"]["name"] += fn["name"]
                    entry["function"]["arguments"] += fn.get("arguments", "")
                finish = choice.get("finish_reason") or finish
            text = content + reasoning
            if len(text) > limits["max_output_chars"]:
                raise CallFailure("Output character cap exceeded")
            n = limits["repetition_window"]
            count = limits["repetition_limit"]
            tail = text[-n * count :]
            if len(tail) == n * count and tail == tail[-n:] * count:
                raise CallFailure("Repeated output guard triggered")
            if usage and usage["output_tokens"] > limits["max_output_tokens"]:
                raise CallFailure("Output token cap exceeded")
            if native_evidence is not None:
                native_output.validate_usage(native_evidence, usage)

        for chunk in response.iter_content(chunk_size=1):
            if guard_error:
                raise CallFailure(guard_error[0])
            if cancelled():
                raise CallFailure("cancelled")
            observed_at = time.monotonic()
            if observed_at - started > limits["total_timeout_s"]:
                raise CallFailure("total request deadline exceeded")
            if not chunk:
                continue
            wire_bytes += len(chunk)
            last_received_at = observed_at
            if wire_bytes > max(limits["max_output_chars"] * 30, 1048576):
                raise CallFailure("Stream byte cap exceeded")
            if stream_file is not None:
                stream_file.write(chunk)
            if activity is not None and observed_at >= next_checkpoint:
                activity.received(wire_bytes, observed_at)
                next_checkpoint = observed_at + CHECKPOINT_SECONDS
            buffer.extend(chunk)
            if len(buffer) > limits["max_output_chars"] * 2:
                raise CallFailure("SSE line cap exceeded")
            if chunk == b"\n":
                line = bytes(buffer).decode("utf-8").rstrip("\r\n")
                buffer.clear()
                if line:
                    event_lines.append(line)
                else:
                    consume(event_lines)
                    event_lines = []
                    if stream_file is not None:
                        stream_file.flush()
                    if done:
                        break
        if event_lines:
            consume(event_lines)
        if guard_error:
            raise CallFailure(guard_error[0])
        if not done or finish not in {"stop", "tool_calls", "function_call", "length"}:
            raise CallFailure(f"Incomplete final response (finish_reason={finish})")
        result = partial()
        if output_policy == "native":
            if not provider_model_observed:
                raise CallFailure(
                    "Native response model acknowledgement missing", result
                )
            native_output.validate_response(native_evidence, model, usage)
        result["output_complete"] = finish != "length"
        if (
            target.get("expected_response_model")
            and model != target["expected_response_model"]
        ):
            raise CallFailure(
                "Response model identity differs from frozen target", result
            )
        result["cost_usd"] = (
            cost_for(usage, model, target.get("prices", {}))
            if target["kind"] == "single"
            else None
        )
        result["response_usage"] = result["usage"]
        if target["kind"] == "mom":
            result.update(
                mom_usage(
                    response.headers,
                    usage,
                    target.get("prices", {}),
                    cache_read_reported=cache_read_reported,
                    cache_write_reported=cache_write_reported,
                )
            )
        else:
            result["inference_call_count"] = 1
        result["cost_complete"] = result["cost_usd"] is not None
        result["config_hash"] = response.headers.get(
            "x-sr-bench-config-hash"
        ) or response.headers.get("x-vsr-config-hash")
        result["provider_model_observed"] = provider_model_observed
        result["selected_model"] = response.headers.get("x-vsr-selected-model") or (
            model if provider_model_observed else None
        )
        result["decision"] = response.headers.get("x-vsr-selected-decision")
        if target.get("config_hash") and result["config_hash"] != target["config_hash"]:
            raise CallFailure(
                "Runtime configuration identity acknowledgement missing or mismatched",
                result,
            )
        return result
    except native_output.NativeOutputError as exc:
        raise CallFailure(str(exc), partial()) from exc
    except CallFailure as exc:
        exc.partial = {**partial(), **exc.partial}
        raise
    except (requests.RequestException, ValueError, UnicodeError) as exc:
        raise CallFailure(
            guard_error[0] if guard_error else type(exc).__name__, partial()
        ) from exc
    finally:
        stop.set()
        if response is not None:
            response.close()
        if stream_file is not None:
            try:
                stream_file.flush()
                os.fsync(stream_file.fileno())
            finally:
                stream_file.close()
        if activity is not None and last_received_at is not None:
            activity.received(wire_bytes, last_received_at, force=True)
