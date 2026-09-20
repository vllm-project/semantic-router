"""Bounded OpenAI-compatible transport with final-channel-only extraction."""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
from contextlib import suppress
from http import HTTPStatus

import requests

from . import native_output
from .activity import CHECKPOINT_SECONDS

MAX_USAGE_RECEIPT_BYTES = 12288
MAX_RECEIPT_CALLS = 256


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
            details.get("cache_creation_tokens"),
            details.get("created_cache_tokens"),
            raw.get("cache_creation_input_tokens"),
        )
        if value is not None
    ]
    if any(value != writes[0] for value in writes):
        raise CallFailure("Conflicting cache-write token usage")
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


def mom_usage(response_headers, response_usage, prices):
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
            receipt.get("version") != 1
            or not isinstance(receipt.get("complete"), bool)
            or not isinstance(receipt.get("calls"), list)
        ):
            raise CallFailure("Invalid MoM usage receipt contract")
        calls = receipt["calls"]
        if not 1 <= len(calls) <= MAX_RECEIPT_CALLS:
            raise CallFailure("Invalid MoM inference call count")
        breakdown = []
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("model"), str):
                raise CallFailure("Invalid MoM child identity")
            raw_usage = call.get("usage") or {}
            usage = normalize_usage(
                {
                    **raw_usage,
                    "cache_read_input_tokens": raw_usage.get("cached_input_tokens", 0),
                    "cache_creation_input_tokens": raw_usage.get(
                        "cache_write_tokens", 0
                    ),
                }
            )
            breakdown.append(
                {
                    **call,
                    "usage": usage,
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
        }
    if response_headers.get("x-vsr-inference-call-count") == "1":
        selected = response_headers.get("x-vsr-selected-model")
        if selected:
            return {
                "usage": response_usage,
                "cost_usd": cost_for(response_usage, selected, prices),
                "inference_call_count": 1,
                "usage_complete": response_usage is not None,
            }
    return {
        "usage": None,
        "cost_usd": None,
        "inference_call_count": None,
        "usage_complete": False,
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
            "tool_calls": list(tool_calls.values()),
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
            data = "\n".join(x[5:].lstrip() for x in lines if x.startswith("data:"))
            if not data:
                return
            if data == "[DONE]":
                done = True
                return
            obj = json.loads(data)
            if obj.get("error"):
                raise CallFailure("Target returned an error event")
            if native_evidence is not None and obj.get("model") is not None:
                native_output.validate_response(native_evidence, obj["model"], None)
                provider_model_observed = True
            model = obj.get("model") or model
            if obj.get("usage"):
                raw_usage = obj["usage"]
                usage = normalize_usage(raw_usage)
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
            result.update(mom_usage(response.headers, usage, target.get("prices", {})))
        else:
            result["inference_call_count"] = 1
        result["cost_complete"] = result["cost_usd"] is not None
        result["config_hash"] = response.headers.get(
            "x-sr-bench-config-hash"
        ) or response.headers.get("x-vsr-config-hash")
        result["selected_model"] = response.headers.get("x-vsr-selected-model") or model
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
                os.fsync(stream_file.fileno())
            finally:
                stream_file.close()
        if activity is not None and last_received_at is not None:
            activity.received(wire_bytes, last_received_at, force=True)
