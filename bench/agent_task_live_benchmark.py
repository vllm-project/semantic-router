#!/usr/bin/env python3
"""Live agent-task benchmark with deterministic completion scoring."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cli.sr_bench.agent_task_kernel import (  # noqa: F401 - retained source API
    SYSTEM_INSTRUCTIONS,
    TaskSpec,
    TaskTurn,
    build_messages,
    long_horizon_task_specs,
    score_answer,
    scoring_instruction,
    smoke_task_specs,
    task_specs,
    tool_messages,
)

HTTP_OK = 200
HTTP_REDIRECT_START = 300
VSR_HEADERS = (
    "x-vsr-selected-model",
    "x-vsr-selected-decision",
    "x-vsr-session-phase",
    "x-vsr-selected-confidence",
    "x-vsr-replay-id",
    "x-vsr-context-token-count",
    "x-vsr-matched-conversation",
    "x-vsr-routing-latency-ms",
    "x-vsr-cost",
    "x-vsr-cost-currency",
)
NON_NEGATIVE_NUMBER_HEADERS = ("x-vsr-routing-latency-ms", "x-vsr-cost")
COST_METRIC = "llm_model_cost_total"
ROUTING_LATENCY_METRIC = "llm_model_routing_latency_seconds"
COST_BASIS = "configured pricing (x-vsr-cost), not a provider bill"
METRIC_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+(\S+)")
METRIC_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
MetricSamples = dict[tuple[str, tuple[tuple[str, str], ...]], float]
CORE_ROUTER_HEADERS = (
    "x-vsr-selected-model",
    "x-vsr-selected-decision",
    "x-vsr-replay-id",
)
DIAGNOSTIC_ROUTER_HEADERS = (
    *CORE_ROUTER_HEADERS,
    "x-vsr-session-phase",
    "x-vsr-selected-confidence",
    "x-vsr-context-token-count",
)
PROTOCOLS = ("chat", "responses")
CONTINUATION_MODES = ("full-history", "previous-response-id")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--session-header", default="x-session-id")
    parser.add_argument("--session-prefix", default="agent-task")
    parser.add_argument(
        "--suite", choices=("smoke", "long-horizon", "all"), default="smoke"
    )
    parser.add_argument("--task-repetitions", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--label", default="live-router")
    parser.add_argument("--evidence-ref", default="")
    parser.add_argument("--evidence-image-tag", default="")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="chat")
    parser.add_argument(
        "--continuation-mode", choices=CONTINUATION_MODES, default="full-history"
    )
    parser.add_argument("--baseline-base-url", default="")
    parser.add_argument("--baseline-model", default="")
    parser.add_argument("--baseline-label", default="direct-backend")
    parser.add_argument("--baseline-protocol", choices=PROTOCOLS, default="")
    parser.add_argument(
        "--baseline-continuation-mode", choices=CONTINUATION_MODES, default=""
    )
    parser.add_argument("--extra-header", action="append", default=[])
    parser.add_argument(
        "--require-router-header",
        action="append",
        choices=VSR_HEADERS,
        default=[],
        help=(
            "Require this x-vsr response header on every successful router request. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--require-router-diagnostics",
        action="store_true",
        help=(
            "Require selected model, decision, replay id, selected confidence, "
            "and context-token-count headers on every successful router request."
        ),
    )
    parser.add_argument("--min-success-rate", type=float, default=0.0)
    parser.add_argument("--min-task-success-rate", type=float, default=0.0)
    parser.add_argument("--min-task-score", type=float, default=0.0)
    parser.add_argument("--max-tool-loop-violations", type=int, default=-1)
    parser.add_argument("--max-context-portability-violations", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--router-metrics-url",
        default="",
        help=(
            "Router Prometheus endpoint, e.g. http://127.0.0.1:9190/metrics. When set, "
            "the summary also reports the cost and routing time the router recorded."
        ),
    )
    args = parser.parse_args(argv)
    validate_protocol_args(parser, args)
    return args


def validate_protocol_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.protocol == "chat" and args.continuation_mode != "full-history":
        parser.error(
            "--continuation-mode previous-response-id requires --protocol responses"
        )
    baseline_protocol = args.baseline_protocol or args.protocol
    baseline_continuation = args.baseline_continuation_mode or args.continuation_mode
    if baseline_protocol == "chat" and baseline_continuation != "full-history":
        parser.error(
            "--baseline-continuation-mode previous-response-id requires "
            "--baseline-protocol responses"
        )


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(".agent-harness/experiments/live-agent-tasks") / stamp


def parse_extra_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        if "=" in value:
            name, raw = value.split("=", 1)
        elif ":" in value:
            name, raw = value.split(":", 1)
        else:
            raise ValueError(f"extra header must be NAME=VALUE, got {value!r}")
        headers[name.strip()] = raw.strip()
    return headers


def request_headers(args: argparse.Namespace, session_id: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "vllm-sr-agent-task-live-benchmark",
        args.session_header: session_id,
    }
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    headers.update(parse_extra_headers(args.extra_header))
    return headers


def responses_message(role: str, content: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": content}


def responses_tool_items(
    task_name: str, turn_index: int, turn: TaskTurn
) -> list[dict[str, Any]]:
    call_id = f"call_{task_name}_{turn_index}"
    return [
        {
            "type": "function_call",
            "id": f"item_{task_name}_{turn_index}",
            "call_id": call_id,
            "name": turn.tool_name or "read_tool_result",
            "arguments": json.dumps({"task": task_name}),
        },
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": turn.tool_result,
        },
    ]


def responses_turn_items(
    task: TaskSpec, turn_index: int, *, include_history: bool
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if include_history:
        for prior_index, prior in enumerate(task.turns[:turn_index]):
            items.append(responses_message("user", prior.prompt))
            if prior.expected_terms:
                items.append(responses_message("assistant", "Acknowledged."))
            elif prior.tool_result:
                items.extend(responses_tool_items(task.name, prior_index, prior))
            else:
                items.append(
                    responses_message("assistant", "I will inspect the evidence.")
                )

    turn = task.turns[turn_index]
    if turn.tool_result:
        items.extend(responses_tool_items(task.name, turn_index, turn))
    prompt = turn.prompt
    if turn.expected_terms:
        prompt = prompt + "\n\n" + scoring_instruction(turn.expected_terms)
    items.append(responses_message("user", prompt))
    return items


def build_body(
    args: argparse.Namespace,
    task: TaskSpec,
    turn_index: int,
    previous_response_id: str,
) -> dict[str, Any]:
    if args.protocol == "chat":
        return {
            "model": args.model,
            "messages": build_messages(task, turn_index),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }

    use_lineage = args.continuation_mode == "previous-response-id"
    body = {
        "model": args.model,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": responses_turn_items(
            task, turn_index, include_history=not use_lineage
        ),
        "temperature": args.temperature,
        "max_output_tokens": args.max_tokens,
    }
    if use_lineage and previous_response_id:
        body["previous_response_id"] = previous_response_id
    return body


def send_request(
    args: argparse.Namespace,
    task: TaskSpec,
    turn_index: int,
    session_id: str,
    previous_response_id: str,
) -> dict[str, Any]:
    if args.dry_run:
        return dry_response(args.protocol, task, turn_index)
    url = request_url(args.base_url, args.protocol)
    body = json.dumps(build_body(args, task, turn_index, previous_response_id)).encode(
        "utf-8"
    )
    started = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers(args, session_id),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
            return response_record(
                response.status, dict(response.headers), payload, started
            )
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        return response_record(exc.code, dict(exc.headers), payload, started)
    except Exception as exc:  # pragma: no cover - network errors vary by platform
        return {
            "status": 0,
            "headers": {},
            "json": {},
            "payload": "",
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": str(exc),
        }


def request_url(base_url: str, protocol: str) -> str:
    endpoint = "chat/completions" if protocol == "chat" else "responses"
    return base_url.rstrip("/") + "/" + endpoint


def response_record(
    status: int, headers: dict[str, str], payload: str, started: float
) -> dict[str, Any]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {"raw_body": payload[:500]}
    return {
        "status": status,
        "headers": {key.lower(): value for key, value in headers.items()},
        "json": parsed if isinstance(parsed, dict) else {},
        "payload": payload,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "error": (
            error_message(parsed, payload) if status >= HTTP_REDIRECT_START else ""
        ),
    }


def dry_response(protocol: str, task: TaskSpec, turn_index: int) -> dict[str, Any]:
    turn = task.turns[turn_index]
    content = " ".join(turn.expected_terms) if turn.expected_terms else "Acknowledged."
    if protocol == "responses":
        response_json = {
            "id": f"dry_{task.name}_{turn_index}",
            "object": "response",
            "model": "dry-model",
            "status": "completed",
            "output": [
                {
                    "id": f"msg_dry_{task.name}_{turn_index}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": content, "annotations": []}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 110,
            },
        }
    else:
        response_json = {
            "id": f"dry_{task.name}_{turn_index}",
            "model": "dry-model",
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        }
    return {
        "status": HTTP_OK,
        "headers": {
            "x-vsr-selected-model": "dry-model",
            "x-vsr-selected-decision": "dry-run",
            "x-vsr-session-phase": turn.phase,
            "x-vsr-selected-confidence": "0.0000",
            "x-vsr-replay-id": f"dry-replay-{task.name}-{turn_index}",
            "x-vsr-context-token-count": "42",
            "x-vsr-routing-latency-ms": "0.250",
            "x-vsr-cost": "0.00001",
            "x-vsr-cost-currency": "USD",
        },
        "json": response_json,
        "payload": "",
        "latency_ms": 1.0,
        "error": "",
    }


def error_message(parsed: Any, payload: str) -> str:
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        return str(error.get("message", ""))
    if isinstance(error, str):
        return error
    return payload[:300]


def run_tasks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for repetition in range(max(1, args.task_repetitions)):
        for task_index, task in enumerate(task_specs(args.suite)):
            session_id = session_id_for(args, task, task_index, repetition)
            previous_response_id = ""
            previous_selected_model = ""
            for turn_index, turn in enumerate(task.turns):
                result = send_request(
                    args, task, turn_index, session_id, previous_response_id
                )
                row = row_from_result(
                    args,
                    task,
                    task_index,
                    repetition,
                    turn_index,
                    turn,
                    session_id,
                    result,
                    previous_selected_model,
                    previous_response_id,
                )
                rows.append(row)
                previous_response_id = response_id(result)
                if row["selected_model"]:
                    previous_selected_model = row["selected_model"]
    elapsed = float(len(rows)) if args.dry_run else time.perf_counter() - started
    return rows, summarize(rows, elapsed, args.label)


def session_id_for(
    args: argparse.Namespace, task: TaskSpec, task_index: int, repetition: int
) -> str:
    if args.task_repetitions <= 1:
        return f"{args.session_prefix}-{task.name}-{task_index:02d}"
    return f"{args.session_prefix}-r{repetition:02d}-{task.name}-{task_index:02d}"


def row_from_result(
    args: argparse.Namespace,
    task: TaskSpec,
    task_index: int,
    task_repetition: int,
    turn_index: int,
    turn: TaskTurn,
    session_id: str,
    result: dict[str, Any],
    previous_selected_model: str,
    previous_response_id: str,
) -> dict[str, Any]:
    response_json = result.get("json") or {}
    selected_model = selected_model_from(result)
    status = int(result.get("status") or 0)
    previous_response_id_sent = bool(
        args.protocol == "responses"
        and args.continuation_mode == "previous-response-id"
        and previous_response_id
    )
    switched = bool(
        previous_selected_model
        and selected_model
        and selected_model != previous_selected_model
    )
    content = response_content(response_json)
    score, missing = score_answer(content, turn.expected_terms)
    return {
        "label": args.label,
        "task_index": task_index,
        "task_repetition": task_repetition,
        "task": task.name,
        "task_suite": task.suite,
        "task_instance": f"r{task_repetition:02d}:{task.name}",
        "session_id": session_id,
        "turn_identity": f"{session_id}:turn-{turn_index:03d}",
        "turn": turn_index,
        "phase": turn.phase,
        "protocol": args.protocol,
        "continuation_mode": args.continuation_mode,
        "status": status,
        "success": HTTP_OK <= status < HTTP_REDIRECT_START,
        "latency_ms": round(float(result.get("latency_ms") or 0), 3),
        "selected_model": selected_model,
        "response_model": response_json.get("model", ""),
        "model_switched": switched,
        "tool_loop_switch_violation": turn.phase == "tool_loop" and switched,
        "context_portability_violation": previous_response_id_sent and switched,
        "previous_response_id_sent": previous_response_id_sent,
        "previous_response_id": (
            previous_response_id if previous_response_id_sent else ""
        ),
        "response_id": response_json.get("id", ""),
        "prompt_tokens": usage_value(response_json, "prompt_tokens", "input_tokens"),
        "completion_tokens": usage_value(
            response_json, "completion_tokens", "output_tokens"
        ),
        "cached_tokens": cached_tokens(response_json),
        "reasoning_tokens": reasoning_tokens(response_json),
        "finish_reason": finish_reason(response_json),
        "routing_latency_ms": header_float(result, "x-vsr-routing-latency-ms"),
        "cost": header_float(result, "x-vsr-cost"),
        "cost_currency": (result.get("headers") or {}).get("x-vsr-cost-currency", ""),
        "answer_score": score,
        "missing_terms": ",".join(missing),
        "answer_excerpt": answer_excerpt(content),
        "scored_turn": bool(turn.expected_terms),
        "error": result.get("error", ""),
        **{
            header: (result.get("headers") or {}).get(header, "")
            for header in VSR_HEADERS
        },
    }


def selected_model_from(result: dict[str, Any]) -> str:
    headers = result.get("headers") or {}
    response_json = result.get("json") or {}
    return headers.get("x-vsr-selected-model") or response_json.get("model", "")


def response_content(response_json: dict[str, Any]) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    output = response_json.get("output")
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            parts = item.get("content")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "output_text"
                    and isinstance(part.get("text"), str)
                ):
                    texts.append(part["text"])
        return "\n".join(texts)
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def answer_excerpt(content: str, limit: int = 600) -> str:
    return content.replace("\n", "\\n")[:limit]


def response_id(result: dict[str, Any]) -> str:
    response_json = result.get("json") or {}
    value = response_json.get("id")
    return value if isinstance(value, str) else ""


def usage_value(response_json: dict[str, Any], *keys: str) -> int:
    usage = response_json.get("usage") or {}
    if not isinstance(usage, dict):
        return 0
    for key in keys:
        value = usage.get(key)
        if value is not None:
            return int(value or 0)
    return 0


def cached_tokens(response_json: dict[str, Any]) -> int:
    usage = response_json.get("usage") if isinstance(response_json, dict) else {}
    details = {}
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details") or usage.get(
            "input_tokens_details", {}
        )
    if not isinstance(details, dict):
        return 0
    return int(details.get("cached_tokens") or 0)


def reasoning_tokens(response_json: dict[str, Any]) -> int:
    usage = response_json.get("usage") if isinstance(response_json, dict) else {}
    details = {}
    if isinstance(usage, dict):
        details = usage.get("completion_tokens_details") or usage.get(
            "output_tokens_details", {}
        )
    if not isinstance(details, dict):
        return 0
    return int(details.get("reasoning_tokens") or 0)


def finish_reason(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return str(choices[0].get("finish_reason") or "")
    status = response_json.get("status")
    if status == "incomplete":
        details = response_json.get("incomplete_details") or {}
        if isinstance(details, dict) and details.get("reason"):
            return str(details["reason"])
    return str(status or "")


def header_float(result: dict[str, Any], header: str) -> float | None:
    value = (result.get("headers") or {}).get(header, "")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def summarize(
    rows: list[dict[str, Any]], elapsed_seconds: float, label: str
) -> dict[str, Any]:
    scored = [row for row in rows if row["scored_turn"]]
    task_names = sorted({str(row["task"]) for row in rows})
    scored_task_names = sorted({str(row["task"]) for row in scored})
    scores = [
        float(row["answer_score"]) for row in scored if row["answer_score"] is not None
    ]
    task_successes = sum(1 for row in scored if row["answer_score"] == 1.0)
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    cached = sum(int(row["cached_tokens"]) for row in rows)
    return {
        "label": label,
        "requests": len(rows),
        "tasks": len({row["task"] for row in rows}),
        "task_count": len({row["task"] for row in rows}),
        "task_names": task_names,
        "task_instances": len({task_key(row) for row in scored}),
        "scored_task_count": len(scored_task_names),
        "scored_task_names": scored_task_names,
        "task_suites": counts(row.get("task_suite", "") for row in rows),
        "protocols": counts(row["protocol"] for row in rows if row.get("protocol")),
        "continuation_modes": counts(
            row["continuation_mode"] for row in rows if row.get("continuation_mode")
        ),
        "successes": sum(1 for row in rows if row["success"]),
        "success_rate": (
            round(sum(1 for row in rows if row["success"]) / len(rows), 4)
            if rows
            else 0.0
        ),
        "mean_final_score": round(statistics.fmean(scores), 4) if scores else None,
        "task_score_mean": round(statistics.fmean(scores), 4) if scores else None,
        "task_successes": task_successes,
        "task_success_rate": round(task_successes / len(scored), 4) if scored else None,
        "task_exact_successes": task_successes,
        "task_exact_success_rate": (
            round(task_successes / len(scored), 4) if scored else None
        ),
        "final_scores": {task_key(row): row["answer_score"] for row in scored},
        "status_counts": counts(str(row["status"]) for row in rows),
        "latency_ms": latency_summary(
            [float(row["latency_ms"]) for row in rows if row["success"]]
        ),
        "model_switches": sum(1 for row in rows if row["model_switched"]),
        "tool_loop_switch_violations": sum(
            1 for row in rows if row["tool_loop_switch_violation"]
        ),
        "context_portability_violations": sum(
            1 for row in rows if row["context_portability_violation"]
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": sum(int(row["completion_tokens"]) for row in rows),
        "cached_tokens": cached,
        "cached_prompt_ratio": (
            round(cached / prompt_tokens, 4) if prompt_tokens else None
        ),
        "reasoning_tokens": sum(int(row.get("reasoning_tokens") or 0) for row in rows),
        "finish_reason_counts": counts(
            row["finish_reason"] for row in rows if row.get("finish_reason")
        ),
        "truncated_requests": sum(
            1
            for row in rows
            if row.get("finish_reason") in ("length", "max_output_tokens")
        ),
        "routing_latency_ms": latency_summary(
            [
                float(row["routing_latency_ms"])
                for row in rows
                if row.get("routing_latency_ms") is not None
            ]
        ),
        "cost": cost_summary(rows),
        "phase_counts": counts(row["phase"] for row in rows),
        "selected_model_counts": counts(
            row["selected_model"] for row in rows if row["selected_model"]
        ),
        "error_counts": counts(
            row.get("error", "") for row in rows if row.get("error")
        ),
        "missing_router_header_counts": missing_router_header_counts(rows),
        "invalid_router_header_counts": invalid_router_header_counts(rows),
        "failed_tasks": sorted(
            {
                task_key(row): row["missing_terms"]
                for row in scored
                if row["answer_score"] != 1.0
            }.items()
        ),
        "requests_per_second": (
            round(len(rows) / elapsed_seconds, 3) if elapsed_seconds > 0 else None
        ),
    }


def task_key(row: dict[str, Any]) -> str:
    value = row.get("task_instance") or row.get("task")
    return str(value)


def missing_router_header_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        header: sum(1 for row in rows if row["success"] and not row.get(header))
        for header in VSR_HEADERS
    }


def invalid_router_header_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        header: sum(
            1
            for row in rows
            if row["success"]
            and row.get(header)
            and not valid_router_header_value(header, str(row[header]))
        )
        for header in VSR_HEADERS
    }


def valid_router_header_value(header: str, value: str) -> bool:
    if header == "x-vsr-selected-confidence":
        try:
            parsed = float(value)
        except ValueError:
            return False
        return 0.0 <= parsed <= 1.0
    if header == "x-vsr-context-token-count":
        try:
            parsed = int(value)
        except ValueError:
            return False
        return parsed >= 0
    if header in NON_NEGATIVE_NUMBER_HEADERS:
        try:
            parsed = float(value)
        except ValueError:
            return False
        return parsed >= 0
    if header == "x-vsr-session-phase":
        return value.strip() in {
            "user_turn",
            "tool_loop",
            "provider_state",
            "idle_boundary",
            "topic_drift",
            "frontier_turn",
            "final",
        }
    return bool(value.strip())


def required_router_headers(args: argparse.Namespace) -> list[str]:
    required = list(args.require_router_header)
    if args.require_router_diagnostics:
        required.extend(DIAGNOSTIC_ROUTER_HEADERS)
    return list(dict.fromkeys(required))


def validate_summary(args: argparse.Namespace, summary: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if args.min_success_rate and summary["success_rate"] < args.min_success_rate:
        failures.append(
            f"success_rate {summary['success_rate']} < {args.min_success_rate}"
        )
    task_success_rate = summary.get("task_success_rate")
    if args.min_task_success_rate and (
        task_success_rate is None or task_success_rate < args.min_task_success_rate
    ):
        failures.append(
            f"task_success_rate {task_success_rate} < {args.min_task_success_rate}"
        )
    task_score = summary.get("task_score_mean")
    if args.min_task_score and (task_score is None or task_score < args.min_task_score):
        failures.append(f"task_score_mean {task_score} < {args.min_task_score}")
    if (
        args.max_tool_loop_violations >= 0
        and summary["tool_loop_switch_violations"] > args.max_tool_loop_violations
    ):
        failures.append(
            "tool_loop_switch_violations "
            f"{summary['tool_loop_switch_violations']} > {args.max_tool_loop_violations}"
        )
    if (
        args.max_context_portability_violations >= 0
        and summary["context_portability_violations"]
        > args.max_context_portability_violations
    ):
        failures.append(
            "context_portability_violations "
            f"{summary['context_portability_violations']} > "
            f"{args.max_context_portability_violations}"
        )
    missing_headers = summary.get("missing_router_header_counts", {})
    invalid_headers = summary.get("invalid_router_header_counts", {})
    for header in required_router_headers(args):
        missing = int(missing_headers.get(header, 0))
        if missing:
            failures.append(
                f"missing_router_header {header}: {missing} successful requests"
            )
        invalid = int(invalid_headers.get(header, 0))
        if invalid:
            failures.append(
                f"invalid_router_header {header}: {invalid} successful requests"
            )
    return failures


def counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "mean": round(statistics.fmean(ordered), 3),
        "p50": round(percentile(ordered, 50), 3),
        "p95": round(percentile(ordered, 95), 3),
        "max": round(max(ordered), 3),
    }


def percentile(ordered: list[float], pct: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def cost_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("cost") is None:
            continue
        currency = row.get("cost_currency") or "unknown"
        totals[currency] = totals.get(currency, 0.0) + float(row["cost"])
        counts[currency] = counts.get(currency, 0) + 1
    priced = sum(counts.values())
    return {
        "basis": COST_BASIS,
        "total": {key: round(value, 8) for key, value in sorted(totals.items())},
        "mean_per_priced_request": {
            key: round(value / counts[key], 8) for key, value in sorted(totals.items())
        },
        "priced_requests_by_currency": dict(sorted(counts.items())),
        "priced_requests": priced,
        "unpriced_requests": len(rows) - priced,
    }


def parse_metrics_text(text: str) -> MetricSamples:
    samples: MetricSamples = {}
    for line in text.splitlines():
        match = METRIC_SAMPLE.match(line)
        if not match or not match.group(1).startswith(
            (COST_METRIC, ROUTING_LATENCY_METRIC)
        ):
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        labels = tuple(sorted(METRIC_LABEL.findall(match.group(2) or "")))
        samples[(match.group(1), labels)] = value
    return samples


def read_router_metrics(args: argparse.Namespace) -> MetricSamples | None:
    """Scrape router /metrics; it's optional, so a failure only warns."""
    if not args.router_metrics_url or args.dry_run:
        return None
    try:
        with urllib.request.urlopen(
            args.router_metrics_url, timeout=args.timeout
        ) as response:
            return parse_metrics_text(response.read().decode("utf-8", errors="replace"))
    except Exception as exc:  # pragma: no cover - network errors vary by platform
        print(f"warning: could not read router metrics: {exc}", file=sys.stderr)
        return None


def metrics_delta(before: MetricSamples, after: MetricSamples) -> dict[str, Any]:
    """What the router recorded between two scrapes, including any other traffic."""
    cost_by_model: dict[str, dict[str, float]] = {}
    for (name, labels), value in after.items():
        if name != COST_METRIC:
            continue
        spent = value - before.get((name, labels), 0.0)
        if spent <= 0:
            continue
        label_map = dict(labels)
        model_costs = cost_by_model.setdefault(label_map.get("model", ""), {})
        model_costs[label_map.get("currency", "")] = round(spent, 8)

    def diff(suffix: str) -> float:
        key = (ROUTING_LATENCY_METRIC + suffix, ())
        return after.get(key, 0.0) - before.get(key, 0.0)

    decisions = diff("_count")
    return {
        "cost_by_model": dict(sorted(cost_by_model.items())),
        "routing_decisions": int(decisions),
        "routing_latency_ms_mean": (
            round(diff("_sum") / decisions * 1000, 3) if decisions > 0 else None
        ),
    }


def write_outputs(
    rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "turns.csv")
    with (output_dir / "turns.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "summary.md").write_text(render_markdown(summary))


def attach_evidence_identity(
    summary: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    if args.evidence_ref:
        summary["evidence_ref"] = args.evidence_ref
    if args.evidence_image_tag:
        summary["evidence_image_tag"] = args.evidence_image_tag
    return summary


def attach_validation_failures(
    summary: dict[str, Any], failures: list[str]
) -> dict[str, Any]:
    summary["validation_failures"] = failures
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(summary: dict[str, Any]) -> str:
    latency = summary["latency_ms"]
    routing = summary.get("routing_latency_ms") or {}
    return "\n".join(
        [
            "# Live Agent Task Benchmark",
            "",
            f"- label: {summary['label']}",
            f"- protocols: {summary.get('protocols', {})}",
            f"- continuation modes: {summary.get('continuation_modes', {})}",
            f"- requests: {summary['requests']}",
            f"- tasks: {summary['task_count']}",
            f"- request success rate: {summary['success_rate']}",
            f"- task success rate: {summary['task_success_rate']}",
            f"- mean final score: {summary['mean_final_score']}",
            f"- latency mean/p95/max ms: {latency['mean']} / {latency['p95']} / {latency['max']}",
            f"- model switches: {summary['model_switches']}",
            f"- tool-loop switch violations: {summary['tool_loop_switch_violations']}",
            f"- context portability violations: {summary['context_portability_violations']}",
            f"- missing router headers: {summary['missing_router_header_counts']}",
            f"- invalid router headers: {summary['invalid_router_header_counts']}",
            f"- cached prompt ratio: {summary['cached_prompt_ratio']}",
            f"- selected models: {summary.get('selected_model_counts', {})}",
            f"- routing latency p50/p95 ms: {routing.get('p50')} / {routing.get('p95')}",
            f"- cost ({COST_BASIS}): {summary.get('cost', {}).get('total')}",
            "- truncated requests (length or max_output_tokens): "
            f"{summary.get('truncated_requests')}",
            f"- router metrics: {summary.get('router_metrics')}",
            f"- validation failures: {summary.get('validation_failures', [])}",
            "",
        ]
    )


def compare_summaries(
    router_summary: dict[str, Any], baseline_summary: dict[str, Any]
) -> dict[str, Any]:
    return {
        "router_label": router_summary["label"],
        "baseline_label": baseline_summary["label"],
        "router_protocols": router_summary.get("protocols", {}),
        "baseline_protocols": baseline_summary.get("protocols", {}),
        "router_continuation_modes": router_summary.get("continuation_modes", {}),
        "baseline_continuation_modes": baseline_summary.get("continuation_modes", {}),
        "requests": {
            "router": router_summary["requests"],
            "baseline": baseline_summary["requests"],
        },
        "success_rate_delta": rounded_delta(
            router_summary["success_rate"], baseline_summary["success_rate"]
        ),
        "task_success_rate_delta": rounded_delta(
            router_summary.get("task_success_rate"),
            baseline_summary.get("task_success_rate"),
        ),
        "mean_final_score_delta": rounded_delta(
            router_summary.get("mean_final_score"),
            baseline_summary.get("mean_final_score"),
        ),
        "router_overhead_ms": {
            key: rounded_delta(
                router_summary["latency_ms"].get(key),
                baseline_summary["latency_ms"].get(key),
            )
            for key in ("mean", "p50", "p95", "max")
        },
        "router_vs_baseline_ratio": {
            "requests_per_second": rounded_ratio(
                router_summary["requests_per_second"],
                baseline_summary["requests_per_second"],
            ),
            "prompt_tokens": rounded_ratio(
                router_summary["prompt_tokens"], baseline_summary["prompt_tokens"]
            ),
            "completion_tokens": rounded_ratio(
                router_summary["completion_tokens"],
                baseline_summary["completion_tokens"],
            ),
        },
        "router_violations": {
            "tool_loop": router_summary["tool_loop_switch_violations"],
            "context_portability": router_summary["context_portability_violations"],
        },
        "baseline_status_counts": baseline_summary["status_counts"],
        "router_status_counts": router_summary["status_counts"],
        "router_final_scores": router_summary["final_scores"],
        "baseline_final_scores": baseline_summary["final_scores"],
    }


def rounded_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 3)


def rounded_ratio(left: Any, right: Any) -> float | None:
    if left is None or right in (None, 0):
        return None
    return round(float(left) / float(right), 4)


def write_comparison(comparison: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "comparison.md").write_text(render_comparison_markdown(comparison))


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    overhead = comparison["router_overhead_ms"]
    ratios = comparison["router_vs_baseline_ratio"]
    return "\n".join(
        [
            "# Router vs Direct Backend Agent Task Comparison",
            "",
            f"- router label: {comparison['router_label']}",
            f"- baseline label: {comparison['baseline_label']}",
            f"- router protocols: {comparison.get('router_protocols', {})}",
            f"- baseline protocols: {comparison.get('baseline_protocols', {})}",
            "- router continuation modes: "
            f"{comparison.get('router_continuation_modes', {})}",
            "- baseline continuation modes: "
            f"{comparison.get('baseline_continuation_modes', {})}",
            f"- success rate delta: {comparison['success_rate_delta']}",
            f"- task success rate delta: {comparison['task_success_rate_delta']}",
            f"- mean final score delta: {comparison['mean_final_score_delta']}",
            f"- mean overhead ms: {overhead['mean']}",
            f"- p95 overhead ms: {overhead['p95']}",
            f"- throughput ratio: {ratios['requests_per_second']}",
            f"- router violations: {comparison['router_violations']}",
            "",
        ]
    )


def namespace_with(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    metrics_before = read_router_metrics(args)
    rows, summary = run_tasks(args)
    metrics_after = read_router_metrics(args) if metrics_before is not None else None
    if metrics_before is not None and metrics_after is not None:
        summary["router_metrics"] = metrics_delta(metrics_before, metrics_after)
    attach_evidence_identity(summary, args)

    baseline_summary = None
    baseline_rows = None
    comparison = None
    if args.baseline_base_url:
        baseline_args = namespace_with(
            args,
            base_url=args.baseline_base_url,
            model=args.baseline_model or args.model,
            label=args.baseline_label,
            protocol=args.baseline_protocol or args.protocol,
            continuation_mode=(
                args.baseline_continuation_mode or args.continuation_mode
            ),
            evidence_ref="",
            evidence_image_tag="",
        )
        baseline_rows, baseline_summary = run_tasks(baseline_args)
        comparison = compare_summaries(summary, baseline_summary)

    validation_failures = validate_summary(args, summary)
    attach_validation_failures(summary, validation_failures)
    write_outputs(rows, summary, output_dir)
    if baseline_rows is not None and baseline_summary is not None:
        write_outputs(baseline_rows, baseline_summary, output_dir / "baseline")
        if comparison is not None:
            write_comparison(comparison, output_dir)
    if validation_failures:
        (output_dir / "validation_failures.json").write_text(
            json.dumps(validation_failures, indent=2) + "\n"
        )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary": summary,
                "baseline_summary": baseline_summary,
                "comparison": comparison,
                "validation_failures": validation_failures,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if validation_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
