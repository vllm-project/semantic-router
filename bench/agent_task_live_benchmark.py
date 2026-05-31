#!/usr/bin/env python3
"""Live agent-task benchmark with deterministic completion scoring."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HTTP_OK = 200
HTTP_REDIRECT_START = 300
VSR_HEADERS = (
    "x-vsr-selected-model",
    "x-vsr-selected-decision",
    "x-vsr-selected-confidence",
    "x-vsr-replay-id",
    "x-vsr-context-token-count",
    "x-vsr-matched-conversation",
)
CORE_ROUTER_HEADERS = (
    "x-vsr-selected-model",
    "x-vsr-selected-decision",
    "x-vsr-replay-id",
)
DIAGNOSTIC_ROUTER_HEADERS = (
    *CORE_ROUTER_HEADERS,
    "x-vsr-selected-confidence",
    "x-vsr-context-token-count",
)


@dataclass(frozen=True)
class TaskTurn:
    phase: str
    prompt: str
    tool_name: str = ""
    tool_result: str = ""
    expected_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    name: str
    turns: tuple[TaskTurn, ...]
    suite: str = "smoke"


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
    parser.add_argument("--include-previous-response-id", action="store_true")
    parser.add_argument("--baseline-base-url", default="")
    parser.add_argument("--baseline-model", default="")
    parser.add_argument("--baseline-label", default="direct-backend")
    parser.add_argument("--baseline-include-previous-response-id", action="store_true")
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
    return parser.parse_args(argv)


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(".agent-harness/experiments/live-agent-tasks") / stamp


def task_specs(suite: str = "smoke") -> tuple[TaskSpec, ...]:
    specs = smoke_task_specs() + long_horizon_task_specs()
    if suite == "all":
        return specs
    return tuple(task for task in specs if task.suite == suite)


def smoke_task_specs() -> tuple[TaskSpec, ...]:
    return (
        TaskSpec(
            name="python-debug",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "We are debugging a Python dataclass test failure. "
                        "First say what evidence you need."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt=("Use the provided tool result to identify the root cause."),
                    tool_name="read_test_log",
                    tool_result=(
                        "pytest: ValueError: mutable default <class 'list'> for "
                        "field tags is not allowed; use default_factory"
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the final fix. Include exact tokens "
                        "ROOT_CAUSE=mutable-default and FIX=default-factory."
                    ),
                    expected_terms=(
                        "ROOT_CAUSE=mutable-default",
                        "FIX=default-factory",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="ops-triage",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "A router failure appeared after inserting a fault proxy. "
                        "Plan the triage briefly."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the network inspection result to decide the fix.",
                    tool_name="inspect_network",
                    tool_result=(
                        "Envoy and vLLM are attached to pr1989-ga-vllm-sr-network; "
                        "the proxy was started on host networking. Envoy reports "
                        "no healthy upstream for fault-proxy."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the final triage. Include exact tokens "
                        "ROOT_CAUSE=wrong-docker-network and FIX=same-network."
                    ),
                    expected_terms=(
                        "ROOT_CAUSE=wrong-docker-network",
                        "FIX=same-network",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="routing-policy",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "We need a session-aware routing decision. Explain what "
                        "state matters before switching models."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the provided tool result before deciding.",
                    tool_name="observe_agent_phase",
                    tool_result=(
                        "The agent is inside an active tool loop and has a stable "
                        "session id. A later turn may pass an idle timeout."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the policy. Include exact tokens "
                        "DO_NOT_SWITCH=tool-loop and MAY_SWITCH=idle-timeout."
                    ),
                    expected_terms=(
                        "DO_NOT_SWITCH=tool-loop",
                        "MAY_SWITCH=idle-timeout",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="cache-reporting",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "We are interpreting cache-token benchmark output. State "
                        "what must be checked."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the probe output to state the limitation.",
                    tool_name="read_cache_probe",
                    tool_result=(
                        "router: cached_token_reporting=missing; direct-backend: "
                        "cached_token_reporting=missing; both paths returned 8/8 HTTP 200."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the limitation. Include exact tokens "
                        "LIMITATION=missing-cached-token-field and "
                        "NEXT=backend-reporting."
                    ),
                    expected_terms=(
                        "LIMITATION=missing-cached-token-field",
                        "NEXT=backend-reporting",
                    ),
                ),
            ),
        ),
    )


def long_horizon_task_specs() -> tuple[TaskSpec, ...]:
    return (
        TaskSpec(
            name="multi-file-regression",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "A release-blocking regression appeared after a router config "
                        "change. Build a short investigation plan."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the failed smoke output to narrow the failure.",
                    tool_name="run_agent_smoke",
                    tool_result=(
                        "make agent-smoke-local: router health OK, dashboard OK, "
                        "Envoy chat returns HTTP 503 no healthy upstream."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the generated Envoy config diff before choosing a fix.",
                    tool_name="diff_envoy_config",
                    tool_result=(
                        "runtime config changed backend base_url to fault-proxy, but "
                        "the generated Envoy cluster still points at the old upstream."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue from the same response state and state whether the "
                        "router should switch models during this diagnosis."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "Now convert the diagnosis into a maintainer-facing release "
                        "note and validation command."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the release-blocking fix. Include exact tokens "
                        "ROOT_CAUSE=stale-envoy-cluster, FIX=regenerate-envoy, "
                        "VALIDATE=agent-smoke-local."
                    ),
                    expected_terms=(
                        "ROOT_CAUSE=stale-envoy-cluster",
                        "FIX=regenerate-envoy",
                        "VALIDATE=agent-smoke-local",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="cluster-boundary",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "We are reviewing how ExtProc should route a selected model. "
                        "Identify which layer owns cluster choice and which layer "
                        "owns endpoint choice."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the Envoy routing fact.",
                    tool_name="inspect_envoy_route_config",
                    tool_result=(
                        "ExtProc can mutate request headers and body before routing; "
                        "Envoy clusters own load balancing among endpoints."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue from the same provider-managed state and decide "
                        "whether the router should probe or fallback between endpoints."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the maintainer boundary rule.",
                    tool_name="check_extproc_boundary",
                    tool_result=(
                        "The router may emit selected-model, selected-decision, and "
                        "cluster-routing signals, but endpoint failover remains an "
                        "Envoy cluster/LB concern."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the routing boundary. Include exact tokens "
                        "ROUTER=cluster-signal, ENVOY=endpoint-lb, "
                        "NO=endpoint-fallback."
                    ),
                    expected_terms=(
                        "ROUTER=cluster-signal",
                        "ENVOY=endpoint-lb",
                        "NO=endpoint-fallback",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="session-switch-policy",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "Design a model-switch policy for a long coding-agent session. "
                        "Start by separating hard locks from soft costs."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the agent phase observation.",
                    tool_name="observe_phase",
                    tool_result=(
                        "The current turn is inside a tool-call/tool-result loop, has "
                        "a stable session id, and has previous_response_id state."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue from provider-managed state. State what the router "
                        "must not do."
                    ),
                ),
                TaskTurn(
                    phase="idle_boundary",
                    prompt=(
                        "Assume the same session later idles past the configured "
                        "timeout and starts a new subtask."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "The matched decision changed from coding-debug to research. "
                        "Decide whether reselection is allowed."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the switch policy. Include exact tokens "
                        "LOCK=tool-loop, LOCK=provider-state, RESET=idle-or-drift."
                    ),
                    expected_terms=(
                        "LOCK=tool-loop",
                        "LOCK=provider-state",
                        "RESET=idle-or-drift",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="cache-economics",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "We need to decide whether switching away from a long frontier "
                        "model session is worth it. Identify the cache accounting input."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the pricing table.",
                    tool_name="read_pricing",
                    tool_result=(
                        "frontier prompt_per_1m=15.0, cached_input_per_1m=1.5; "
                        "small prompt_per_1m=0.2, cached_input_per_1m=0.04."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the cache probe result.",
                    tool_name="read_cache_probe",
                    tool_result=(
                        "Current backend responses omit usage.prompt_tokens_details."
                        "cached_tokens, so observed cached-token ratio is unavailable."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt="Continue the cost analysis in the same session state.",
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the cache economics conclusion. Include exact tokens "
                        "COST=input-checkout-delta, FRONTIER=stricter-cache, "
                        "LIMITATION=missing-cached-token-field."
                    ),
                    expected_terms=(
                        "COST=input-checkout-delta",
                        "FRONTIER=stricter-cache",
                        "LIMITATION=missing-cached-token-field",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="release-triage",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "Act as maintainer for a release candidate. Decide what evidence "
                        "is still blocking GA."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the benchmark status.",
                    tool_name="read_benchmark_status",
                    tool_result=(
                        "Local matrix complete; AMD overlay non-disruption complete; "
                        "repeat-failure recovery complete; full branch image blocked by "
                        "host disk pressure; cached-token field missing."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "Now classify what belongs in release notes versus remaining "
                        "GA blockers."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the PR status.",
                    tool_name="read_pr_status",
                    tool_result=(
                        "Unified PR is draft and mergeable; blog PR is open; branch-image "
                        "benchmark and positive cached-token backend evidence are missing."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the release triage. Include exact tokens "
                        "READY=overlay-evidence, BLOCKER=branch-image, "
                        "BLOCKER=positive-cached-tokens."
                    ),
                    expected_terms=(
                        "READY=overlay-evidence",
                        "BLOCKER=branch-image",
                        "BLOCKER=positive-cached-tokens",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="observability-debug",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "A maintainer asks why the logical model is auto but the physical "
                        "model changed. Plan the observability answer."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the route trace.",
                    tool_name="read_route_trace",
                    tool_result=(
                        "trace includes current_model=qwen-small, selected_model="
                        "frontier-reasoner, decision_drift=true, idle_expired=false, "
                        "hard_locked=false, replay_id=abc123."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same explanation and decide which trace fields "
                        "should be exposed."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the dashboard fields.",
                    tool_name="read_dashboard_headers",
                    tool_result=(
                        "response headers expose x-vsr-selected-model, x-vsr-selected-"
                        "decision, and x-vsr-replay-id."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the debug answer. Include exact tokens "
                        "EXPLAIN=decision-drift, TRACE=replay-id, "
                        "HEADER=x-vsr-selected-model."
                    ),
                    expected_terms=(
                        "EXPLAIN=decision-drift",
                        "TRACE=replay-id",
                        "HEADER=x-vsr-selected-model",
                    ),
                ),
            ),
        ),
    )


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


def build_messages(task: TaskSpec, turn_index: int) -> list[dict[str, Any]]:
    turn = task.turns[turn_index]
    system = {
        "role": "system",
        "content": (
            "You are a concise coding-agent benchmark assistant. Follow exact "
            "answer-token instructions when they are present."
        ),
    }
    messages: list[dict[str, Any]] = [system]
    for prior_index, prior in enumerate(task.turns[:turn_index]):
        messages.append({"role": "user", "content": prior.prompt})
        if prior.expected_terms:
            messages.append({"role": "assistant", "content": "Acknowledged."})
        elif prior.tool_result:
            messages.extend(tool_messages(task.name, prior_index, prior))
        else:
            messages.append(
                {"role": "assistant", "content": "I will inspect the evidence."}
            )
    if turn.tool_result:
        messages.extend(tool_messages(task.name, turn_index, turn))
    prompt = turn.prompt
    if turn.expected_terms:
        prompt = prompt + "\n\n" + scoring_instruction(turn.expected_terms)
    messages.append({"role": "user", "content": prompt})
    return messages


def scoring_instruction(expected_terms: tuple[str, ...]) -> str:
    tokens = "\n".join(f"- {term}" for term in expected_terms)
    return (
        "Scoring uses exact substring matching. Copy every required token "
        "exactly as written, one token per line, before any explanation:\n"
        f"{tokens}"
    )


def tool_messages(
    task_name: str, turn_index: int, turn: TaskTurn
) -> list[dict[str, Any]]:
    call_id = f"call_{task_name}_{turn_index}"
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": turn.tool_name or "read_tool_result",
                        "arguments": json.dumps({"task": task_name}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": turn.tool_result},
    ]


def build_body(
    args: argparse.Namespace,
    task: TaskSpec,
    turn_index: int,
    previous_response_id: str,
) -> dict[str, Any]:
    body = {
        "model": args.model,
        "messages": build_messages(task, turn_index),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.include_previous_response_id and previous_response_id:
        body["previous_response_id"] = previous_response_id
    return body


def send_chat(
    args: argparse.Namespace,
    task: TaskSpec,
    turn_index: int,
    session_id: str,
    previous_response_id: str,
) -> dict[str, Any]:
    if args.dry_run:
        return dry_response(task, turn_index)
    url = args.base_url.rstrip("/") + "/chat/completions"
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


def dry_response(task: TaskSpec, turn_index: int) -> dict[str, Any]:
    turn = task.turns[turn_index]
    content = " ".join(turn.expected_terms) if turn.expected_terms else "Acknowledged."
    return {
        "status": HTTP_OK,
        "headers": {
            "x-vsr-selected-model": "dry-model",
            "x-vsr-selected-decision": "dry-run",
            "x-vsr-selected-confidence": "0.0000",
            "x-vsr-replay-id": f"dry-replay-{task.name}-{turn_index}",
            "x-vsr-context-token-count": "42",
        },
        "json": {
            "id": f"dry_{task.name}_{turn_index}",
            "model": "dry-model",
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        },
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
                result = send_chat(
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
    elapsed = time.perf_counter() - started
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
        args.include_previous_response_id and previous_response_id
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
        "turn": turn_index,
        "phase": turn.phase,
        "status": status,
        "success": HTTP_OK <= status < HTTP_REDIRECT_START,
        "latency_ms": round(float(result.get("latency_ms") or 0), 3),
        "selected_model": selected_model,
        "response_model": response_json.get("model", ""),
        "model_switched": switched,
        "tool_loop_switch_violation": turn.phase == "tool_loop" and switched,
        "context_portability_violation": previous_response_id_sent and switched,
        "previous_response_id_sent": previous_response_id_sent,
        "response_id": response_json.get("id", ""),
        "prompt_tokens": usage_value(response_json, "prompt_tokens"),
        "completion_tokens": usage_value(response_json, "completion_tokens"),
        "cached_tokens": cached_tokens(response_json),
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
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def score_answer(
    content: str, expected_terms: tuple[str, ...]
) -> tuple[float | None, list[str]]:
    if not expected_terms:
        return None, []
    normalized = content.lower()
    missing = [term for term in expected_terms if term.lower() not in normalized]
    found = len(expected_terms) - len(missing)
    return round(found / len(expected_terms), 4), missing


def answer_excerpt(content: str, limit: int = 600) -> str:
    return content.replace("\n", "\\n")[:limit]


def response_id(result: dict[str, Any]) -> str:
    response_json = result.get("json") or {}
    value = response_json.get("id")
    return value if isinstance(value, str) else ""


def usage_value(response_json: dict[str, Any], key: str) -> int:
    usage = response_json.get("usage") or {}
    value = usage.get(key) if isinstance(usage, dict) else 0
    return int(value or 0)


def cached_tokens(response_json: dict[str, Any]) -> int:
    usage = response_json.get("usage") if isinstance(response_json, dict) else {}
    details = usage.get("prompt_tokens_details", {}) if isinstance(usage, dict) else {}
    if not isinstance(details, dict):
        return 0
    return int(details.get("cached_tokens") or 0)


def summarize(
    rows: list[dict[str, Any]], elapsed_seconds: float, label: str
) -> dict[str, Any]:
    scored = [row for row in rows if row["scored_turn"]]
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
        "task_instances": len({task_key(row) for row in scored}),
        "task_suites": counts(row.get("task_suite", "") for row in rows),
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


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(summary: dict[str, Any]) -> str:
    latency = summary["latency_ms"]
    return "\n".join(
        [
            "# Live Agent Task Benchmark",
            "",
            f"- label: {summary['label']}",
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
            "",
        ]
    )


def compare_summaries(
    router_summary: dict[str, Any], baseline_summary: dict[str, Any]
) -> dict[str, Any]:
    return {
        "router_label": router_summary["label"],
        "baseline_label": baseline_summary["label"],
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
    rows, summary = run_tasks(args)
    write_outputs(rows, summary, output_dir)

    baseline_summary = None
    comparison = None
    if args.baseline_base_url:
        baseline_args = namespace_with(
            args,
            base_url=args.baseline_base_url,
            model=args.baseline_model or args.model,
            label=args.baseline_label,
            include_previous_response_id=args.baseline_include_previous_response_id,
        )
        baseline_rows, baseline_summary = run_tasks(baseline_args)
        write_outputs(baseline_rows, baseline_summary, output_dir / "baseline")
        comparison = compare_summaries(summary, baseline_summary)
        write_comparison(comparison, output_dir)

    validation_failures = validate_summary(args, summary)
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
