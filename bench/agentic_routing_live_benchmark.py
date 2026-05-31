#!/usr/bin/env python3
"""Run live session-aware agentic routing benchmarks against an OpenAI API.

The deterministic policy benchmark in ``agentic_routing_experiment.py`` checks
control-plane invariants. This script measures a running router stack: request
success, latency, selected-model continuity, cached-token evidence, and
session-aware violation counters under repeatable agentic workloads.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
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


@dataclass(frozen=True)
class TurnPlan:
    session_id: str
    turn: int
    phase: str
    prompt: str
    previous_response_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--session-header", default="x-session-id")
    parser.add_argument("--scenario", default="balanced", choices=scenario_names())
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--label", default="live-router")
    parser.add_argument("--include-previous-response-id", action="store_true")
    parser.add_argument("--metrics-url", default="")
    parser.add_argument("--extra-header", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def scenario_names() -> list[str]:
    return [
        "balanced",
        "tool-heavy",
        "frontier-heavy",
        "idle-heavy",
        "stateful-heavy",
        "drift-heavy",
    ]


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(".agent-harness/experiments/live-agentic-routing") / stamp


def phase_for_turn(scenario: str, turn: int) -> str:
    if scenario == "tool-heavy":
        return "tool_loop" if turn % 5 in (2, 3, 4) else "user_turn"
    if scenario == "frontier-heavy":
        return "frontier_turn" if turn % 4 in (1, 2) else "user_turn"
    if scenario == "idle-heavy":
        return "idle_boundary" if turn > 0 and turn % 4 == 0 else "user_turn"
    if scenario == "stateful-heavy":
        return "provider_state" if turn > 0 and turn % 5 in (1, 2) else "user_turn"
    if scenario == "drift-heavy":
        return "topic_drift" if turn > 0 and turn % 6 in (1, 2) else "user_turn"
    return "tool_loop" if turn % 6 in (2, 3) else "user_turn"


def prompt_for_phase(scenario: str, session_idx: int, turn: int, phase: str) -> str:
    base = f"Session {session_idx}, turn {turn}. "
    if phase == "tool_loop":
        return (
            base
            + "Use the provided tool result and continue the investigation in one concise sentence."
        )
    if phase == "provider_state":
        return (
            base
            + "Continue from the previous response state and preserve the same analysis thread."
        )
    if phase == "topic_drift":
        return (
            base
            + "The task direction changed: now compare the routing policy against a simpler baseline."
        )
    if phase == "frontier_turn":
        return (
            base
            + "Solve a harder planning step with careful reasoning but keep the answer short."
        )
    if phase == "idle_boundary":
        return (
            base
            + "This follows an idle pause. Re-evaluate the next best model for a fresh subtask."
        )
    return base + f"Summarize the next action for the {scenario} agent workload."


def build_messages(plan: TurnPlan) -> list[dict[str, Any]]:
    system = {
        "role": "system",
        "content": "You are a concise benchmark assistant. Answer in one short sentence.",
    }
    if plan.phase != "tool_loop":
        return [system, {"role": "user", "content": plan.prompt}]

    tool_call_id = f"call_{plan.session_id}_{plan.turn}"
    return [
        system,
        {"role": "user", "content": "Look up the current routing task status."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "lookup_task_status",
                        "arguments": json.dumps({"task": plan.session_id}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps({"status": "tool result ready", "turn": plan.turn}),
        },
        {"role": "user", "content": plan.prompt},
    ]


def build_body(args: argparse.Namespace, plan: TurnPlan) -> dict[str, Any]:
    body = {
        "model": args.model,
        "messages": build_messages(plan),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.include_previous_response_id and plan.previous_response_id:
        body["previous_response_id"] = plan.previous_response_id
    return body


def parse_extra_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--extra-header must be name=value, got {value!r}")
        name, header_value = value.split("=", 1)
        headers[name.strip()] = header_value.strip()
    return headers


def post_json(
    url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
            return response_record(
                response.status, dict(response.headers), payload, started
            )
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        return response_record(exc.code, dict(exc.headers), payload, started)
    except Exception as exc:  # pragma: no cover - network errors vary by platform
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "status": 0,
            "latency_ms": elapsed_ms,
            "headers": {},
            "json": {},
            "error": str(exc),
        }


def response_record(
    status: int, headers: dict[str, str], payload: str, started: float
) -> dict[str, Any]:
    elapsed_ms = (time.perf_counter() - started) * 1000
    try:
        parsed = json.loads(payload) if payload else {}
    except json.JSONDecodeError:
        parsed = {"raw_body": payload[:500]}
    return {
        "status": status,
        "latency_ms": elapsed_ms,
        "headers": {key.lower(): value for key, value in headers.items()},
        "json": parsed,
        "error": (
            ""
            if HTTP_OK <= status < HTTP_REDIRECT_START
            else error_message(parsed, payload)
        ),
    }


def error_message(parsed: dict[str, Any], payload: str) -> str:
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        return str(error.get("message", ""))
    if isinstance(error, str):
        return error
    return payload[:300]


def run_session(
    args: argparse.Namespace, session_idx: int, base_headers: dict[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    session_id = f"{args.label}-{args.scenario}-{session_idx:04d}"
    previous_response_id = ""
    previous_selected_model = ""
    url = args.base_url.rstrip("/") + "/chat/completions"

    for turn in range(args.turns):
        phase = phase_for_turn(args.scenario, turn)
        plan = TurnPlan(
            session_id=session_id,
            turn=turn,
            phase=phase,
            prompt=prompt_for_phase(args.scenario, session_idx, turn, phase),
            previous_response_id=previous_response_id,
        )
        if args.dry_run:
            result = dry_run_response(plan)
        else:
            headers = dict(base_headers)
            headers[args.session_header] = session_id
            result = post_json(url, build_body(args, plan), headers, args.timeout)
        row = row_from_result(args, plan, result, previous_selected_model)
        rows.append(row)
        if row["selected_model"]:
            previous_selected_model = row["selected_model"]
        previous_response_id = response_id(result)
    return rows


def dry_run_response(plan: TurnPlan) -> dict[str, Any]:
    model = (
        "frontier-reasoner"
        if plan.phase in {"frontier_turn", "topic_drift"}
        else "qwen-small"
    )
    return {
        "status": 200,
        "latency_ms": 0.0,
        "headers": {
            "x-vsr-selected-model": model,
            "x-vsr-selected-decision": "dry-run",
        },
        "json": {
            "id": f"dry_{plan.session_id}_{plan.turn}",
            "model": model,
            "usage": {},
        },
        "error": "",
    }


def row_from_result(
    args: argparse.Namespace,
    plan: TurnPlan,
    result: dict[str, Any],
    previous_selected_model: str,
) -> dict[str, Any]:
    response_json = result.get("json") or {}
    headers = result.get("headers") or {}
    selected_model = selected_model_from(result)
    status = int(result.get("status") or 0)
    switched = bool(
        previous_selected_model
        and selected_model
        and selected_model != previous_selected_model
    )
    return {
        "label": args.label,
        "scenario": args.scenario,
        "session_id": plan.session_id,
        "turn": plan.turn,
        "phase": plan.phase,
        "status": status,
        "success": HTTP_OK <= status < HTTP_REDIRECT_START,
        "latency_ms": round(float(result.get("latency_ms") or 0), 3),
        "selected_model": selected_model,
        "response_model": response_json.get("model", ""),
        "model_switched": switched,
        "tool_loop_switch_violation": plan.phase == "tool_loop" and switched,
        "context_portability_violation": bool(plan.previous_response_id) and switched,
        "previous_response_id_sent": bool(plan.previous_response_id),
        "response_id": response_json.get("id", ""),
        "prompt_tokens": usage_value(response_json, "prompt_tokens"),
        "completion_tokens": usage_value(response_json, "completion_tokens"),
        "cached_tokens": cached_tokens(response_json),
        "error": result.get("error", ""),
        **{header: headers.get(header, "") for header in VSR_HEADERS},
    }


def selected_model_from(result: dict[str, Any]) -> str:
    headers = result.get("headers") or {}
    response_json = result.get("json") or {}
    return headers.get("x-vsr-selected-model") or response_json.get("model", "")


def response_id(result: dict[str, Any]) -> str:
    response_json = result.get("json") or {}
    value = response_json.get("id")
    return value if isinstance(value, str) else ""


def usage_value(response_json: dict[str, Any], key: str) -> int:
    usage = response_json.get("usage") if isinstance(response_json, dict) else {}
    value = usage.get(key, 0) if isinstance(usage, dict) else 0
    return int(value or 0)


def cached_tokens(response_json: dict[str, Any]) -> int:
    usage = response_json.get("usage") if isinstance(response_json, dict) else {}
    details = usage.get("prompt_tokens_details", {}) if isinstance(usage, dict) else {}
    if not isinstance(details, dict):
        return 0
    return int(details.get("cached_tokens") or 0)


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "vllm-sr-agentic-live-benchmark",
    }
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    headers.update(parse_extra_headers(args.extra_header))

    workers = max(1, min(args.concurrency, args.sessions))
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_session, args, idx, headers) for idx in range(args.sessions)
        ]
        for future in as_completed(futures):
            rows.extend(future.result())
    return sorted(rows, key=lambda row: (row["session_id"], row["turn"]))


def summarize(
    rows: list[dict[str, Any]],
    metrics_text: str = "",
    wall_time_seconds: float | None = None,
) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows if row["success"]]
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    completion_tokens = sum(int(row["completion_tokens"]) for row in rows)
    cached = sum(int(row["cached_tokens"]) for row in rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
    return {
        "requests": len(rows),
        "sessions": len({row["session_id"] for row in rows}),
        "successes": sum(bool(row["success"]) for row in rows),
        "success_rate": round(
            sum(bool(row["success"]) for row in rows) / max(len(rows), 1), 4
        ),
        "wall_time_seconds": (
            round(wall_time_seconds, 3) if wall_time_seconds is not None else None
        ),
        "requests_per_second": (
            round(len(rows) / wall_time_seconds, 3)
            if wall_time_seconds and wall_time_seconds > 0
            else None
        ),
        "status_counts": status_counts,
        "latency_ms": latency_summary(latencies),
        "phase_latency_ms": phase_latency(rows),
        "model_switches": sum(bool(row["model_switched"]) for row in rows),
        "tool_loop_switch_violations": sum(
            bool(row["tool_loop_switch_violation"]) for row in rows
        ),
        "context_portability_violations": sum(
            bool(row["context_portability_violation"]) for row in rows
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached,
        "cached_prompt_ratio": (
            round(cached / prompt_tokens, 4) if prompt_tokens else None
        ),
        "selected_model_counts": counts(
            row["selected_model"] for row in rows if row["selected_model"]
        ),
        "phase_counts": counts(row["phase"] for row in rows),
        "metrics_excerpt": metrics_text[:4000],
    }


def phase_latency(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    by_phase: dict[str, list[float]] = {}
    for row in rows:
        if row["success"]:
            by_phase.setdefault(str(row["phase"]), []).append(float(row["latency_ms"]))
    return {
        phase: latency_summary(values) for phase, values in sorted(by_phase.items())
    }


def latency_summary(latencies: list[float]) -> dict[str, float | None]:
    if not latencies:
        return {"mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "mean": round(statistics.mean(latencies), 3),
        "p50": round(percentile(latencies, 50), 3),
        "p95": round(percentile(latencies, 95), 3),
        "p99": round(percentile(latencies, 99), 3),
        "max": round(max(latencies), 3),
    }


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def fetch_metrics(metrics_url: str, timeout: float) -> str:
    if not metrics_url:
        return ""
    try:
        with urllib.request.urlopen(metrics_url, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - network errors vary by platform
        return f"metrics_fetch_error: {exc}"
    prefixes = ("llm_decision_match_total", "llm_model_selection")
    lines = [line for line in raw.splitlines() if line.startswith(prefixes)]
    return "\n".join(lines[:200])


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
            "# Live Agentic Routing Benchmark",
            "",
            f"- requests: {summary['requests']}",
            f"- sessions: {summary['sessions']}",
            f"- success rate: {summary['success_rate']}",
            f"- throughput rps: {summary['requests_per_second']}",
            f"- latency mean/p95/max ms: {latency['mean']} / {latency['p95']} / {latency['max']}",
            f"- model switches: {summary['model_switches']}",
            f"- tool-loop switch violations: {summary['tool_loop_switch_violations']}",
            f"- context portability violations: {summary['context_portability_violations']}",
            f"- cached prompt ratio: {summary['cached_prompt_ratio']}",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    started = time.perf_counter()
    rows = run_benchmark(args)
    elapsed = time.perf_counter() - started
    metrics_text = fetch_metrics(args.metrics_url, args.timeout)
    summary = summarize(rows, metrics_text, elapsed)
    write_outputs(rows, summary, output_dir)
    print(
        json.dumps(
            {"output_dir": str(output_dir), "summary": summary},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
