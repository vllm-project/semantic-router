#!/usr/bin/env python3
"""Probe OpenAI-compatible cached-token reporting for router experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HTTP_OK = 200
HTTP_REDIRECT_START = 300
CACHE_REPORTING_ORDER = {
    "missing": 0,
    "reported_zero": 1,
    "positive": 2,
}
CACHE_PROBE_KIND = "repeated-prefix-cache-token-probe"
CACHE_PROBE_SUFFIX_PATTERN = "probe_turn_index"
CACHED_TOKEN_USAGE_PATHS = (
    (
        "usage.prompt_tokens_details.cached_tokens",
        ("prompt_tokens_details", "cached_tokens"),
    ),
    (
        "usage.input_tokens_details.cached_tokens",
        ("input_tokens_details", "cached_tokens"),
    ),
    ("usage.cached_tokens", ("cached_tokens",)),
    ("usage.prompt_cache_hit_tokens", ("prompt_cache_hit_tokens",)),
)
PROMPT_TOKEN_KEYS = ("prompt_tokens", "input_tokens")
COMPLETION_TOKEN_KEYS = ("completion_tokens", "output_tokens")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--session-header", default="x-session-id")
    parser.add_argument("--session-id", default="cache-probe-session")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--label", default="router")
    parser.add_argument("--extra-header", action="append", default=[])
    parser.add_argument("--baseline-base-url", default="")
    parser.add_argument("--baseline-model", default="")
    parser.add_argument("--baseline-label", default="direct-backend")
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument(
        "--min-cached-token-reporting",
        choices=tuple(CACHE_REPORTING_ORDER.keys()),
        default="missing",
        help=(
            "Minimum cached-token reporting state required for each measured "
            "path. Use 'reported_zero' to require the field and 'positive' to "
            "require a positive cached-token observation."
        ),
    )
    parser.add_argument("--min-cached-token-field-rate", type=float, default=0.0)
    parser.add_argument("--min-cached-prompt-ratio", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path(".agent-harness/experiments/cache-token-probe") / stamp


def parse_extra_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"extra header must be NAME:VALUE, got {value!r}")
        name, raw = value.split(":", 1)
        headers[name.strip()] = raw.strip()
    return headers


def request_headers(args: argparse.Namespace) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        args.session_header: args.session_id,
    }
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    headers.update(parse_extra_headers(args.extra_header))
    return headers


def probe_prompt() -> str:
    reusable = (
        "You are validating prefix cache observability for a session-aware "
        "LLM router. Keep the repeated context stable so the backend has an "
        "opportunity to reuse the prompt prefix. "
    )
    return (reusable * 80).strip()


def probe_prompt_hash() -> str:
    return hashlib.sha256(probe_prompt().encode("utf-8")).hexdigest()[:16]


def cache_probe_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "probe_kind": CACHE_PROBE_KIND,
        "probe_repeats": len(rows),
        "stable_prefix_chars": len(probe_prompt()),
        "stable_prefix_sha256": probe_prompt_hash(),
        "unique_suffix_pattern": CACHE_PROBE_SUFFIX_PATTERN,
    }


def build_body(args: argparse.Namespace, repeat_index: int) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise benchmark assistant.",
            },
            {
                "role": "user",
                "content": (
                    f"{probe_prompt()}\n\n"
                    f"Probe turn {repeat_index}: answer with one short sentence."
                ),
            },
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }


def send_chat(args: argparse.Namespace, repeat_index: int) -> dict[str, Any]:
    url = args.base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(build_body(args, repeat_index)).encode("utf-8")
    started = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers(args),
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
        parsed = {}
    return {
        "status": status,
        "headers": headers,
        "json": parsed if isinstance(parsed, dict) else {},
        "payload": payload,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "error": (
            error_message(parsed, payload) if status >= HTTP_REDIRECT_START else ""
        ),
    }


def error_message(parsed: Any, payload: str) -> str:
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        return str(error.get("message", ""))
    if isinstance(error, str):
        return error
    return payload[:300]


def row_from_result(
    args: argparse.Namespace, repeat_index: int, result: dict[str, Any]
) -> dict[str, Any]:
    response_json = result.get("json") or {}
    return {
        "label": args.label,
        "repeat_index": repeat_index,
        "status": int(result.get("status") or 0),
        "success": HTTP_OK <= int(result.get("status") or 0) < HTTP_REDIRECT_START,
        "latency_ms": round(float(result.get("latency_ms") or 0), 3),
        "prompt_tokens": usage_value(response_json, PROMPT_TOKEN_KEYS),
        "completion_tokens": usage_value(response_json, COMPLETION_TOKEN_KEYS),
        "cached_tokens": cached_tokens(response_json),
        "cached_token_field_present": cached_token_field_present(response_json),
        "cached_token_source": cached_token_source(response_json),
        "response_model": response_json.get("model", ""),
        "error": result.get("error", ""),
    }


def usage_value(response_json: dict[str, Any], keys: tuple[str, ...]) -> int:
    usage = response_json.get("usage") or {}
    if not isinstance(usage, dict):
        return 0
    for key in keys:
        if key in usage:
            return int(usage.get(key) or 0)
    return 0


def cached_tokens(response_json: dict[str, Any]) -> int:
    value = cached_token_value(response_json)
    return int(value or 0)


def cached_token_field_present(response_json: dict[str, Any]) -> bool:
    return cached_token_value(response_json) is not None


def cached_token_source(response_json: dict[str, Any]) -> str:
    _, source = cached_token_observation(response_json)
    return source


def cached_token_value(response_json: dict[str, Any]) -> Any:
    value, _ = cached_token_observation(response_json)
    return value


def cached_token_observation(response_json: dict[str, Any]) -> tuple[Any, str]:
    usage = response_json.get("usage")
    if not isinstance(usage, dict):
        return None, ""
    for source, path in CACHED_TOKEN_USAGE_PATHS:
        value = nested_value(usage, path)
        if value is not None:
            return value, source
    return None, ""


def nested_value(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def run_probe(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for repeat_index in range(args.repeats):
        rows.append(row_from_result(args, repeat_index, send_chat(args, repeat_index)))
    elapsed = time.perf_counter() - started
    return rows, summarize(rows, elapsed, args.label)


def summarize(
    rows: list[dict[str, Any]], elapsed_seconds: float, label: str
) -> dict[str, Any]:
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    cached = sum(int(row["cached_tokens"]) for row in rows)
    successes = sum(1 for row in rows if row["success"])
    field_present = sum(
        1 for row in rows if row["success"] and row["cached_token_field_present"]
    )
    return {
        "label": label,
        "requests": len(rows),
        "successes": successes,
        "success_rate": (round(successes / len(rows), 4) if rows else 0.0),
        "status_counts": counts(str(row["status"]) for row in rows),
        "latency_ms": latency_summary(
            [float(row["latency_ms"]) for row in rows if row["success"]]
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": sum(int(row["completion_tokens"]) for row in rows),
        "cached_tokens": cached,
        "cached_prompt_ratio": (
            round(cached / prompt_tokens, 6) if prompt_tokens else 0.0
        ),
        "cached_token_reporting": cache_reporting_state(rows),
        "cached_token_field_present": field_present,
        "cached_token_field_rate": (
            round(field_present / successes, 4) if successes else 0.0
        ),
        "cached_token_source_counts": counts(
            row.get("cached_token_source", "")
            for row in rows
            if row["success"] and row.get("cached_token_source")
        ),
        **cache_probe_metadata(rows),
        "errors": counts(row["error"] for row in rows if row["error"]),
        "requests_per_second": (
            round(len(rows) / elapsed_seconds, 3) if elapsed_seconds > 0 else None
        ),
    }


def validate_summary(
    args: argparse.Namespace, summary: dict[str, Any], label: str
) -> list[str]:
    failures: list[str] = []
    if args.min_success_rate and summary["success_rate"] < args.min_success_rate:
        failures.append(
            f"{label}: success_rate {summary['success_rate']} < {args.min_success_rate}"
        )
    required_state = args.min_cached_token_reporting
    actual_state = str(summary["cached_token_reporting"])
    if CACHE_REPORTING_ORDER[actual_state] < CACHE_REPORTING_ORDER[required_state]:
        failures.append(
            f"{label}: cached_token_reporting {actual_state} < {required_state}"
        )
    if (
        args.min_cached_token_field_rate
        and summary["cached_token_field_rate"] < args.min_cached_token_field_rate
    ):
        failures.append(
            f"{label}: cached_token_field_rate {summary['cached_token_field_rate']} "
            f"< {args.min_cached_token_field_rate}"
        )
    if (
        args.min_cached_prompt_ratio
        and summary["cached_prompt_ratio"] < args.min_cached_prompt_ratio
    ):
        failures.append(
            f"{label}: cached_prompt_ratio {summary['cached_prompt_ratio']} "
            f"< {args.min_cached_prompt_ratio}"
        )
    return failures


def cache_reporting_state(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "missing"
    present = [
        row for row in rows if row["success"] and row["cached_token_field_present"]
    ]
    if not present:
        return "missing"
    if any(int(row["cached_tokens"]) > 0 for row in present):
        return "positive"
    return "reported_zero"


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
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "turns.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def namespace_with(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    rows, summary = run_probe(args)
    write_outputs(rows, summary, output_dir)

    baseline_summary = None
    if args.baseline_base_url:
        baseline_args = namespace_with(
            args,
            base_url=args.baseline_base_url,
            model=args.baseline_model or args.model,
            label=args.baseline_label,
        )
        baseline_rows, baseline_summary = run_probe(baseline_args)
        write_outputs(baseline_rows, baseline_summary, output_dir / "baseline")

    result = {
        "output_dir": str(output_dir),
        "summary": summary,
        "baseline_summary": baseline_summary,
    }
    failures = validate_summary(args, summary, args.label)
    if baseline_summary is not None:
        failures.extend(validate_summary(args, baseline_summary, args.baseline_label))
    result["validation_failures"] = failures
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
