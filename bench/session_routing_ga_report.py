#!/usr/bin/env python3
"""Aggregate Session-Aware Agentic Routing GA evidence into one gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

CACHE_REPORTING_ORDER = {
    "missing": 0,
    "reported_zero": 1,
    "positive": 2,
}
CACHE_PROBE_KIND = "repeated-prefix-cache-token-probe"
FULL_BRANCH_IMAGE_KINDS = {
    "branch-image-benchmark",
    "full-branch-image-benchmark",
}
MOUNTED_BINARY_MARKERS = ("mounted-binary", "mounted_binary", "mounted binary")
PASSING_STATUS = "passed"
BLOCKING_STATUS = "blocked"
MISSING_STATUS = "missing"
MARKDOWN_METRIC_LIMIT = 220
MARKDOWN_METRIC_TRUNCATED_LIMIT = MARKDOWN_METRIC_LIMIT - 3
DEFAULT_MIN_AGENT_TASK_REQUESTS = 255
DEFAULT_MIN_AGENT_TASK_COUNT = 15
DEFAULT_MIN_AGENT_TASK_INSTANCES = 45
DEFAULT_REQUIRED_AGENT_TASK_NAMES = (
    "multi-file-regression",
    "code-review-followup",
    "research-synthesis",
    "maintainer-handoff",
    "cluster-boundary",
    "session-switch-policy",
    "cache-economics",
    "release-triage",
    "observability-debug",
    "test-fix-iteration",
    "codebase-refactor-planning",
    "research-artifact-review",
    "tool-error-recovery-loop",
    "paper-evidence-audit",
    "multi-agent-delegation",
)
DEFAULT_REQUIRED_AGENT_TASK_PHASES = (
    "user_turn",
    "tool_loop",
    "provider_state",
    "topic_drift",
    "idle_boundary",
    "final",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-matrix-summary", type=Path)
    parser.add_argument("--synthetic-ablation-summary", type=Path)
    parser.add_argument("--live-aggregate", action="append", type=Path, default=[])
    parser.add_argument("--failure-aggregate", action="append", type=Path, default=[])
    parser.add_argument("--agent-task-summary", action="append", type=Path, default=[])
    parser.add_argument("--cache-aggregate", type=Path)
    parser.add_argument("--branch-image-summary", type=Path)
    parser.add_argument("--min-synthetic-turns", type=int, default=1000)
    parser.add_argument("--min-switch-reduction-pct", type=float, default=50.0)
    parser.add_argument("--min-cost-reduction-pct", type=float, default=30.0)
    parser.add_argument("--max-synthetic-quality-loss", type=float, default=0.08)
    parser.add_argument("--min-ablation-policies", type=int, default=4)
    parser.add_argument("--min-live-success-rate", type=float, default=1.0)
    parser.add_argument("--min-failure-success-rate", type=float, default=0.75)
    parser.add_argument("--min-rps-ratio", type=float, default=0.8)
    parser.add_argument("--max-overhead-p95-ms", type=float, default=300.0)
    parser.add_argument("--max-continuity-violations", type=int, default=0)
    parser.add_argument("--min-session-recovery-rate", type=float, default=1.0)
    parser.add_argument("--min-sessions-with-errors", type=int, default=1)
    parser.add_argument("--min-agent-task-success-rate", type=float, default=0.75)
    parser.add_argument("--min-agent-task-score", type=float, default=0.75)
    parser.add_argument(
        "--min-agent-task-requests",
        type=int,
        default=DEFAULT_MIN_AGENT_TASK_REQUESTS,
    )
    parser.add_argument(
        "--min-agent-task-count",
        type=int,
        default=DEFAULT_MIN_AGENT_TASK_COUNT,
    )
    parser.add_argument(
        "--min-agent-task-instances",
        type=int,
        default=DEFAULT_MIN_AGENT_TASK_INSTANCES,
    )
    parser.add_argument(
        "--required-agent-task-name",
        action="append",
        default=list(DEFAULT_REQUIRED_AGENT_TASK_NAMES),
    )
    parser.add_argument(
        "--required-agent-task-phase",
        action="append",
        default=list(DEFAULT_REQUIRED_AGENT_TASK_PHASES),
    )
    parser.add_argument(
        "--required-agent-task-header",
        action="append",
        default=[
            "x-vsr-selected-model",
            "x-vsr-selected-decision",
            "x-vsr-replay-id",
            "x-vsr-session-phase",
            "x-vsr-selected-confidence",
            "x-vsr-context-token-count",
        ],
    )
    parser.add_argument(
        "--min-cached-token-reporting",
        choices=tuple(CACHE_REPORTING_ORDER.keys()),
        default="positive",
    )
    parser.add_argument("--min-cached-token-field-rate", type=float, default=1.0)
    parser.add_argument("--min-cached-prompt-ratio", type=float, default=0.01)
    parser.add_argument("--min-cache-probe-repeats", type=int, default=2)
    parser.add_argument(
        "--require-cache-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require a direct backend baseline in cache-token evidence so "
            "positive cached tokens prove backend behavior, not only router "
            "summary shaping."
        ),
    )
    parser.add_argument("--allow-blockers", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path(".agent-harness/reports/session-routing-ga") / stamp


def load_json(path: Path | None) -> tuple[dict[str, Any] | None, str]:
    if path is None:
        return None, "not configured"
    if not path.exists():
        return None, f"missing file: {path}"
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON in {path}: {exc}"
    if not isinstance(data, dict):
        return None, f"expected JSON object in {path}"
    return data, str(path)


def requirement(
    requirement_id: str,
    title: str,
    status: str,
    evidence: str,
    metrics: dict[str, Any] | None = None,
    failures: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": requirement_id,
        "title": title,
        "status": status,
        "evidence": evidence,
        "metrics": metrics or {},
        "failures": failures or [],
    }


def evaluate_numeric(
    failures: list[str], label: str, actual: Any, operator: str, expected: float
) -> None:
    if actual is None:
        failures.append(f"{label} missing")
        return
    try:
        actual_float = float(actual)
    except (TypeError, ValueError):
        failures.append(f"{label} is not numeric: {actual!r}")
        return
    if operator == ">=" and actual_float < expected:
        failures.append(f"{label} {actual_float} < {expected}")
    elif operator == "<=" and actual_float > expected:
        failures.append(f"{label} {actual_float} > {expected}")


def evaluate_synthetic_matrix(args: argparse.Namespace) -> dict[str, Any]:
    data, evidence = load_json(args.synthetic_matrix_summary)
    if data is None:
        return requirement(
            "synthetic_policy_matrix",
            "Synthetic policy matrix",
            MISSING_STATUS,
            evidence,
            failures=[evidence],
        )
    overall = data.get("overall") if isinstance(data.get("overall"), dict) else data
    failures: list[str] = []
    evaluate_numeric(
        failures, "turns", overall.get("turns"), ">=", args.min_synthetic_turns
    )
    evaluate_numeric(
        failures,
        "switch_reduction_pct",
        overall.get("switch_reduction_pct"),
        ">=",
        args.min_switch_reduction_pct,
    )
    evaluate_numeric(
        failures,
        "cost_reduction_pct",
        overall.get("cost_reduction_pct"),
        ">=",
        args.min_cost_reduction_pct,
    )
    evaluate_numeric(
        failures,
        "quality_delta",
        overall.get("quality_delta"),
        ">=",
        -args.max_synthetic_quality_loss,
    )
    continuity = int(overall.get("agentic_tool_loop_switch_violations") or 0) + int(
        overall.get("agentic_context_portability_violations") or 0
    )
    if continuity > args.max_continuity_violations:
        failures.append(
            f"agentic continuity violations {continuity} > "
            f"{args.max_continuity_violations}"
        )
    metrics = {
        key: overall.get(key)
        for key in [
            "turns",
            "switch_reduction_pct",
            "cost_reduction_pct",
            "quality_delta",
            "agentic_tool_loop_switch_violations",
            "agentic_context_portability_violations",
        ]
    }
    return requirement(
        "synthetic_policy_matrix",
        "Synthetic policy matrix",
        BLOCKING_STATUS if failures else PASSING_STATUS,
        evidence,
        metrics,
        failures,
    )


def evaluate_synthetic_ablation(args: argparse.Namespace) -> dict[str, Any]:
    data, evidence = load_json(args.synthetic_ablation_summary)
    if data is None:
        return requirement(
            "synthetic_policy_ablation",
            "Synthetic policy ablation",
            MISSING_STATUS,
            evidence,
            failures=[evidence],
        )
    policies = data.get("by_policy")
    if not isinstance(policies, list):
        return requirement(
            "synthetic_policy_ablation",
            "Synthetic policy ablation",
            BLOCKING_STATUS,
            evidence,
            failures=["by_policy missing"],
        )
    names = [str(row.get("policy", "")) for row in policies if isinstance(row, dict)]
    normalized_names = [normalize_policy_name(name) for name in names]
    full = next(
        (
            row
            for row in policies
            if isinstance(row, dict)
            and normalize_policy_name(str(row.get("policy", "")))
            in {"full-acr", "acr-full"}
        ),
        None,
    )
    failures: list[str] = []
    if len(names) < args.min_ablation_policies:
        failures.append(f"policy count {len(names)} < {args.min_ablation_policies}")
    if "single-turn" not in normalized_names:
        failures.append("single-turn baseline missing")
    if "acr-initial" not in normalized_names:
        failures.append("initial implementation baseline missing")
    if full is None:
        failures.append("full ACR policy missing")
    else:
        continuity = int(full.get("tool_loop_switch_violations") or 0) + int(
            full.get("context_portability_violations") or 0
        )
        if continuity > args.max_continuity_violations:
            failures.append(
                f"full ACR continuity violations {continuity} > "
                f"{args.max_continuity_violations}"
            )
    metrics = {
        "policy_count": len(names),
        "policies": names,
        "full_acr": full or {},
    }
    return requirement(
        "synthetic_policy_ablation",
        "Synthetic policy ablation",
        BLOCKING_STATUS if failures else PASSING_STATUS,
        evidence,
        metrics,
        failures,
    )


def normalize_policy_name(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def task_name_from_instance(value: str) -> str:
    if ":" not in value:
        return value
    return value.split(":", 1)[1]


def strings_from_list(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value if str(item)}


def agent_task_names(data: dict[str, Any]) -> set[str]:
    names = strings_from_list(data.get("task_names"))
    names.update(strings_from_list(data.get("scored_task_names")))
    final_scores = data.get("final_scores")
    if isinstance(final_scores, dict):
        names.update(task_name_from_instance(str(key)) for key in final_scores)
    return {name for name in names if name}


def agent_task_phases(data: dict[str, Any]) -> set[str]:
    phase_counts = data.get("phase_counts")
    if not isinstance(phase_counts, dict):
        return set()
    return {str(key) for key, value in phase_counts.items() if int(value or 0) > 0}


def validation_failures(row: dict[str, Any]) -> list[Any]:
    failures = row.get("validation_failures")
    return failures if isinstance(failures, list) else []


def continuity_violations(row: dict[str, Any]) -> int:
    return int(row.get("tool_loop_switch_violations") or 0) + int(
        row.get("context_portability_violations") or 0
    )


def rows_from_aggregate(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return [data]


def evaluate_live_aggregate(
    args: argparse.Namespace, path: Path, index: int
) -> dict[str, Any]:
    data, evidence = load_json(path)
    requirement_id = f"amd_live_matrix_{index}"
    if data is None:
        return requirement(
            requirement_id,
            "AMD live matrix",
            MISSING_STATUS,
            evidence,
            failures=[evidence],
        )
    rows = rows_from_aggregate(data)
    failures: list[str] = []
    metrics_rows: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("name") or row.get("label") or evidence)
        success_rate = row.get("router_success_rate", row.get("success_rate"))
        evaluate_numeric(
            failures,
            f"{name} success_rate",
            success_rate,
            ">=",
            args.min_live_success_rate,
        )
        continuity = continuity_violations(row)
        if continuity > args.max_continuity_violations:
            failures.append(
                f"{name} continuity violations {continuity} > "
                f"{args.max_continuity_violations}"
            )
        if validation_failures(row):
            failures.append(f"{name} validation_failures={validation_failures(row)}")
        rps_ratio = row.get("rps_ratio")
        if rps_ratio is not None:
            evaluate_numeric(
                failures, f"{name} rps_ratio", rps_ratio, ">=", args.min_rps_ratio
            )
        overhead_p95 = row.get("overhead_p95_ms")
        if overhead_p95 is not None and args.max_overhead_p95_ms:
            evaluate_numeric(
                failures,
                f"{name} overhead_p95_ms",
                overhead_p95,
                "<=",
                args.max_overhead_p95_ms,
            )
        metrics_rows.append(
            {
                "name": name,
                "requests": row.get("router_requests", row.get("requests")),
                "success_rate": success_rate,
                "rps_ratio": rps_ratio,
                "overhead_p95_ms": overhead_p95,
                "continuity_violations": continuity,
            }
        )
    return requirement(
        requirement_id,
        "AMD live matrix",
        BLOCKING_STATUS if failures else PASSING_STATUS,
        evidence,
        {"rows": metrics_rows},
        failures,
    )


def evaluate_failure_aggregate(
    args: argparse.Namespace, path: Path, index: int
) -> dict[str, Any]:
    data, evidence = load_json(path)
    requirement_id = f"amd_failure_recovery_{index}"
    if data is None:
        return requirement(
            requirement_id,
            "AMD failure recovery",
            MISSING_STATUS,
            evidence,
            failures=[evidence],
        )
    rows = rows_from_aggregate(data)
    failures: list[str] = []
    metrics_rows: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("name") or row.get("label") or evidence)
        evaluate_numeric(
            failures,
            f"{name} success_rate",
            row.get("success_rate"),
            ">=",
            args.min_failure_success_rate,
        )
        continuity = continuity_violations(row)
        if continuity > args.max_continuity_violations:
            failures.append(
                f"{name} continuity violations {continuity} > "
                f"{args.max_continuity_violations}"
            )
        if validation_failures(row):
            failures.append(f"{name} validation_failures={validation_failures(row)}")
        injected = int(row.get("injected") or 0)
        if injected <= 0:
            failures.append(f"{name} injected failures missing")
        sessions = int(row.get("sessions_with_errors") or 0)
        if sessions < args.min_sessions_with_errors:
            failures.append(
                f"{name} sessions_with_errors {sessions} < "
                f"{args.min_sessions_with_errors}"
            )
        evaluate_numeric(
            failures,
            f"{name} session_recovery_rate_after_error",
            row.get("session_recovery_rate_after_error"),
            ">=",
            args.min_session_recovery_rate,
        )
        metrics_rows.append(
            {
                "name": name,
                "requests": row.get("requests"),
                "success_rate": row.get("success_rate"),
                "injected": injected,
                "sessions_with_errors": sessions,
                "session_recovery_rate_after_error": row.get(
                    "session_recovery_rate_after_error"
                ),
                "continuity_violations": continuity,
            }
        )
    return requirement(
        requirement_id,
        "AMD failure recovery",
        BLOCKING_STATUS if failures else PASSING_STATUS,
        evidence,
        {"rows": metrics_rows},
        failures,
    )


def evaluate_agent_task_summary(
    args: argparse.Namespace, path: Path, index: int
) -> dict[str, Any]:
    data, evidence = load_json(path)
    requirement_id = f"amd_agent_task_quality_{index}"
    if data is None:
        return requirement(
            requirement_id,
            "AMD agent-task quality",
            MISSING_STATUS,
            evidence,
            failures=[evidence],
        )
    failures: list[str] = []
    evaluate_numeric(
        failures,
        "success_rate",
        data.get("success_rate"),
        ">=",
        args.min_live_success_rate,
    )
    evaluate_numeric(
        failures,
        "requests",
        data.get("requests"),
        ">=",
        args.min_agent_task_requests,
    )
    evaluate_numeric(
        failures,
        "task_count",
        data.get("task_count", data.get("tasks")),
        ">=",
        args.min_agent_task_count,
    )
    evaluate_numeric(
        failures,
        "task_instances",
        data.get("task_instances"),
        ">=",
        args.min_agent_task_instances,
    )
    evaluate_numeric(
        failures,
        "task_success_rate",
        data.get("task_success_rate"),
        ">=",
        args.min_agent_task_success_rate,
    )
    evaluate_numeric(
        failures,
        "task_score_mean",
        data.get("task_score_mean"),
        ">=",
        args.min_agent_task_score,
    )
    continuity = continuity_violations(data)
    if continuity > args.max_continuity_violations:
        failures.append(
            f"continuity violations {continuity} > {args.max_continuity_violations}"
        )
    required_headers = list(dict.fromkeys(args.required_agent_task_header))
    missing_headers = missing_required_router_headers(data, required_headers)
    invalid_counts = data.get("invalid_router_header_counts") or {}
    invalid_headers = {
        key: invalid_counts.get(key, 0)
        for key in required_headers
        if int(invalid_counts.get(key, 0) or 0) > 0
    }
    if missing_headers:
        failures.append(f"missing router headers: {missing_headers}")
    if invalid_headers:
        failures.append(f"invalid router headers: {invalid_headers}")
    task_names = agent_task_names(data)
    required_task_names = set(args.required_agent_task_name or [])
    missing_task_names = sorted(required_task_names - task_names)
    if missing_task_names:
        failures.append(f"missing task names: {missing_task_names}")
    phase_names = agent_task_phases(data)
    required_phases = set(args.required_agent_task_phase or [])
    missing_phases = sorted(required_phases - phase_names)
    if missing_phases:
        failures.append(f"missing task phases: {missing_phases}")
    metrics = {
        "requests": data.get("requests"),
        "task_count": data.get("task_count", data.get("tasks")),
        "task_instances": data.get("task_instances"),
        "scored_task_count": data.get("scored_task_count"),
        "task_names": sorted(task_names),
        "required_task_names": sorted(required_task_names),
        "phase_counts": data.get("phase_counts", {}),
        "required_phases": sorted(required_phases),
        "success_rate": data.get("success_rate"),
        "task_success_rate": data.get("task_success_rate"),
        "task_score_mean": data.get("task_score_mean"),
        "continuity_violations": continuity,
        "required_headers": required_headers,
    }
    return requirement(
        requirement_id,
        "AMD agent-task quality",
        BLOCKING_STATUS if failures else PASSING_STATUS,
        evidence,
        metrics,
        failures,
    )


def missing_required_router_headers(
    data: dict[str, Any], required_headers: list[str]
) -> dict[str, int]:
    missing_counts = data.get("missing_router_header_counts")
    missing = missing_counts if isinstance(missing_counts, dict) else {}
    default_missing = int(data.get("successes", data.get("requests", 0)) or 0)
    result: dict[str, int] = {}
    for key in required_headers:
        count = int(missing.get(key, default_missing) or 0)
        if count > 0:
            result[key] = count
    return result


def cache_paths(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(data.get("router"), dict):
        paths = [("router", data["router"])]
        if isinstance(data.get("baseline"), dict):
            paths.append(("baseline", data["baseline"]))
        return paths
    paths = []
    if isinstance(data.get("summary"), dict):
        paths.append(("router", data["summary"]))
    if isinstance(data.get("baseline_summary"), dict):
        paths.append(("baseline", data["baseline_summary"]))
    return paths or [("router", data)]


def cached_token_field_rate(summary: dict[str, Any]) -> float:
    if "cached_token_field_rate" in summary:
        return float(summary.get("cached_token_field_rate") or 0.0)
    present = summary.get("cached_token_field_present")
    successes = summary.get("successes") or summary.get("requests") or 0
    if not successes:
        return 0.0
    return round(float(present or 0) / float(successes), 4)


def should_require_cache_probe_metadata(required_reporting: str) -> bool:
    return (
        CACHE_REPORTING_ORDER[required_reporting] >= CACHE_REPORTING_ORDER["positive"]
    )


def cache_probe_repeats(summary: dict[str, Any]) -> int:
    for key in ("probe_repeats", "requests"):
        try:
            return int(summary.get(key) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def evaluate_cache_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    data, evidence = load_json(args.cache_aggregate)
    if data is None:
        return requirement(
            "cache_token_reporting",
            "Cache-token reporting",
            MISSING_STATUS,
            evidence,
            failures=[evidence],
        )
    failures: list[str] = []
    metrics: dict[str, Any] = {}
    required = args.min_cached_token_reporting
    require_probe_metadata = should_require_cache_probe_metadata(required)
    paths = cache_paths(data)
    has_baseline = any(label == "baseline" for label, _ in paths)
    if args.require_cache_baseline and not has_baseline:
        failures.append(
            "direct backend baseline cache evidence missing; run "
            "cache_token_probe.py with --baseline-base-url"
        )
    for label, summary in paths:
        reporting = str(summary.get("cached_token_reporting", "missing"))
        if CACHE_REPORTING_ORDER.get(reporting, -1) < CACHE_REPORTING_ORDER[required]:
            failures.append(f"{label}: cached_token_reporting {reporting} < {required}")
        probe_kind = str(summary.get("probe_kind", ""))
        probe_repeats = cache_probe_repeats(summary)
        if require_probe_metadata and probe_kind != CACHE_PROBE_KIND:
            failures.append(
                f"{label}: probe_kind {probe_kind or 'missing'} != {CACHE_PROBE_KIND}"
            )
        if require_probe_metadata and probe_repeats < args.min_cache_probe_repeats:
            failures.append(
                f"{label}: probe_repeats {probe_repeats} < {args.min_cache_probe_repeats}"
            )
        field_rate = cached_token_field_rate(summary)
        if field_rate < args.min_cached_token_field_rate:
            failures.append(
                f"{label}: cached_token_field_rate {field_rate} < "
                f"{args.min_cached_token_field_rate}"
            )
        ratio = float(summary.get("cached_prompt_ratio") or 0.0)
        if ratio < args.min_cached_prompt_ratio:
            failures.append(
                f"{label}: cached_prompt_ratio {ratio} < "
                f"{args.min_cached_prompt_ratio}"
            )
        if float(summary.get("success_rate") or 0.0) < args.min_live_success_rate:
            failures.append(
                f"{label}: success_rate {summary.get('success_rate')} < "
                f"{args.min_live_success_rate}"
            )
        metrics[label] = {
            "requests": summary.get("requests"),
            "success_rate": summary.get("success_rate"),
            "cached_token_reporting": reporting,
            "cached_token_field_rate": field_rate,
            "cached_prompt_ratio": summary.get("cached_prompt_ratio"),
            "probe_kind": probe_kind,
            "probe_repeats": probe_repeats,
            "stable_prefix_chars": summary.get("stable_prefix_chars"),
        }
    metrics["baseline_required"] = args.require_cache_baseline
    metrics["baseline_present"] = has_baseline
    return requirement(
        "cache_token_reporting",
        "Cache-token reporting",
        BLOCKING_STATUS if failures else PASSING_STATUS,
        evidence,
        metrics,
        failures,
    )


def evaluate_branch_image_summary(args: argparse.Namespace) -> dict[str, Any]:
    data, evidence = load_json(args.branch_image_summary)
    if data is None:
        return requirement(
            "branch_image_amd_validation",
            "Branch-image AMD benchmark",
            MISSING_STATUS,
            evidence,
            failures=["full branch-image AMD benchmark artifact is required before GA"],
        )
    failures = [str(item) for item in validation_failures(data)]
    checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
    for key in ["chat_completion_ok", "diagnostic_headers_ok"]:
        if key in checks and checks[key] is not True:
            failures.append(f"{key} is not true")
    for header in data.get("missing_diagnostic_headers") or []:
        failures.append(f"missing diagnostic header: {header}")
    for header in data.get("invalid_diagnostic_headers") or []:
        failures.append(f"invalid diagnostic header: {header}")
    validation_kind = normalize_policy_name(
        str(
            data.get("validation_kind")
            or data.get("benchmark_kind")
            or data.get("kind")
            or ""
        )
    )
    image_tag = str(data.get("image_tag", ""))
    label = str(data.get("label", ""))
    full_branch_image = bool(data.get("branch_image_benchmark")) or (
        validation_kind in FULL_BRANCH_IMAGE_KINDS
    )
    source_fingerprint = " ".join([validation_kind, image_tag, label]).lower()
    mounted_binary = any(
        marker in source_fingerprint for marker in MOUNTED_BINARY_MARKERS
    )
    if not full_branch_image:
        failures.append(
            "full branch-image benchmark marker missing; diagnostic probes do not "
            "satisfy GA"
        )
    if mounted_binary:
        failures.append(
            "mounted-binary diagnostic does not satisfy full branch-image AMD "
            "benchmark"
        )
    failures = list(dict.fromkeys(failures))
    status = BLOCKING_STATUS if failures else PASSING_STATUS
    return requirement(
        "branch_image_amd_validation",
        "Branch-image AMD benchmark",
        status,
        evidence,
        {
            "validation_kind": validation_kind,
            "checks": checks,
            "image_tag": image_tag,
            "label": label,
            "ref": data.get("ref", ""),
        },
        failures,
    )


def generate_report(args: argparse.Namespace) -> dict[str, Any]:
    requirements: list[dict[str, Any]] = [
        evaluate_synthetic_matrix(args),
        evaluate_synthetic_ablation(args),
    ]
    requirements.extend(
        evaluate_live_aggregate(args, path, idx)
        for idx, path in enumerate(args.live_aggregate, start=1)
    )
    requirements.extend(
        evaluate_failure_aggregate(args, path, idx)
        for idx, path in enumerate(args.failure_aggregate, start=1)
    )
    requirements.extend(
        evaluate_agent_task_summary(args, path, idx)
        for idx, path in enumerate(args.agent_task_summary, start=1)
    )
    requirements.extend(
        [
            evaluate_cache_aggregate(args),
            evaluate_branch_image_summary(args),
        ]
    )
    blockers = [
        item
        for item in requirements
        if item["status"] in {BLOCKING_STATUS, MISSING_STATUS}
    ]
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ga_ready": not blockers,
        "blocker_count": len(blockers),
        "passed_count": sum(
            1 for item in requirements if item["status"] == PASSING_STATUS
        ),
        "blockers": blockers,
        "requirements": requirements,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Session-Aware Agentic Routing GA Readiness",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- ga_ready: {str(report['ga_ready']).lower()}",
        f"- passed: {report['passed_count']}",
        f"- blockers: {report['blocker_count']}",
        "",
        "| Requirement | Status | Evidence | Key Metrics | Blockers |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report["requirements"]:
        metrics = compact_metrics(item.get("metrics", {}))
        failures = compact_failures(item.get("failures", []))
        lines.append(
            "| {title} | {status} | `{evidence}` | {metrics} | {failures} |".format(
                title=markdown_table_cell(item["title"]),
                status=markdown_table_cell(item["status"]),
                evidence=markdown_table_cell(item["evidence"]),
                metrics=markdown_table_cell(metrics),
                failures=markdown_table_cell(failures),
            )
        )
    lines.append("")
    return "\n".join(lines)


def compact_failures(failures: list[Any]) -> str:
    if not failures:
        return ""
    return "<br>".join(
        f"{index}. {failure}" for index, failure in enumerate(failures, 1)
    )


def markdown_table_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", r"\|")


def compact_metrics(metrics: Any) -> str:
    if not metrics:
        return ""
    rendered = json.dumps(metrics, sort_keys=True)
    if len(rendered) <= MARKDOWN_METRIC_LIMIT:
        return rendered
    return rendered[:MARKDOWN_METRIC_TRUNCATED_LIMIT] + "..."


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ga-readiness.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "ga-readiness.md").write_text(render_markdown(report))


def stdout_blocker_summary(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.get("id", ""),
            "title": item.get("title", ""),
            "status": item.get("status", ""),
        }
        for item in report.get("blockers", [])
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or default_output_dir()
    report = generate_report(args)
    write_report(report, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "ga_ready": report["ga_ready"],
                "blocker_count": report["blocker_count"],
                "blockers": stdout_blocker_summary(report),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["ga_ready"] or args.allow_blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
