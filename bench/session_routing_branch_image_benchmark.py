#!/usr/bin/env python3
"""Assemble full branch-image session-routing benchmark evidence.

This script does not replace the individual AMD benchmark commands. It turns
their machine-readable summaries into a single artifact that can satisfy the
GA report's branch-image requirement only when the evidence came from a real
branch image and all required subchecks passed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    from session_routing_ga_report import (
        CACHE_PROBE_KIND,
        CACHE_REPORTING_ORDER,
        DEFAULT_MIN_AGENT_TASK_COUNT,
        DEFAULT_MIN_AGENT_TASK_INSTANCES,
        DEFAULT_MIN_AGENT_TASK_REQUESTS,
        DEFAULT_REQUIRED_AGENT_TASK_NAMES,
        DEFAULT_REQUIRED_AGENT_TASK_PHASES,
        MOUNTED_BINARY_MARKERS,
    )
except ImportError:  # pragma: no cover - used when loaded outside bench/.
    CACHE_REPORTING_ORDER = {
        "missing": 0,
        "reported_zero": 1,
        "positive": 2,
    }
    CACHE_PROBE_KIND = "repeated-prefix-cache-token-probe"
    DEFAULT_MIN_AGENT_TASK_REQUESTS = 453
    DEFAULT_MIN_AGENT_TASK_COUNT = 26
    DEFAULT_MIN_AGENT_TASK_INSTANCES = 78
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
        "issue-pr-maintenance-loop",
        "configuration-contract-review",
        "repo-bisect-debug",
        "dependency-upgrade-regression",
        "literature-data-extraction",
        "stale-pr-rebase-triage",
        "benchmark-regression-root-cause",
        "paper-figure-quality-review",
        "feature-implementation-loop",
        "research-claim-grounding-loop",
        "tool-timeout-retry-loop",
    )
    DEFAULT_REQUIRED_AGENT_TASK_PHASES = (
        "user_turn",
        "tool_loop",
        "provider_state",
        "topic_drift",
        "idle_boundary",
        "final",
    )
    MOUNTED_BINARY_MARKERS = ("mounted-binary", "mounted_binary", "mounted binary")

DEFAULT_REQUIRED_HEADERS = (
    "x-vsr-selected-model",
    "x-vsr-selected-decision",
    "x-vsr-replay-id",
    "x-vsr-session-phase",
    "x-vsr-selected-confidence",
    "x-vsr-context-token-count",
)
FULL_BRANCH_IMAGE_KIND = "full-branch-image-benchmark"
EVIDENCE_REF_KEYS = ("evidence_ref", "ref")
EVIDENCE_IMAGE_TAG_KEYS = ("evidence_image_tag", "image_tag")
CACHE_PROBE_IDENTITY_FIELDS = (
    "stable_prefix_sha256",
    "stable_prefix_chars",
    "unique_suffix_pattern",
)
CACHE_PROBE_ENDPOINT_FIELDS = ("base_url", "model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-summary", required=True, type=Path)
    parser.add_argument("--live-aggregate", action="append", type=Path, default=[])
    parser.add_argument("--failure-aggregate", action="append", type=Path, default=[])
    parser.add_argument("--agent-task-summary", action="append", type=Path, default=[])
    parser.add_argument("--cache-aggregate", action="append", type=Path, default=[])
    parser.add_argument("--ref", required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--label", default="branch-image-ga")
    parser.add_argument("--min-live-success-rate", type=float, default=1.0)
    parser.add_argument("--min-failure-success-rate", type=float, default=0.75)
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
        default=list(DEFAULT_REQUIRED_HEADERS),
    )
    parser.add_argument(
        "--min-cached-token-reporting",
        choices=tuple(CACHE_REPORTING_ORDER.keys()),
        default="missing",
    )
    parser.add_argument("--min-cached-token-field-rate", type=float, default=0.0)
    parser.add_argument("--min-cached-prompt-ratio", type=float, default=0.0)
    parser.add_argument("--min-cache-probe-repeats", type=int, default=2)
    parser.add_argument(
        "--require-cache-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require a direct backend baseline in cache-token evidence so the "
            "branch-image artifact records both router and backend paths."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return (
        Path(".agent-harness/experiments/live-agentic-routing/branch-image-ga") / stamp
    )


def load_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.exists():
        return None, f"missing file: {path}"
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON in {path}: {exc}"
    if not isinstance(data, dict):
        return None, f"expected JSON object in {path}"
    return data, str(path)


def rows_from_aggregate(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return [data]


def validation_failures(row: dict[str, Any]) -> list[str]:
    failures = row.get("validation_failures")
    if isinstance(failures, list):
        return [str(item) for item in failures]
    return []


def add_validation_failures(
    failures: list[str], label: str, source: dict[str, Any]
) -> None:
    for item in validation_failures(source):
        failures.append(f"{label} validation_failure: {item}")


def continuity_violations(row: dict[str, Any]) -> int:
    return int(row.get("tool_loop_switch_violations") or 0) + int(
        row.get("context_portability_violations") or 0
    )


def task_name_from_instance(value: str) -> str:
    return value.split(":", 1)[1] if ":" in value else value


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


def first_string_value(
    data: dict[str, Any], row: dict[str, Any] | None, keys: tuple[str, ...]
) -> str:
    for source in (row, data):
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = source.get(key)
            if value is not None and str(value):
                return str(value)
    return ""


def add_evidence_identity_failures(
    failures: list[str],
    args: argparse.Namespace,
    data: dict[str, Any],
    row: dict[str, Any] | None,
    name: str,
) -> None:
    actual_ref = first_string_value(data, row, EVIDENCE_REF_KEYS)
    if args.ref:
        if not actual_ref:
            failures.append(f"{name} evidence_ref missing")
        elif actual_ref != args.ref:
            failures.append(f"{name} evidence_ref {actual_ref} != {args.ref}")

    actual_image_tag = first_string_value(data, row, EVIDENCE_IMAGE_TAG_KEYS)
    if not actual_image_tag:
        failures.append(f"{name} evidence_image_tag missing")
    elif actual_image_tag != args.image_tag:
        failures.append(
            f"{name} evidence_image_tag {actual_image_tag} != {args.image_tag}"
        )


def add_numeric_failure(
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


def cache_probe_repeats(summary: dict[str, Any]) -> int:
    for key in ("probe_repeats", "requests"):
        try:
            return int(summary.get(key) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def add_cache_probe_identity_failures(
    failures: list[str], name: str, summary: dict[str, Any]
) -> None:
    for key in CACHE_PROBE_IDENTITY_FIELDS:
        value = summary.get(key)
        if value is None or value == "":
            failures.append(f"{name} {key} missing")


def add_cache_probe_pair_failures(
    failures: list[str],
    router_summary: dict[str, Any] | None,
    baseline_summary: dict[str, Any] | None,
) -> None:
    if not router_summary or not baseline_summary:
        return
    for key in CACHE_PROBE_IDENTITY_FIELDS:
        router_value = router_summary.get(key)
        baseline_value = baseline_summary.get(key)
        if router_value is None or router_value == "":
            continue
        if baseline_value is None or baseline_value == "":
            continue
        if str(router_value) != str(baseline_value):
            failures.append(
                f"cache router/baseline {key} mismatch: "
                f"{router_value} != {baseline_value}"
            )


def normalized_base_url(summary: dict[str, Any] | None) -> str:
    if not isinstance(summary, dict):
        return ""
    return str(summary.get("base_url") or "").rstrip("/")


def add_cache_probe_endpoint_failures(
    failures: list[str],
    router_summary: dict[str, Any] | None,
    baseline_summary: dict[str, Any] | None,
) -> None:
    for label, summary in (("router", router_summary), ("baseline", baseline_summary)):
        if not isinstance(summary, dict):
            continue
        for field in CACHE_PROBE_ENDPOINT_FIELDS:
            if not str(summary.get(field) or ""):
                failures.append(f"cache {label} {field} missing")
    router_base_url = normalized_base_url(router_summary)
    baseline_base_url = normalized_base_url(baseline_summary)
    if router_base_url and baseline_base_url and router_base_url == baseline_base_url:
        failures.append(
            "cache baseline base_url must differ from router base_url for "
            "direct-backend cache evidence"
        )


def evaluate_diagnostic(
    args: argparse.Namespace, data: dict[str, Any] | None, evidence: str
) -> tuple[dict[str, Any], list[str]]:
    if data is None:
        return {"evidence": evidence}, [evidence]
    failures = validation_failures(data)
    validation_kind = str(data.get("validation_kind") or "")
    if validation_kind != "branch-image-diagnostic-probe":
        failures.append(
            "diagnostic validation_kind "
            f"{validation_kind or 'missing'} != branch-image-diagnostic-probe"
        )
    checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
    for key in ["chat_completion_ok", "diagnostic_headers_ok"]:
        if checks.get(key) is not True:
            failures.append(f"diagnostic {key} is not true")
    for header in data.get("missing_diagnostic_headers") or []:
        failures.append(f"diagnostic missing header: {header}")
    for header in data.get("invalid_diagnostic_headers") or []:
        failures.append(f"diagnostic invalid header: {header}")
    diagnostic_ref = str(data.get("ref") or "")
    if not diagnostic_ref:
        failures.append("diagnostic ref missing")
    elif diagnostic_ref != args.ref:
        failures.append(f"diagnostic ref {diagnostic_ref} != {args.ref}")
    diagnostic_image_tag = str(data.get("image_tag") or "")
    if not diagnostic_image_tag:
        failures.append("diagnostic image_tag missing")
    elif diagnostic_image_tag != args.image_tag:
        failures.append(
            f"diagnostic image_tag {diagnostic_image_tag} != {args.image_tag}"
        )
    return {
        "evidence": evidence,
        "checks": checks,
        "ref": data.get("ref", ""),
        "image_tag": data.get("image_tag", ""),
        "label": data.get("label", ""),
    }, failures


def evaluate_live_aggregate(
    args: argparse.Namespace, data: dict[str, Any] | None, evidence: str
) -> tuple[dict[str, Any], list[str]]:
    if data is None:
        return {"evidence": evidence, "rows": []}, [evidence]
    failures: list[str] = []
    metrics: list[dict[str, Any]] = []
    for row in rows_from_aggregate(data):
        name = str(row.get("name") or row.get("label") or evidence)
        add_evidence_identity_failures(failures, args, data, row, name)
        success_rate = row.get("router_success_rate", row.get("success_rate"))
        add_numeric_failure(
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
        add_validation_failures(failures, name, row)
        metrics.append(
            {
                "name": name,
                "requests": row.get("router_requests", row.get("requests")),
                "success_rate": success_rate,
                "continuity_violations": continuity,
            }
        )
    return {"evidence": evidence, "rows": metrics}, failures


def evaluate_failure_aggregate(
    args: argparse.Namespace, data: dict[str, Any] | None, evidence: str
) -> tuple[dict[str, Any], list[str]]:
    if data is None:
        return {"evidence": evidence, "rows": []}, [evidence]
    failures: list[str] = []
    metrics: list[dict[str, Any]] = []
    for row in rows_from_aggregate(data):
        name = str(row.get("name") or row.get("label") or evidence)
        add_evidence_identity_failures(failures, args, data, row, name)
        add_numeric_failure(
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
        add_validation_failures(failures, name, row)
        injected = int(row.get("injected") or row.get("injected_503") or 0)
        if injected <= 0:
            failures.append(f"{name} injected failures missing")
        sessions = int(row.get("sessions_with_errors") or 0)
        if sessions < args.min_sessions_with_errors:
            failures.append(
                f"{name} sessions_with_errors {sessions} < "
                f"{args.min_sessions_with_errors}"
            )
        add_numeric_failure(
            failures,
            f"{name} session_recovery_rate_after_error",
            row.get("session_recovery_rate_after_error"),
            ">=",
            args.min_session_recovery_rate,
        )
        metrics.append(
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
    return {"evidence": evidence, "rows": metrics}, failures


def evaluate_agent_task_summary(
    args: argparse.Namespace, data: dict[str, Any] | None, evidence: str
) -> tuple[dict[str, Any], list[str]]:
    if data is None:
        return {"evidence": evidence}, [evidence]
    failures: list[str] = []
    add_evidence_identity_failures(failures, args, data, None, "agent task")
    add_numeric_failure(
        failures, "agent task success_rate", data.get("success_rate"), ">=", 1.0
    )
    add_numeric_failure(
        failures,
        "agent task requests",
        data.get("requests"),
        ">=",
        args.min_agent_task_requests,
    )
    add_numeric_failure(
        failures,
        "agent task task_count",
        data.get("task_count", data.get("tasks")),
        ">=",
        args.min_agent_task_count,
    )
    add_numeric_failure(
        failures,
        "agent task task_instances",
        data.get("task_instances"),
        ">=",
        args.min_agent_task_instances,
    )
    add_numeric_failure(
        failures,
        "agent task task_success_rate",
        data.get("task_success_rate"),
        ">=",
        args.min_agent_task_success_rate,
    )
    add_numeric_failure(
        failures,
        "agent task task_score_mean",
        data.get("task_score_mean"),
        ">=",
        args.min_agent_task_score,
    )
    continuity = continuity_violations(data)
    if continuity > args.max_continuity_violations:
        failures.append(
            f"agent task continuity violations {continuity} > "
            f"{args.max_continuity_violations}"
        )
    add_validation_failures(failures, "agent task", data)

    required_headers = list(dict.fromkeys(args.required_agent_task_header))
    missing_counts = data.get("missing_router_header_counts") or {}
    invalid_counts = data.get("invalid_router_header_counts") or {}
    missing_headers = {
        header: missing_counts.get(header, 0)
        for header in required_headers
        if int(missing_counts.get(header, 0) or 0) > 0
    }
    invalid_headers = {
        header: invalid_counts.get(header, 0)
        for header in required_headers
        if int(invalid_counts.get(header, 0) or 0) > 0
    }
    if missing_headers:
        failures.append(f"agent task missing router headers: {missing_headers}")
    if invalid_headers:
        failures.append(f"agent task invalid router headers: {invalid_headers}")

    names = agent_task_names(data)
    missing_names = sorted(set(args.required_agent_task_name or []) - names)
    if missing_names:
        failures.append(f"agent task missing task names: {missing_names}")
    phases = agent_task_phases(data)
    missing_phases = sorted(set(args.required_agent_task_phase or []) - phases)
    if missing_phases:
        failures.append(f"agent task missing phases: {missing_phases}")

    return {
        "evidence": evidence,
        "requests": data.get("requests"),
        "task_count": data.get("task_count", data.get("tasks")),
        "task_instances": data.get("task_instances"),
        "task_names": sorted(names),
        "phase_counts": data.get("phase_counts", {}),
        "continuity_violations": continuity,
    }, failures


def evaluate_cache_aggregate(
    args: argparse.Namespace, data: dict[str, Any] | None, evidence: str
) -> tuple[dict[str, Any], list[str]]:
    if data is None:
        return {"evidence": evidence, "paths": []}, [evidence]
    failures: list[str] = []
    metrics: dict[str, Any] = {"evidence": evidence, "paths": {}}
    add_validation_failures(failures, "cache aggregate", data)
    paths = cache_paths(data)
    path_summaries = dict(paths)
    has_baseline = any(label == "baseline" for label, _ in paths)
    if args.require_cache_baseline and not has_baseline:
        failures.append(
            "cache aggregate direct backend baseline missing; run "
            "cache_token_probe.py with --baseline-base-url"
        )
    required = args.min_cached_token_reporting
    for label, summary in paths:
        name = f"cache {label}"
        add_validation_failures(failures, name, summary)
        if label == "router":
            add_evidence_identity_failures(failures, args, data, summary, name)
        add_numeric_failure(
            failures,
            f"{name} success_rate",
            summary.get("success_rate"),
            ">=",
            args.min_live_success_rate,
        )
        reporting = str(summary.get("cached_token_reporting", "missing"))
        if CACHE_REPORTING_ORDER.get(reporting, -1) < CACHE_REPORTING_ORDER[required]:
            failures.append(f"{name} cached_token_reporting {reporting} < {required}")
        probe_kind = str(summary.get("probe_kind", ""))
        if probe_kind != CACHE_PROBE_KIND:
            failures.append(
                f"{name} probe_kind {probe_kind or 'missing'} != {CACHE_PROBE_KIND}"
            )
        probe_repeats = cache_probe_repeats(summary)
        if probe_repeats < args.min_cache_probe_repeats:
            failures.append(
                f"{name} probe_repeats {probe_repeats} < "
                f"{args.min_cache_probe_repeats}"
            )
        add_cache_probe_identity_failures(failures, name, summary)
        field_rate = cached_token_field_rate(summary)
        if field_rate < args.min_cached_token_field_rate:
            failures.append(
                f"{name} cached_token_field_rate {field_rate} < "
                f"{args.min_cached_token_field_rate}"
            )
        ratio = float(summary.get("cached_prompt_ratio") or 0.0)
        if ratio < args.min_cached_prompt_ratio:
            failures.append(
                f"{name} cached_prompt_ratio {ratio} < "
                f"{args.min_cached_prompt_ratio}"
            )
        metrics["paths"][label] = {
            "base_url": normalized_base_url(summary),
            "model": summary.get("model"),
            "requests": summary.get("requests"),
            "success_rate": summary.get("success_rate"),
            "cached_token_reporting": reporting,
            "cached_token_field_rate": field_rate,
            "cached_prompt_ratio": summary.get("cached_prompt_ratio"),
            "probe_kind": probe_kind,
            "probe_repeats": probe_repeats,
            "stable_prefix_sha256": summary.get("stable_prefix_sha256"),
            "stable_prefix_chars": summary.get("stable_prefix_chars"),
            "unique_suffix_pattern": summary.get("unique_suffix_pattern"),
            "evidence_ref": first_string_value(summary, None, EVIDENCE_REF_KEYS),
            "evidence_image_tag": first_string_value(
                summary, None, EVIDENCE_IMAGE_TAG_KEYS
            ),
        }
    add_cache_probe_pair_failures(
        failures,
        path_summaries.get("router"),
        path_summaries.get("baseline"),
    )
    if args.require_cache_baseline:
        add_cache_probe_endpoint_failures(
            failures,
            path_summaries.get("router"),
            path_summaries.get("baseline"),
        )
    metrics["baseline_required"] = args.require_cache_baseline
    metrics["baseline_present"] = has_baseline
    return metrics, failures


def has_mounted_binary_marker(*values: str) -> bool:
    fingerprint = " ".join(values).lower()
    return any(marker in fingerprint for marker in MOUNTED_BINARY_MARKERS)


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    diagnostic, diagnostic_evidence = load_json(args.diagnostic_summary)
    diagnostic_metrics, failures = evaluate_diagnostic(
        args, diagnostic, diagnostic_evidence
    )
    checks: dict[str, bool] = {}
    checks["diagnostic_ok"] = not failures

    live_metrics = []
    live_failures: list[str] = []
    for path in args.live_aggregate:
        data, evidence = load_json(path)
        metrics, row_failures = evaluate_live_aggregate(args, data, evidence)
        live_metrics.append(metrics)
        live_failures.extend(row_failures)
    if not args.live_aggregate:
        live_failures.append("at least one live aggregate is required")
    checks["live_matrix_ok"] = not live_failures
    failures.extend(live_failures)

    failure_metrics = []
    failure_failures: list[str] = []
    for path in args.failure_aggregate:
        data, evidence = load_json(path)
        metrics, row_failures = evaluate_failure_aggregate(args, data, evidence)
        failure_metrics.append(metrics)
        failure_failures.extend(row_failures)
    if not args.failure_aggregate:
        failure_failures.append("at least one failure aggregate is required")
    checks["failure_recovery_ok"] = not failure_failures
    failures.extend(failure_failures)

    agent_task_metrics = []
    agent_task_failures: list[str] = []
    for path in args.agent_task_summary:
        data, evidence = load_json(path)
        metrics, task_failures = evaluate_agent_task_summary(args, data, evidence)
        agent_task_metrics.append(metrics)
        agent_task_failures.extend(task_failures)
    if not args.agent_task_summary:
        agent_task_failures.append("at least one agent-task summary is required")
    checks["agent_task_ok"] = not agent_task_failures
    failures.extend(agent_task_failures)

    cache_metrics = []
    cache_failures: list[str] = []
    for path in args.cache_aggregate:
        data, evidence = load_json(path)
        metrics, cache_row_failures = evaluate_cache_aggregate(args, data, evidence)
        cache_metrics.append(metrics)
        cache_failures.extend(cache_row_failures)
    if not args.cache_aggregate:
        cache_failures.append("at least one cache aggregate is required")
    checks["cache_token_probe_ok"] = not cache_failures
    failures.extend(cache_failures)

    if has_mounted_binary_marker(args.image_tag, args.label):
        failures.append("mounted-binary evidence cannot satisfy branch-image benchmark")
    checks["mounted_binary_absent"] = not has_mounted_binary_marker(
        args.image_tag, args.label
    )

    failures = list(dict.fromkeys(failures))
    checks["branch_image_benchmark_ok"] = not failures
    return {
        "validation_kind": FULL_BRANCH_IMAGE_KIND,
        "branch_image_benchmark": True,
        "label": args.label,
        "ref": args.ref,
        "image_tag": args.image_tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": checks,
        "evidence": {
            "diagnostic": diagnostic_metrics,
            "live_aggregates": live_metrics,
            "failure_aggregates": failure_metrics,
            "agent_task_summaries": agent_task_metrics,
            "cache_aggregates": cache_metrics,
        },
        "validation_failures": failures,
    }


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "summary.md").write_text(render_markdown(summary))


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Branch Image Session-Routing Benchmark",
        "",
        f"- validation_kind: {summary['validation_kind']}",
        f"- branch_image_benchmark: {summary['branch_image_benchmark']}",
        f"- label: {summary['label']}",
        f"- ref: {summary['ref'] or 'unspecified'}",
        f"- image_tag: {summary['image_tag']}",
        f"- validation_failures: {summary['validation_failures']}",
        "",
        "| Check | Passed |",
        "| --- | --- |",
    ]
    for name, passed in summary["checks"].items():
        lines.append(f"| `{name}` | `{passed}` |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or default_output_dir()
    summary = build_summary(args)
    write_outputs(summary, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "validation_failures": summary["validation_failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if summary["validation_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
