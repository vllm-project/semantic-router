"""Support functions for the router calibration loop CLI."""

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from router_calibration_evaluation import (
    EVALUATION_SCOPES,
    compare_eval_selection,
    compare_eval_trace,
    compare_expected_plugins,
    compare_expected_signals,
    expected_alias_matches,
    expected_signals_by_type,
    find_missing_expected_signals,
    materialize_probe_messages,
    materialize_probe_text,
    probe_generated_text_metadata,
    probe_materialized_messages_metadata,
    probe_padding_metadata,
    probe_playground_metadata,
    summarize_message_content,
    summarize_probe_messages,
)
from router_calibration_evaluation import (
    evaluate_probe as evaluate_probe_request,
)
from router_calibration_http import (
    HTTP_OK_MIN,
    HTTP_REDIRECT_MIN,
    ensure_success,
    http_json,
    normalize_router_url,
)
from router_calibration_manifest import (
    Probe,
    resolve_acceptance,
    summarize_decision_results,
    summarize_tag_results,
)
from router_calibration_signal_values import compare_signal_values

__all__ = [
    "compare_eval_selection",
    "compare_eval_trace",
    "compare_expected_plugins",
    "compare_expected_signals",
    "expected_alias_matches",
    "find_missing_expected_signals",
    "materialize_probe_messages",
    "materialize_probe_text",
    "probe_generated_text_metadata",
    "probe_materialized_messages_metadata",
    "summarize_message_content",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
SEMANTIC_ROUTER_MODULE_ROOT = REPO_ROOT / "src" / "semantic-router"
DEFAULT_REPORT_ROOT = REPO_ROOT / ".augment" / "router-loop"
DEFAULT_EVAL_REQUEST_TIMEOUT_SECONDS = 60.0
MAX_EVAL_REQUEST_TIMEOUT_SECONDS = 1200.0
DEFAULT_EVAL_CONCURRENCY = 1
MAX_EVAL_CONCURRENCY = 64


@dataclass(frozen=True)
class EvaluationSettings:
    request_timeout_seconds: float
    concurrency: int


def resolve_repo_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def fetch_router_snapshot(router_url: str) -> dict[str, Any]:
    base = normalize_router_url(router_url)
    router_cfg = ensure_success(
        *http_json("GET", f"{base}/api/v1/config"),
        action="GET /api/v1/config",
    )
    versions = ensure_success(
        *http_json("GET", f"{base}/api/v1/config/versions"),
        action="GET /api/v1/config/versions",
    )
    ready_status, ready_payload = http_json("GET", f"{base}/ready")
    health_status, health_payload = http_json("GET", f"{base}/health")
    return {
        "router_url": base,
        "captured_at": utc_now(),
        "config_router": router_cfg,
        "config_versions": versions,
        "ready": {"status_code": ready_status, "payload": ready_payload},
        "health": {"status_code": health_status, "payload": health_payload},
    }


def wait_for_router_ready(
    router_url: str, timeout_seconds: float = 300.0, interval_seconds: float = 5.0
) -> dict[str, Any]:
    base = normalize_router_url(router_url)
    deadline = time.monotonic() + timeout_seconds
    last_status = 0
    last_payload: Any = {"status": "unknown", "ready": False}

    while time.monotonic() < deadline:
        status, payload = http_json("GET", f"{base}/ready")
        last_status = status
        last_payload = payload
        if (
            HTTP_OK_MIN <= status < HTTP_REDIRECT_MIN
            and isinstance(payload, dict)
            and bool(payload.get("ready"))
        ):
            return {
                "router_url": base,
                "checked_at": utc_now(),
                "status_code": status,
                "payload": payload,
            }
        time.sleep(max(interval_seconds, 0.1))

    raise RuntimeError(
        "router did not become ready after deploy: "
        f"status={last_status}, payload={json.dumps(last_payload, ensure_ascii=False)}"
    )


def wait_for_config_activation(
    router_url: str,
    expected_runtime_hash: str,
    timeout_seconds: float = 300.0,
    interval_seconds: float = 5.0,
) -> dict[str, Any]:
    """Wait until the immutable runtime matching a deploy is published.

    `/ready` remains true while a replacement router is prepared off-path, so
    readiness alone cannot close the management API's read-after-write gap.
    `/api/v1/config/hash` identifies the exact runtime document that was atomically
    published and also detects a later deploy superseding this one.
    """
    base = normalize_router_url(router_url)
    expected = expected_runtime_hash.strip()
    if not expected:
        raise ValueError("deploy response did not include generated_runtime_hash")

    deadline = time.monotonic() + timeout_seconds
    last_status = 0
    last_payload: Any = {"status": "unknown"}
    while time.monotonic() < deadline:
        status, payload = http_json("GET", f"{base}/api/v1/config/hash")
        last_status = status
        last_payload = payload
        if HTTP_OK_MIN <= status < HTTP_REDIRECT_MIN and isinstance(payload, dict):
            runtime_hash = str(payload.get("generated_runtime_hash") or "").strip()
            active_hash = str(payload.get("active_runtime_hash") or "").strip()
            activation_status = str(payload.get("activation_status") or "").strip()
            if runtime_hash and runtime_hash != expected:
                raise RuntimeError(
                    "config deploy was superseded before activation: "
                    f"expected runtime_hash={expected}, current runtime_hash={runtime_hash}"
                )
            if activation_status == "active" and active_hash == expected:
                return {
                    "router_url": base,
                    "checked_at": utc_now(),
                    "status_code": status,
                    "payload": payload,
                }
        time.sleep(max(interval_seconds, 0.1))

    raise RuntimeError(
        "router config did not become active after deploy: "
        f"expected_runtime_hash={expected}, status={last_status}, "
        f"payload={json.dumps(last_payload, ensure_ascii=False)}"
    )


def evaluate_probe(
    router_url: str,
    probe: Probe,
    request_timeout_seconds: float = 60.0,
    allowed_decisions: frozenset[str] | None = None,
    *,
    scope: str = "deployment",
) -> dict[str, Any]:
    """Evaluate one probe while preserving the patchable transport seam."""
    return evaluate_probe_request(
        router_url,
        probe,
        request_timeout_seconds=request_timeout_seconds,
        allowed_decisions=allowed_decisions,
        http_client=http_json,
        scope=scope,
    )


def evaluate_probes(
    router_url: str,
    probes: Iterable[Probe],
    manifest: dict[str, Any] | None = None,
    selected_probe_ids: Iterable[str] | None = None,
    *,
    scope: str = "deployment",
) -> dict[str, Any]:
    if scope not in EVALUATION_SCOPES:
        raise ValueError(f"unknown evaluation scope {scope!r}")
    manifest = manifest or {}
    settings = resolve_evaluation_settings(manifest)
    all_probes = list(probes)
    decisions_by_recipe: dict[str, set[str]] = {}
    for probe in all_probes:
        recipe_key = probe.expected_recipe or "default"
        decisions_by_recipe.setdefault(recipe_key, set()).add(probe.expected_decision)
    probe_list = select_probes(all_probes, selected_probe_ids)
    started = time.perf_counter()

    def evaluate_one(probe: Probe) -> dict[str, Any]:
        probe_started = time.perf_counter()
        try:
            recipe_key = probe.expected_recipe or "default"
            result = evaluate_probe(
                router_url,
                probe,
                settings.request_timeout_seconds,
                frozenset(decisions_by_recipe[recipe_key]),
                scope=scope,
            )
        except RuntimeError as exc:
            result = failed_probe_result(probe, exc)
            result["evaluation_scope"] = scope
        result["latency_ms"] = round((time.perf_counter() - probe_started) * 1000, 3)
        return result

    if settings.concurrency == 1:
        results = [evaluate_one(probe) for probe in probe_list]
    else:
        with ThreadPoolExecutor(max_workers=settings.concurrency) as executor:
            # executor.map preserves manifest order, so result IDs and failure
            # reports remain deterministic even when requests finish out of order.
            results = list(executor.map(evaluate_one, probe_list))

    wall_time_seconds = time.perf_counter() - started
    decision_summaries = summarize_decision_results(results, manifest)
    tag_summaries = summarize_tag_results(results)
    matched = sum(1 for result in results if result["matched"])
    total = len(results)
    matched_decisions = sum(
        1 for summary in decision_summaries if bool(summary.get("passed"))
    )
    total_decisions = len(decision_summaries)
    acceptance = resolve_acceptance(manifest)
    probe_success_rate = round((matched / total) * 100, 1) if total else 0.0
    decision_success_rate = (
        round((matched_decisions / total_decisions) * 100, 1)
        if total_decisions
        else 0.0
    )
    return {
        "evaluation_scope": scope,
        "scopes": {
            name: summarize_scope_results(results, manifest, name)
            for name in EVALUATION_SCOPES
        },
        "selection_status_counts": dict(
            sorted(
                Counter(
                    result.get("selection_status") or "missing" for result in results
                ).items()
            )
        ),
        "selection_reasons": [
            {
                "id": result["id"],
                "status": result.get("selection_status") or "missing",
                "reason": result.get("selection_reason") or "",
                "error": result.get("error"),
            }
            for result in results
            if result.get("selection_reason") or result.get("error")
        ],
        "router_url": normalize_router_url(router_url),
        "evaluated_at": utc_now(),
        "request_timeout_seconds": settings.request_timeout_seconds,
        "performance": summarize_performance(
            results, settings.concurrency, wall_time_seconds
        ),
        "matched": matched,
        "total": total,
        "success_rate": probe_success_rate,
        "matched_decisions": matched_decisions,
        "total_decisions": total_decisions,
        "decision_success_rate": decision_success_rate,
        "acceptance": acceptance,
        "passed": (
            probe_success_rate >= acceptance["min_probe_pass_rate"]
            and all(summary["passed"] for summary in decision_summaries)
        ),
        "decisions": decision_summaries,
        "tags": tag_summaries,
        "results": results,
    }


def summarize_scope_results(
    results: list[dict[str, Any]],
    manifest: dict[str, Any],
    scope: str,
) -> dict[str, Any]:
    scoped = [
        {**result, "matched": bool(result.get(f"{scope}_matched"))}
        for result in results
    ]
    matched = sum(result["matched"] for result in scoped)
    total = len(scoped)
    success_rate = round(matched / total * 100, 1) if total else 0.0
    decisions = summarize_decision_results(scoped, manifest)
    acceptance = resolve_acceptance(manifest)
    return {
        "matched": matched,
        "total": total,
        "success_rate": success_rate,
        "matched_decisions": sum(bool(item["passed"]) for item in decisions),
        "total_decisions": len(decisions),
        "passed": success_rate >= acceptance["min_probe_pass_rate"]
        and all(item["passed"] for item in decisions),
    }


def select_probes(
    probes: list[Probe], selected_probe_ids: Iterable[str] | None
) -> list[Probe]:
    if selected_probe_ids is None:
        return probes
    requested = [str(probe_id).strip() for probe_id in selected_probe_ids]
    if not requested or any(not probe_id for probe_id in requested):
        raise ValueError("at least one non-empty probe ID is required")
    if len(requested) != len(set(requested)):
        raise ValueError("selected probe IDs must be unique")
    by_id = {probe.probe_id: probe for probe in probes}
    missing = [probe_id for probe_id in requested if probe_id not in by_id]
    if missing:
        raise ValueError(f"unknown probe IDs: {', '.join(missing)}")
    return [by_id[probe_id] for probe_id in requested]


def summarize_performance(
    results: list[dict[str, Any]], concurrency: int, wall_time_seconds: float
) -> dict[str, Any]:
    latencies = sorted(float(result.get("latency_ms") or 0.0) for result in results)
    request_count = len(results)
    return {
        "concurrency": concurrency,
        "requests": request_count,
        "errors": sum(1 for result in results if result.get("error")),
        "wall_time_seconds": round(wall_time_seconds, 3),
        "throughput_rps": round(
            request_count / wall_time_seconds if wall_time_seconds > 0 else 0.0, 3
        ),
        "latency_ms": {
            "min": round(latencies[0], 3) if latencies else 0.0,
            "p50": round(percentile(latencies, 50), 3),
            "p95": round(percentile(latencies, 95), 3),
            "p99": round(percentile(latencies, 99), 3),
            "max": round(latencies[-1], 3) if latencies else 0.0,
        },
    }


def percentile(sorted_values: list[float], percent: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (
        sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def resolve_evaluation_settings(manifest: dict[str, Any]) -> EvaluationSettings:
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        evaluation = {}
    raw_timeout = evaluation.get(
        "request_timeout_seconds", DEFAULT_EVAL_REQUEST_TIMEOUT_SECONDS
    )
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation.request_timeout_seconds must be numeric") from exc
    if timeout < 1 or timeout > MAX_EVAL_REQUEST_TIMEOUT_SECONDS:
        raise ValueError(
            "evaluation.request_timeout_seconds must be between "
            f"1 and {MAX_EVAL_REQUEST_TIMEOUT_SECONDS:g}"
        )

    raw_concurrency = evaluation.get("concurrency", DEFAULT_EVAL_CONCURRENCY)
    if isinstance(raw_concurrency, bool):
        raise TypeError("evaluation.concurrency must be an integer")
    try:
        concurrency = int(raw_concurrency)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation.concurrency must be an integer") from exc
    if concurrency < 1 or concurrency > MAX_EVAL_CONCURRENCY:
        raise ValueError(
            f"evaluation.concurrency must be between 1 and {MAX_EVAL_CONCURRENCY}"
        )
    return EvaluationSettings(timeout, concurrency)


def resolve_eval_request_timeout(manifest: dict[str, Any]) -> float:
    return resolve_evaluation_settings(manifest).request_timeout_seconds


def failed_probe_result(probe: Probe, exc: RuntimeError) -> dict[str, Any]:
    result = {
        "evaluation_scope": "deployment",
        "policy_matched": False,
        "deployment_matched": False,
        "raw_response": getattr(exc, "raw_response", None),
        "http_status": getattr(exc, "http_status", None),
        "id": probe.probe_id,
        "decision_id": probe.decision_id,
        "variant_id": probe.variant_id,
        "expected_decision": probe.expected_decision,
        "model": probe.model,
        "actual_model": "",
        "selected_model": "",
        "selection_status": "",
        "expected_selection_status": probe.expected_selection_status,
        "selection_reason": "",
        "selection_method": "",
        "signal_errors": {},
        "expected_recipe": probe.expected_recipe or "default",
        "actual_recipe": "",
        "expected_algorithm": probe.expected_algorithm,
        "actual_algorithm": "",
        "expected_plugins": list(probe.expected_plugins),
        "forbidden_plugins": list(probe.forbidden_plugins),
        "actual_plugins": [],
        "plugin_match": probe.plugin_match,
        "missing_expected_plugins": list(probe.expected_plugins),
        "unexpected_plugins": [],
        "forbidden_plugin_matches": [],
        "expected_signals": expected_signals_by_type(probe.expected_signals),
        "expected_signal_values": probe.expected_signal_values,
        "observed_signal_values": {},
        "signal_value_errors": [
            "Eval request failed; raw values do not establish success"
        ],
        "forbidden_signals": expected_signals_by_type(probe.forbidden_signals),
        "signal_match": probe.signal_match,
        "missing_expected_signals": [
            f"{signal_type}:{name}" for signal_type, name in probe.expected_signals
        ],
        "unexpected_signals": [],
        "forbidden_signal_matches": [],
        "expected_alias": probe.expected_alias,
        "query": probe.query or summarize_probe_messages(probe.messages),
        "display_prompt": probe.display_prompt,
        "playground": probe_playground_metadata(probe),
        "repeat": probe.repeat,
        "padding": probe_padding_metadata(probe),
        "generated_text": probe_generated_text_metadata(probe),
        "materialized_messages": probe_materialized_messages_metadata(probe),
        "messages": list(probe.messages),
        "tools": list(probe.tools),
        "notes": probe.notes,
        "tags": list(probe.tags),
        "actual_decision": "",
        "matched": False,
        "model_matched": False,
        "recipe_matched": False,
        "algorithm_matched": False,
        "plugins_matched": False,
        "signals_matched": False,
        "signal_values_matched": False,
        "alias_matched": False,
        "trace_matched": False,
        "signal_errors_matched": False,
        "selection_matched": False,
        "selection_structure_matched": False,
        "selection_structure_errors": [
            "Eval request failed; returned selection diagnostics do not establish success"
        ],
        "selection_errors": [
            "Eval request failed; returned selection diagnostics do not establish success"
        ],
        "trace_decisions": [],
        "trace_errors": [str(exc)],
        "recommended_models": [],
        "used_signals": {},
        "matched_signals": {},
        "unmatched_signals": {},
        "signal_confidences": {},
        "metrics": {},
        "error": str(exc),
    }
    result.update(_failed_response_diagnostics(result["raw_response"]))
    payload = result["raw_response"]
    values = payload.get("signal_values") if isinstance(payload, dict) else None
    comparison = compare_signal_values(
        probe.expected_signal_values, values, result["signal_errors"]
    )
    result["observed_signal_values"] = comparison["observed"]
    result["signal_value_errors"].extend(comparison["errors"])
    return result


def _failed_response_diagnostics(payload: Any) -> dict[str, Any]:
    """Retain returned observations without accepting remote evaluation outcomes."""
    if not isinstance(payload, dict):
        return {}
    result: dict[str, Any] = {}
    fields = {
        "requested_model": ("actual_model", str),
        "recipe": ("actual_recipe", str),
        "routing_decision": ("actual_decision", str),
        "selected_model": ("selected_model", str),
        "selection_status": ("selection_status", str),
        "selection_method": ("selection_method", str),
        "selection_reason": ("selection_reason", str),
        "recommended_models": ("recommended_models", list),
        "signal_errors": ("signal_errors", dict),
        "signal_error_matches": ("signal_error_matches", dict),
        "signal_confidences": ("signal_confidences", dict),
        "signal_values": ("signal_values", dict),
        "applied_unknown_policies": ("applied_unknown_policies", dict),
        "decision_error": ("decision_error", str),
        "metrics": ("metrics", dict),
        "eval_trace": ("eval_trace", list),
    }
    for source, (target, value_type) in fields.items():
        value = payload.get(source)
        if isinstance(value, value_type):
            result[target] = copy.deepcopy(value)
    decision = payload.get("decision_result")
    if isinstance(decision, dict):
        for source, target, value_type in (
            ("algorithm", "actual_algorithm", str),
            ("plugins", "actual_plugins", list),
            ("used_signals", "used_signals", dict),
            ("matched_signals", "matched_signals", dict),
            ("unmatched_signals", "unmatched_signals", dict),
        ):
            value = decision.get(source)
            if isinstance(value, value_type):
                result[target] = copy.deepcopy(value)
        name = decision.get("decision_name")
        if not result.get("actual_decision") and isinstance(name, str):
            result["actual_decision"] = name
    return result


def run_validate(dsl_path: Path | None, yaml_path: Path | None) -> dict[str, Any]:
    dsl_path = resolve_repo_path(dsl_path)
    yaml_path = resolve_repo_path(yaml_path)

    if dsl_path is None and yaml_path is None:
        return {"skipped": True, "reason": "no local DSL or YAML asset provided"}

    temp_dsl: Path | None = None
    target_dsl = dsl_path
    repo_cwd = str(SEMANTIC_ROUTER_MODULE_ROOT)

    try:
        if target_dsl is None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".dsl", prefix="router-calibration-", delete=False
            ) as temp_file:
                temp_dsl = Path(temp_file.name)
            decompile_cmd = [
                "go",
                "run",
                "./cmd/dsl",
                "decompile",
                "-o",
                str(temp_dsl),
                str(yaml_path),
            ]
            decompile_run = subprocess.run(
                decompile_cmd,
                cwd=repo_cwd,
                capture_output=True,
                text=True,
                check=False,
            )
            if decompile_run.returncode != 0:
                return {
                    "skipped": False,
                    "valid": False,
                    "mode": "yaml->dsl",
                    "command": decompile_cmd,
                    "returncode": decompile_run.returncode,
                    "stdout": decompile_run.stdout,
                    "stderr": decompile_run.stderr,
                }
            target_dsl = temp_dsl

        validate_cmd = [
            "go",
            "run",
            "./cmd/dsl",
            "validate",
            str(target_dsl),
        ]
        validate_run = subprocess.run(
            validate_cmd,
            cwd=repo_cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "skipped": False,
            "valid": validate_run.returncode == 0,
            "mode": "dsl",
            "command": validate_cmd,
            "returncode": validate_run.returncode,
            "stdout": validate_run.stdout,
            "stderr": validate_run.stderr,
        }
    finally:
        if temp_dsl is not None:
            temp_dsl.unlink(missing_ok=True)


def deploy_config(router_url: str, yaml_path: Path) -> dict[str, Any]:
    yaml_path = resolve_repo_path(yaml_path)
    payload = {"yaml": yaml_path.read_text(encoding="utf-8")}
    base = normalize_router_url(router_url)
    plan_status, plan_response = http_json(
        "POST",
        f"{base}/api/v1/config/plan",
        {**payload, "mode": "replace"},
    )
    plan = ensure_success(plan_status, plan_response, "POST /api/v1/config/plan")
    if not isinstance(plan, dict) or not str(plan.get("current_etag") or "").strip():
        raise RuntimeError("config plan did not return current_etag")
    status, response = http_json(
        "PUT",
        f"{base}/api/v1/config",
        payload,
        if_match=str(plan["current_etag"]),
    )
    return ensure_success(status, response, "PUT /api/v1/config")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_report_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return DEFAULT_REPORT_ROOT / stamp
