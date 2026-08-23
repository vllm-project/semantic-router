"""Support functions for the router calibration loop CLI."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from router_calibration_evaluation import (
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
from router_calibration_http import http_json, normalize_router_url
from router_calibration_manifest import (
    Probe,
    resolve_acceptance,
    summarize_decision_results,
    summarize_tag_results,
)

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


def evaluate_probe(
    router_url: str,
    probe: Probe,
    request_timeout_seconds: float = 60.0,
    allowed_decisions: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Evaluate one probe while preserving the patchable transport seam."""
    return evaluate_probe_request(
        router_url,
        probe,
        request_timeout_seconds=request_timeout_seconds,
        allowed_decisions=allowed_decisions,
        http_client=http_json,
    )


def evaluate_probes(
    router_url: str,
    probes: Iterable[Probe],
    manifest: dict[str, Any] | None = None,
    selected_probe_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
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
            )
        except RuntimeError as exc:
            result = failed_probe_result(probe, exc)
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
    return {
        "id": probe.probe_id,
        "decision_id": probe.decision_id,
        "variant_id": probe.variant_id,
        "expected_decision": probe.expected_decision,
        "model": probe.model,
        "actual_model": "",
        "selected_model": "",
        "selection_status": "",
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
        "alias_matched": False,
        "trace_matched": False,
        "signal_errors_matched": False,
        "selection_matched": False,
        "selection_errors": ["Eval request failed before model selection"],
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
