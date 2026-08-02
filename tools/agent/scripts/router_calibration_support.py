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
from urllib import error, request

from router_calibration_manifest import (
    Probe,
    resolve_acceptance,
    summarize_decision_results,
    summarize_tag_results,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SEMANTIC_ROUTER_MODULE_ROOT = REPO_ROOT / "src" / "semantic-router"
DEFAULT_REPORT_ROOT = REPO_ROOT / ".augment" / "router-loop"
HTTP_OK_MIN = 200
HTTP_REDIRECT_MIN = 300
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


def normalize_router_url(router_url: str) -> str:
    normalized = router_url.strip().rstrip("/")
    eval_suffix = "/api/v1/eval"
    normalized = normalized.removesuffix(eval_suffix)
    return normalized


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout_seconds: float = 60.0,
) -> tuple[int, dict[str, Any] | list[Any] | str]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url=url, method=method.upper(), data=body, headers=headers)
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            status = response.getcode()
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return exc.code, parsed
    except (error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"request to {url} failed: {exc}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return status, parsed


def ensure_success(status: int, payload: Any, action: str) -> Any:
    if HTTP_OK_MIN <= status < HTTP_REDIRECT_MIN:
        return payload
    raise RuntimeError(
        f"{action} failed with status {status}: {json.dumps(payload, ensure_ascii=False)}"
    )


def fetch_router_snapshot(router_url: str) -> dict[str, Any]:
    base = normalize_router_url(router_url)
    router_cfg = ensure_success(
        *http_json("GET", f"{base}/config/router"),
        action="GET /config/router",
    )
    versions = ensure_success(
        *http_json("GET", f"{base}/config/router/versions"),
        action="GET /config/router/versions",
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
    `/config/hash` identifies the exact runtime document that was atomically
    published and also detects a later deploy superseding this one.
    """
    base = normalize_router_url(router_url)
    expected = expected_runtime_hash.strip()
    if not expected:
        raise ValueError("deploy response did not include runtime_hash")

    deadline = time.monotonic() + timeout_seconds
    last_status = 0
    last_payload: Any = {"status": "unknown"}
    while time.monotonic() < deadline:
        status, payload = http_json("GET", f"{base}/config/hash")
        last_status = status
        last_payload = payload
        if HTTP_OK_MIN <= status < HTTP_REDIRECT_MIN and isinstance(payload, dict):
            runtime_hash = str(payload.get("runtime_hash") or "").strip()
            active_hash = str(payload.get("active_hash") or "").strip()
            activation_status = str(payload.get("status") or "").strip()
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
    router_url: str, probe: Probe, request_timeout_seconds: float = 60.0
) -> dict[str, Any]:
    request_payload: dict[str, Any]
    if probe.messages:
        request_payload = {"messages": list(probe.messages)}
    else:
        request_payload = {"text": materialize_probe_text(probe)}
    if probe.model:
        request_payload["model"] = probe.model
    if probe.tools:
        request_payload["tools"] = list(probe.tools)
    status, payload = http_json(
        "POST",
        f"{normalize_router_url(router_url)}/api/v1/eval",
        request_payload,
        timeout_seconds=request_timeout_seconds,
    )
    data = ensure_success(status, payload, "POST /api/v1/eval")
    if not isinstance(data, dict):
        raise RuntimeError(
            f"unexpected eval payload for probe {probe.probe_id}: {data!r}"
        )

    decision_result = data.get("decision_result") or {}
    actual_decision = (
        str(data.get("routing_decision") or "").strip()
        or str(decision_result.get("decision_name") or "").strip()
    )
    actual_models = data.get("recommended_models") or []
    actual_model = str(data.get("requested_model") or "").strip()
    actual_recipe = str(data.get("recipe") or "").strip()
    actual_algorithm = str(decision_result.get("algorithm") or "").strip()
    actual_plugins = tuple(
        str(plugin).strip()
        for plugin in decision_result.get("plugins") or []
        if str(plugin).strip()
    )
    matched_signals = decision_result.get("matched_signals") or {}
    missing_expected_signals = find_missing_expected_signals(
        probe.expected_signals, matched_signals
    )
    model_matched = probe.model is None or actual_model == probe.model
    recipe_matched = (
        probe.expected_recipe is None or actual_recipe == probe.expected_recipe
    )
    algorithm_matched = (
        probe.expected_algorithm is None or actual_algorithm == probe.expected_algorithm
    )
    plugins_matched = set(probe.expected_plugins).issubset(actual_plugins)
    signals_matched = not missing_expected_signals
    matched = (
        actual_decision == probe.expected_decision
        and model_matched
        and recipe_matched
        and algorithm_matched
        and plugins_matched
        and signals_matched
    )
    return {
        "id": probe.probe_id,
        "decision_id": probe.decision_id,
        "variant_id": probe.variant_id,
        "expected_decision": probe.expected_decision,
        "model": probe.model,
        "actual_model": actual_model,
        "expected_recipe": probe.expected_recipe,
        "actual_recipe": actual_recipe,
        "expected_algorithm": probe.expected_algorithm,
        "actual_algorithm": actual_algorithm,
        "expected_plugins": list(probe.expected_plugins),
        "actual_plugins": list(actual_plugins),
        "expected_signals": expected_signals_by_type(probe.expected_signals),
        "missing_expected_signals": missing_expected_signals,
        "expected_alias": probe.expected_alias,
        "query": probe.query or summarize_probe_messages(probe.messages),
        "repeat": probe.repeat,
        "padding": probe_padding_metadata(probe),
        "messages": list(probe.messages),
        "tools": list(probe.tools),
        "notes": probe.notes,
        "tags": list(probe.tags),
        "actual_decision": actual_decision,
        "matched": matched,
        "model_matched": model_matched,
        "recipe_matched": recipe_matched,
        "algorithm_matched": algorithm_matched,
        "plugins_matched": plugins_matched,
        "signals_matched": signals_matched,
        "recommended_models": actual_models,
        "used_signals": decision_result.get("used_signals") or {},
        "matched_signals": matched_signals,
        "unmatched_signals": decision_result.get("unmatched_signals") or {},
        "signal_confidences": data.get("signal_confidences") or {},
        "metrics": data.get("metrics") or {},
    }


def find_missing_expected_signals(
    expected: tuple[tuple[str, str], ...], actual: Any
) -> list[str]:
    if not isinstance(actual, dict):
        actual = {}
    actual_by_type = {
        str(signal_type): {str(name) for name in names}
        for signal_type, names in actual.items()
        if isinstance(names, list)
    }
    return [
        f"{signal_type}:{name}"
        for signal_type, name in expected
        if name not in actual_by_type.get(signal_type, set())
    ]


def expected_signals_by_type(
    expected: tuple[tuple[str, str], ...],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for signal_type, name in expected:
        result.setdefault(signal_type, []).append(name)
    return result


def materialize_probe_text(probe: Probe) -> str:
    """Build adversarial long inputs without duplicating the trigger itself."""
    query = "\n".join([probe.query or ""] * probe.repeat)
    if probe.padding is None:
        return query

    padding_lines = [probe.padding.text] * probe.padding.repeat
    if probe.padding.placement == "before":
        parts = [*padding_lines, query]
    elif probe.padding.placement == "after":
        parts = [query, *padding_lines]
    else:
        midpoint = len(padding_lines) // 2
        parts = [*padding_lines[:midpoint], query, *padding_lines[midpoint:]]
    return "\n".join(part for part in parts if part)


def probe_padding_metadata(probe: Probe) -> dict[str, Any] | None:
    if probe.padding is None:
        return None
    return {
        "text": probe.padding.text,
        "repeat": probe.padding.repeat,
        "placement": probe.padding.placement,
    }


def summarize_probe_messages(messages: tuple[dict[str, Any], ...]) -> str:
    if not messages:
        return ""
    for message in reversed(messages):
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        content = summarize_message_content(message.get("content"))
        if content:
            return content
    return json.dumps(list(messages), ensure_ascii=False)


def summarize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        part_type = str(item.get("type") or "").strip().lower()
        if part_type not in ("", "text", "input_text"):
            continue
        text = str(item.get("text") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def evaluate_probes(
    router_url: str, probes: Iterable[Probe], manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    manifest = manifest or {}
    settings = resolve_evaluation_settings(manifest)
    probe_list = list(probes)
    started = time.perf_counter()

    def evaluate_one(probe: Probe) -> dict[str, Any]:
        probe_started = time.perf_counter()
        try:
            result = evaluate_probe(router_url, probe, settings.request_timeout_seconds)
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
        "expected_recipe": probe.expected_recipe,
        "actual_recipe": "",
        "expected_algorithm": probe.expected_algorithm,
        "actual_algorithm": "",
        "expected_plugins": list(probe.expected_plugins),
        "actual_plugins": [],
        "expected_signals": expected_signals_by_type(probe.expected_signals),
        "missing_expected_signals": [
            f"{signal_type}:{name}" for signal_type, name in probe.expected_signals
        ],
        "expected_alias": probe.expected_alias,
        "query": probe.query or summarize_probe_messages(probe.messages),
        "repeat": probe.repeat,
        "padding": probe_padding_metadata(probe),
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


def deploy_config(
    router_url: str, yaml_path: Path, dsl_path: Path | None
) -> dict[str, Any]:
    yaml_path = resolve_repo_path(yaml_path)
    dsl_path = resolve_repo_path(dsl_path)
    payload = {"yaml": yaml_path.read_text(encoding="utf-8")}
    if dsl_path is not None:
        payload["dsl"] = dsl_path.read_text(encoding="utf-8")
    status, response = http_json(
        "PUT",
        f"{normalize_router_url(router_url)}/config/router",
        payload,
    )
    return ensure_success(status, response, "PUT /config/router")


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
