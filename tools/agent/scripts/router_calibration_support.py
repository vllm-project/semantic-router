"""Support functions for the router calibration loop CLI."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from router_calibration_manifest import (
    Probe,
    resolve_acceptance,
    summarize_decision_results,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SEMANTIC_ROUTER_MODULE_ROOT = REPO_ROOT / "src" / "semantic-router"
DEFAULT_REPORT_ROOT = REPO_ROOT / ".augment" / "router-loop"
HTTP_OK_MIN = 200
HTTP_REDIRECT_MIN = 300


def normalize_router_url(router_url: str) -> str:
    normalized = router_url.strip().rstrip("/")
    eval_suffix = "/api/v1/eval"
    if normalized.endswith(eval_suffix):
        normalized = normalized[: -len(eval_suffix)]
    return normalized


def http_json(
    method: str, url: str, payload: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any] | list[Any] | str]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url=url, method=method.upper(), data=body, headers=headers)
    try:
        with request.urlopen(req, timeout=60) as response:
            status = response.getcode()
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return exc.code, parsed
    except error.URLError as exc:
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


def _try_get_json(base: str, path: str) -> dict[str, Any] | None:
    """Attempt a GET and return the parsed body, or None on non-2xx."""
    status, payload = http_json("GET", f"{base}{path}")
    if HTTP_OK_MIN <= status < HTTP_REDIRECT_MIN:
        return payload if isinstance(payload, dict) else {"_raw": payload}
    return None


def fetch_router_snapshot(router_url: str) -> dict[str, Any]:
    base = normalize_router_url(router_url)
    ready_status, ready_payload = http_json("GET", f"{base}/ready")
    health_status, health_payload = http_json("GET", f"{base}/health")
    return {
        "router_url": base,
        "captured_at": utc_now(),
        "config_router": _try_get_json(base, "/config/router"),
        "config_versions": _try_get_json(base, "/config/versions"),
        "config_classification": _try_get_json(base, "/config/classification"),
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


def refresh_runtime_classification(router_url: str) -> dict[str, Any]:
    base = normalize_router_url(router_url)
    current_cfg = _try_get_json(base, "/config/classification")
    if current_cfg is None:
        return {
            "router_url": base,
            "refreshed_at": utc_now(),
            "skipped": True,
            "reason": "GET /config/classification not available",
        }
    status, updated_cfg = http_json("PUT", f"{base}/config/classification", current_cfg)
    if not (HTTP_OK_MIN <= status < HTTP_REDIRECT_MIN):
        return {
            "router_url": base,
            "refreshed_at": utc_now(),
            "skipped": True,
            "reason": f"PUT /config/classification returned {status}",
        }
    return {
        "router_url": base,
        "refreshed_at": utc_now(),
        "config_classification": updated_cfg,
    }


def _count_signals(signals: dict[str, Any]) -> int:
    return sum(
        len(v) if isinstance(v, list) else (1 if v else 0) for v in signals.values()
    )


def compute_trace_quality(eval_data: dict[str, Any]) -> dict[str, Any]:
    """Compute trace quality metrics from an eval response.

    Returns signal_dominance, avg_confidence, and an overall
    trace_quality score in [0, 1].
    """
    confidences = eval_data.get("signal_confidences") or {}
    decision_result = eval_data.get("decision_result") or {}
    matched_signals = decision_result.get("matched_signals") or {}
    used_signals = decision_result.get("used_signals") or {}
    metrics = eval_data.get("metrics") or {}

    used_count = _count_signals(used_signals)
    matched_count = _count_signals(matched_signals)
    signal_dominance = matched_count / max(used_count, 1)

    matched_confidences: list[float] = []
    for signal_type, names in matched_signals.items():
        if isinstance(names, list):
            for name in names:
                key = f"{signal_type}:{name}"
                if key in confidences:
                    matched_confidences.append(confidences[key])

    # Fallback: when signal_confidences is empty (e.g. non-ML signals like
    # context), use the per-signal-type confidence from the metrics map.
    if not matched_confidences:
        for signal_type in matched_signals:
            type_metrics = metrics.get(signal_type)
            if isinstance(type_metrics, dict):
                conf = type_metrics.get("confidence", 0)
                if isinstance(conf, (int, float)) and conf > 0:
                    matched_confidences.append(float(conf))

    avg_confidence = (
        sum(matched_confidences) / len(matched_confidences)
        if matched_confidences
        else 0.0
    )

    trace_quality = (signal_dominance * max(avg_confidence, 0.0)) ** 0.5

    return {
        "signal_dominance": round(signal_dominance, 4),
        "avg_confidence": round(avg_confidence, 4),
        "matched_signal_count": matched_count,
        "used_signal_count": used_count,
        "trace_quality": round(trace_quality, 4),
    }


def classify_failure(probe_result: dict[str, Any]) -> dict[str, Any]:
    """Classify a probe failure into root-cause buckets.

    Uses the eval trace to determine whether the failure is:
    - query_quality: probe doesn't trigger the right signals
    - routing_design: signals fire but decision doesn't match
    - signal_overlap: multiple decisions compete, wrong one wins
    - confidence_gap: right signals fire but below threshold
    """
    matched_sigs = probe_result.get("matched_signals") or {}
    used_sigs = probe_result.get("used_signals") or {}
    unmatched_sigs = probe_result.get("unmatched_signals") or {}
    confidences = probe_result.get("signal_confidences") or {}

    matched_count = _count_signals(matched_sigs)
    used_count = _count_signals(used_sigs)
    unmatched_count = _count_signals(unmatched_sigs)

    if matched_count == 0:
        return {"root_cause": "query_quality", "detail": "no expected signals matched"}

    if matched_count > 0 and unmatched_count > 0:
        low_conf = [
            k
            for k, v in confidences.items()
            if v < 0.5
            and any(
                k.endswith(f":{n}")
                for names in used_sigs.values()
                if isinstance(names, list)
                for n in names
            )
        ]
        if low_conf:
            return {
                "root_cause": "confidence_gap",
                "detail": f"signals below threshold: {low_conf}",
            }
        return {
            "root_cause": "routing_design",
            "detail": f"{unmatched_count}/{used_count} required signals did not match",
        }

    return {
        "root_cause": "signal_overlap",
        "detail": "all signals matched but a competing decision won",
    }


def evaluate_probe(router_url: str, probe: Probe) -> dict[str, Any]:
    status, payload = http_json(
        "POST",
        f"{normalize_router_url(router_url)}/api/v1/eval",
        {"text": probe.query},
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
    matched = actual_decision == probe.expected_decision

    trace_quality = compute_trace_quality(data)
    root_cause_classification = (
        classify_failure(
            {
                "matched_signals": decision_result.get("matched_signals") or {},
                "used_signals": decision_result.get("used_signals") or {},
                "unmatched_signals": decision_result.get("unmatched_signals") or {},
                "signal_confidences": data.get("signal_confidences") or {},
            }
        )
        if not matched
        else {}
    )

    return {
        "id": probe.probe_id,
        "decision_id": probe.decision_id,
        "variant_id": probe.variant_id,
        "expected_decision": probe.expected_decision,
        "expected_alias": probe.expected_alias,
        "query": probe.query,
        "notes": probe.notes,
        "tags": list(probe.tags),
        "actual_decision": actual_decision,
        "matched": matched,
        "recommended_models": actual_models,
        "used_signals": decision_result.get("used_signals") or {},
        "matched_signals": decision_result.get("matched_signals") or {},
        "unmatched_signals": decision_result.get("unmatched_signals") or {},
        "signal_confidences": data.get("signal_confidences") or {},
        "metrics": data.get("metrics") or {},
        "trace_quality": trace_quality,
        "root_cause_classification": root_cause_classification,
    }


def evaluate_probes(
    router_url: str, probes: Iterable[Probe], manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    results = [evaluate_probe(router_url, probe) for probe in probes]
    decision_summaries = summarize_decision_results(results, manifest or {})
    matched = sum(1 for result in results if result["matched"])
    total = len(results)
    matched_decisions = sum(
        1 for summary in decision_summaries if bool(summary.get("passed"))
    )
    total_decisions = len(decision_summaries)
    acceptance = resolve_acceptance(manifest or {})
    probe_success_rate = round((matched / total) * 100, 1) if total else 0.0
    decision_success_rate = (
        round((matched_decisions / total_decisions) * 100, 1)
        if total_decisions
        else 0.0
    )

    trace_qualities = [
        r.get("trace_quality", {}).get("trace_quality", 0) for r in results
    ]
    avg_trace_quality = (
        sum(trace_qualities) / len(trace_qualities) if trace_qualities else 0.0
    )
    hybrid_reward = probe_success_rate * (1 + avg_trace_quality * 100) / 200

    fragile_matches = [
        r
        for r in results
        if r["matched"] and r.get("trace_quality", {}).get("trace_quality", 0) < 0.6
    ]

    return {
        "router_url": normalize_router_url(router_url),
        "evaluated_at": utc_now(),
        "matched": matched,
        "total": total,
        "success_rate": probe_success_rate,
        "matched_decisions": matched_decisions,
        "total_decisions": total_decisions,
        "decision_success_rate": decision_success_rate,
        "avg_trace_quality": round(avg_trace_quality, 4),
        "hybrid_reward": round(hybrid_reward, 4),
        "fragile_match_count": len(fragile_matches),
        "fragile_matches": [r["id"] for r in fragile_matches],
        "acceptance": acceptance,
        "passed": (
            probe_success_rate >= acceptance["min_probe_pass_rate"]
            and all(summary["passed"] for summary in decision_summaries)
        ),
        "decisions": decision_summaries,
        "results": results,
    }


def run_validate(dsl_path: Path | None, yaml_path: Path | None) -> dict[str, Any]:
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
    payload = {"yaml": yaml_path.read_text(encoding="utf-8")}
    if dsl_path is not None:
        payload["dsl"] = dsl_path.read_text(encoding="utf-8")
    status, response = http_json(
        "POST",
        f"{normalize_router_url(router_url)}/config/deploy",
        payload,
    )
    return ensure_success(status, response, "POST /config/deploy")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_report_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_REPORT_ROOT / stamp
