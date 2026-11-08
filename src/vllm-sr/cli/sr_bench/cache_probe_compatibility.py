from __future__ import annotations

import math
from collections import Counter
from http import HTTPStatus
from typing import Any

from .grading import CACHE_OBSERVATION_RESULT
from .transport import CACHE_OBSERVATION_VERSION

VERSION: str = "cache-probe-compatibility-v1"
MAX_HTTP_STATUS: int = 599


def _reported_count(raw: object, keys: tuple[str, ...]) -> int | None:
    if not isinstance(raw, dict):
        return None
    values = [raw[key] for key in keys if key in raw]
    if not values or any(type(value) is not int or value < 0 for value in values):
        return None
    return values[0] if all(value == values[0] for value in values) else None


def _sum_known(values: list[int | None]) -> int | None:
    return (
        sum(value for value in values if value is not None)
        if values and all(value is not None for value in values)
        else None
    )


def _turn(
    *, label: str, repeat: dict[str, Any], call: dict[str, Any]
) -> dict[str, Any]:
    observation = call.get("provider_cache_observation") or {}
    known = observation.get("version") == CACHE_OBSERVATION_VERSION
    qualified = known and observation.get("qualification") in {
        "qualified",
        "incomplete",
    }
    status = call.get("response_status")
    if type(status) is not int or not HTTPStatus.CONTINUE <= status <= MAX_HTTP_STATUS:
        status = None
    elapsed = call.get("latency_s")
    latency = (
        round(elapsed * 1000, 3)
        if type(elapsed) in {int, float} and math.isfinite(elapsed) and elapsed >= 0
        else None
    )
    source = observation.get("source") if known else None
    sources = observation.get("sources", [])
    if source is None and known and len(sources) == 1:
        source = sources[0]
    model = call.get("model")
    return {
        "label": label,
        "repeat_index": repeat["repeat_index"],
        "status": status,
        "success": (
            HTTPStatus.OK <= status < HTTPStatus.MULTIPLE_CHOICES if status else None
        ),
        "latency_ms": latency,
        "prompt_tokens": _reported_count(
            raw=call.get("raw_usage"), keys=("prompt_tokens", "input_tokens")
        ),
        "completion_tokens": _reported_count(
            raw=call.get("raw_usage"), keys=("completion_tokens", "output_tokens")
        ),
        "cached_tokens": (
            observation.get("cached_tokens")
            if qualified and not observation.get("event_issues")
            else None
        ),
        "cached_token_field_present": (
            observation.get("field_present") if known else None
        ),
        "cached_token_source": source,
        "response_model": (
            model
            if call.get("provider_model_observed") is True
            and isinstance(model, str)
            and model
            else None
        ),
        "error": call.get("error", "") if call else None,
        "call_id": repeat["call_id"],
        "transport_status": repeat["status"],
        "output_complete": repeat["output_complete"],
        "selected_model": repeat["selected_model"],
        "session_id": repeat["session_id"],
        "config_hash": repeat["config_hash"],
        "router_phase": repeat["router_phase"],
        "phase_source": repeat["phase_source"],
        "provider_cache_observation": observation if known else None,
    }


def _summary(
    *, metric: dict[str, Any], calls: dict[str, dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    probe = metric["cache_observation"]
    target = probe["target"]
    spec = probe["native_probes"][0]
    label = target["id"]
    turns = [
        _turn(label=label, repeat=repeat, call=calls.get(repeat["call_id"], {}))
        for repeat in probe["repeats"]
    ]
    failures: list[str] = []
    if not probe["execution_complete"]:
        failures.append(f"{label}: repeat sequence is incomplete")
    if any(row["selected_model"] is None for row in turns):
        failures.append(f"{label}: selected model observation is incomplete")
    prompt = _sum_known([row["prompt_tokens"] for row in turns])
    output = _sum_known([row["completion_tokens"] for row in turns])
    ratio = probe["cached_prompt_ratio"]
    if prompt is None or output is None:
        failures.append(f"{label}: reported token usage is incomplete")
        ratio = None
    if any(row["provider_cache_observation"] is None for row in turns):
        failures.append(f"{label}: versioned cache observation is unavailable")
        ratio = None
    if ratio is None:
        failures.append(f"{label}: complete qualified cache usage is unavailable")
    for row in turns:
        observation = row["provider_cache_observation"] or {}
        normalized = observation.get("usage")
        if isinstance(normalized, dict) and row["prompt_tokens"] is not None:
            observed_prompt = sum(
                normalized[key]
                for key in ("input_tokens", "cached_input_tokens", "cache_write_tokens")
            )
            if observed_prompt != row["prompt_tokens"]:
                failures.append(
                    f"{label}: cache and final prompt usage snapshots differ"
                )
                ratio = None
                break
    for name, predicate in probe["validation"]["predicates"].items():
        if predicate["status"] != "pass":
            failures.append(
                f"{label}: {name} {predicate['status']} ({predicate['reason']})"
            )
    if target["kind"] == "mom" and any(
        not target["expected_config_hash"]
        or row["config_hash"] != target["expected_config_hash"]
        for row in turns
    ):
        failures.append(f"{label}: Router configuration acknowledgement is incomplete")
    if (
        any(row["session_id"] is None for row in turns)
        or len({row["session_id"] for row in turns}) != 1
    ):
        failures.append(f"{label}: repeat session evidence is incomplete")
    elapsed = (
        metric.get("case_wall_time_p50_s") if probe["execution_complete"] else None
    )
    throughput = (
        round(len(turns) / elapsed, 3)
        if type(elapsed) in {int, float} and math.isfinite(elapsed) and elapsed > 0
        else None
    )
    return {
        "label": label,
        "base_url": target["base_url"].rstrip("/"),
        "model": target["requested_model"],
        "requests": len(turns),
        "successes": probe["successes"],
        "success_rate": probe["success_rate"],
        "status_counts": probe["http_status_counts"],
        "latency_ms": probe["latency_ms"],
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "cached_tokens": probe["cached_tokens"],
        "cached_prompt_ratio": ratio,
        "cached_token_reporting": probe["cached_token_reporting"],
        "cached_token_field_present": probe["cached_token_field_present"],
        "cached_token_field_rate": probe["cached_token_field_rate"],
        "cached_token_source_counts": dict(
            sorted(
                Counter(
                    row["cached_token_source"]
                    for row in turns
                    if row["success"] and row["cached_token_source"] is not None
                ).items()
            )
        ),
        "probe_kind": spec["probe_kind"],
        "probe_repeats": spec["repeats"],
        "stable_prefix_chars": spec["stable_prefix_chars"],
        "stable_prefix_sha256": spec["native_prefix_sha256"],
        "unique_suffix_pattern": spec["unique_suffix_pattern"],
        "errors": dict(
            sorted(Counter(row["error"] for row in turns if row["error"]).items())
        ),
        "requests_per_second": throughput,
        "sequence_wall_time_s": elapsed,
        "stable_prefix_full_sha256": spec["stable_prefix_sha256"],
        "source_revision": spec["source_revision"],
        "source_sha256": spec["source_sha256"],
        "request_settings": spec["request_settings"],
        "turns": turns,
        "target": target,
        "validation_failures": failures,
        **{
            key: config[key]
            for key in ("evidence_ref", "evidence_image_tag")
            if target["kind"] == "mom" and key in config
        },
    }


def project(
    *,
    run: dict[str, Any],
    benchmarks: list[dict[str, Any]],
    calls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest = run["manifest"]
    metrics = [
        metric
        for metric in benchmarks
        if metric.get("result_kind") == CACHE_OBSERVATION_RESULT
    ]
    result: dict[str, Any] = {
        "version": VERSION,
        "run_id": run["id"],
        "plan_sha256": manifest["plan_sha256"],
        "status": "unsupported",
        "evidence_identity_source": "declared",
        "scope": "response cache observations; not model equivalence or release readiness",
        "validation_failures": [],
    }
    if (
        manifest["mode"] != "live"
        or manifest.get("recovery")
        or len(manifest["cases"]) != 1
        or len(metrics) != len(manifest["targets"])
        or [target["kind"] for target in manifest["targets"]]
        not in (["mom"], ["mom", "single"])
        or any(
            len(metric["cache_observation"]["native_probes"]) != 1 for metric in metrics
        )
    ):
        result["validation_failures"] = [
            "Native cache report projection requires one live Router sequence with an optional direct baseline; mixed, preview and recovery runs are unsupported"
        ]
        return result
    by_target = {metric["target_id"]: metric for metric in metrics}
    for target in manifest["targets"]:
        metric = by_target[target["id"]]
        summary = _summary(
            metric=metric,
            calls=calls,
            config=manifest["benchmark_options"][metric["benchmark"]],
        )
        result["router" if target["kind"] == "mom" else "baseline"] = summary
        result["validation_failures"].extend(summary["validation_failures"])
    if run["status"] != "completed":
        result["validation_failures"].append("Cache run is not completed")
    result["status"] = "incomplete" if result["validation_failures"] else "complete"
    return result
