"""Support helpers for meta-routing feedback replay and calibration scripts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def load_feedback_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    parsed: Any
    if raw.startswith("["):
        parsed = json.loads(raw)
    else:
        parsed = [json.loads(line) for line in raw.splitlines() if line.strip()]

    if not isinstance(parsed, list):
        raise ValueError(f"{path} must contain a JSON array or NDJSON objects")

    records: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if "observation" in item and "action" in item and "outcome" in item:
            records.append(item)
            continue
        request_body = item.get("request_body")
        if isinstance(request_body, str) and request_body.strip():
            nested = json.loads(request_body)
            if (
                isinstance(nested, dict)
                and "observation" in nested
                and "action" in nested
                and "outcome" in nested
            ):
                records.append(nested)
    return records


def flatten_feedback_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for record in records:
        observation = record.get("observation") or {}
        action = record.get("action") or {}
        outcome = record.get("outcome") or {}
        trace = observation.get("trace") or {}
        flattened.append(
            {
                "request_id": observation.get("request_id"),
                "request_model": observation.get("request_model"),
                "mode": record.get("mode"),
                "policy_provider": trace.get("policy_provider") or {},
                "pass_count": trace.get("pass_count", 0),
                "triggers": trace.get("trigger_names") or [],
                "refined_signal_families": trace.get("refined_signal_families") or [],
                "overturned_decision": bool(trace.get("overturned_decision")),
                "latency_delta_ms": trace.get("latency_delta_ms", 0),
                "decision_margin_delta": trace.get("decision_margin_delta", 0.0),
                "projection_boundary_delta": trace.get("projection_boundary_delta"),
                "planned": bool(action.get("planned")),
                "executed": bool(action.get("executed")),
                "executed_pass_count": action.get("executed_pass_count", 0),
                "executed_action_types": action.get("executed_action_types") or [],
                "executed_signal_families": action.get("executed_signal_families")
                or [],
                "final_decision_name": outcome.get("final_decision_name"),
                "final_decision_confidence": outcome.get(
                    "final_decision_confidence", 0.0
                ),
                "final_model": outcome.get("final_model"),
                "response_status": outcome.get("response_status", 0),
                "streaming": bool(outcome.get("streaming")),
                "cache_hit": bool(outcome.get("cache_hit")),
                "pii_blocked": bool(outcome.get("pii_blocked")),
                "hallucination_detected": bool(outcome.get("hallucination_detected")),
                "unverified_factual_response": bool(
                    outcome.get("unverified_factual_response")
                ),
                "response_jailbreak_detected": bool(
                    outcome.get("response_jailbreak_detected")
                ),
                "rag_backend": outcome.get("rag_backend"),
                "router_replay_id": outcome.get("router_replay_id"),
                "user_feedback_signals": outcome.get("user_feedback_signals") or [],
            }
        )
    return flattened


def summarize_feedback_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    flattened = flatten_feedback_records(records)
    total = len(flattened)
    if total == 0:
        return {
            "total_records": 0,
            "executed_rate": 0.0,
            "overturn_rate": 0.0,
            "cache_hit_rate": 0.0,
            "avg_pass_count": 0.0,
            "avg_latency_delta_ms": 0.0,
            "policy_provider_counts": {},
            "trigger_counts": {},
            "action_counts": {},
        }

    trigger_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    policy_provider_counts: dict[str, int] = {}
    executed = 0
    overturned = 0
    cache_hits = 0
    total_passes = 0
    total_latency_delta = 0.0

    for row in flattened:
        total_passes += int(row["pass_count"])
        total_latency_delta += float(row["latency_delta_ms"])
        executed += int(bool(row["executed"]))
        overturned += int(bool(row["overturned_decision"]))
        cache_hits += int(bool(row["cache_hit"]))
        provider = row.get("policy_provider") or {}
        provider_key = (
            f"{provider.get('kind') or 'unknown'}:"
            f"{provider.get('name') or 'unknown'}:"
            f"{provider.get('version') or 'unknown'}"
        )
        policy_provider_counts[provider_key] = (
            policy_provider_counts.get(provider_key, 0) + 1
        )
        for trigger in row["triggers"]:
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        for action in row["executed_action_types"]:
            action_counts[action] = action_counts.get(action, 0) + 1

    return {
        "total_records": total,
        "executed_rate": round((executed / total) * 100, 2),
        "overturn_rate": round((overturned / total) * 100, 2),
        "cache_hit_rate": round((cache_hits / total) * 100, 2),
        "avg_pass_count": round(total_passes / total, 3),
        "avg_latency_delta_ms": round(total_latency_delta / total, 3),
        "policy_provider_counts": dict(sorted(policy_provider_counts.items())),
        "trigger_counts": dict(sorted(trigger_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
    }
