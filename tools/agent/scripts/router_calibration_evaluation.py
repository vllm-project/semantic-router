"""Probe request evaluation and exact routing comparisons."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from router_calibration_http import ensure_success, http_json, normalize_router_url
from router_calibration_probe import Probe


def evaluate_probe(
    router_url: str,
    probe: Probe,
    request_timeout_seconds: float = 60.0,
    allowed_decisions: frozenset[str] | None = None,
    http_client: Callable[..., tuple[int, Any]] = http_json,
) -> dict[str, Any]:
    status, payload = http_client(
        "POST",
        f"{normalize_router_url(router_url)}/api/v1/eval?trace=true",
        _build_request_payload(probe),
        timeout_seconds=request_timeout_seconds,
    )
    data = ensure_success(status, payload, "POST /api/v1/eval")
    if not isinstance(data, dict):
        raise RuntimeError(
            f"unexpected eval payload for probe {probe.probe_id}: {data!r}"
        )
    outcome = _compare_probe_outcome(data, probe, allowed_decisions)
    return _build_probe_result(data, probe, outcome)


def _build_request_payload(probe: Probe) -> dict[str, Any]:
    if probe.messages:
        payload: dict[str, Any] = {"messages": list(probe.messages)}
    else:
        payload = {"text": materialize_probe_text(probe)}
    if probe.model:
        payload["model"] = probe.model
    if probe.tools:
        payload["tools"] = list(probe.tools)
    return payload


def _compare_probe_outcome(
    data: dict[str, Any],
    probe: Probe,
    allowed_decisions: frozenset[str] | None,
) -> dict[str, Any]:
    decision_result = data.get("decision_result") or {}
    actual_decision = (
        str(data.get("routing_decision") or "").strip()
        or str(decision_result.get("decision_name") or "").strip()
    )
    actual_models = tuple(
        str(model).strip()
        for model in data.get("recommended_models") or []
        if str(model).strip()
    )
    actual_model = str(data.get("requested_model") or "").strip()
    actual_recipe = str(data.get("recipe") or "").strip()
    expected_recipe = probe.expected_recipe or "default"
    actual_algorithm = str(decision_result.get("algorithm") or "").strip()
    actual_plugins = tuple(
        str(plugin).strip()
        for plugin in decision_result.get("plugins") or []
        if str(plugin).strip()
    )
    signal_comparison = compare_expected_signals(
        expected=probe.expected_signals,
        forbidden=probe.forbidden_signals,
        actual=decision_result.get("matched_signals") or {},
        match_mode=probe.signal_match,
    )
    plugin_comparison = compare_expected_plugins(
        expected=probe.expected_plugins,
        forbidden=probe.forbidden_plugins,
        actual=actual_plugins,
        match_mode=probe.plugin_match,
    )
    trace_comparison = compare_eval_trace(
        data.get("eval_trace"),
        expected_decision=probe.expected_decision,
        allowed_decisions=allowed_decisions,
    )
    checks = {
        "decision": actual_decision == probe.expected_decision,
        "model": probe.model is None or actual_model == probe.model,
        "recipe": actual_recipe == expected_recipe,
        "algorithm": (
            probe.expected_algorithm is None
            or actual_algorithm == probe.expected_algorithm
        ),
        "plugins": plugin_comparison["matched"],
        "signals": signal_comparison["matched"],
        "alias": expected_alias_matches(probe.expected_alias, actual_models),
        "trace": trace_comparison["matched"],
    }
    return {
        "decision_result": decision_result,
        "actual_decision": actual_decision,
        "actual_models": actual_models,
        "actual_model": actual_model,
        "actual_recipe": actual_recipe,
        "expected_recipe": expected_recipe,
        "actual_algorithm": actual_algorithm,
        "actual_plugins": actual_plugins,
        "signal_comparison": signal_comparison,
        "plugin_comparison": plugin_comparison,
        "trace_comparison": trace_comparison,
        "checks": checks,
        "matched": all(checks.values()),
    }


def _build_probe_result(
    data: dict[str, Any], probe: Probe, outcome: dict[str, Any]
) -> dict[str, Any]:
    checks = outcome["checks"]
    signals = outcome["signal_comparison"]
    plugins = outcome["plugin_comparison"]
    trace = outcome["trace_comparison"]
    decision_result = outcome["decision_result"]
    return {
        "id": probe.probe_id,
        "decision_id": probe.decision_id,
        "variant_id": probe.variant_id,
        "expected_decision": probe.expected_decision,
        "model": probe.model,
        "actual_model": outcome["actual_model"],
        "expected_recipe": outcome["expected_recipe"],
        "actual_recipe": outcome["actual_recipe"],
        "expected_algorithm": probe.expected_algorithm,
        "actual_algorithm": outcome["actual_algorithm"],
        "expected_plugins": list(probe.expected_plugins),
        "forbidden_plugins": list(probe.forbidden_plugins),
        "actual_plugins": list(outcome["actual_plugins"]),
        "plugin_match": probe.plugin_match,
        "missing_expected_plugins": plugins["missing"],
        "unexpected_plugins": plugins["unexpected"],
        "forbidden_plugin_matches": plugins["forbidden"],
        "expected_signals": expected_signals_by_type(probe.expected_signals),
        "forbidden_signals": expected_signals_by_type(probe.forbidden_signals),
        "signal_match": probe.signal_match,
        "missing_expected_signals": signals["missing"],
        "unexpected_signals": signals["unexpected"],
        "forbidden_signal_matches": signals["forbidden"],
        "expected_alias": probe.expected_alias,
        "query": probe.query or summarize_probe_messages(probe.messages),
        "repeat": probe.repeat,
        "padding": probe_padding_metadata(probe),
        "messages": list(probe.messages),
        "tools": list(probe.tools),
        "notes": probe.notes,
        "tags": list(probe.tags),
        "actual_decision": outcome["actual_decision"],
        "matched": outcome["matched"],
        "model_matched": checks["model"],
        "recipe_matched": checks["recipe"],
        "algorithm_matched": checks["algorithm"],
        "plugins_matched": checks["plugins"],
        "signals_matched": checks["signals"],
        "alias_matched": checks["alias"],
        "trace_matched": checks["trace"],
        "trace_decisions": trace["decisions"],
        "trace_errors": trace["errors"],
        "recommended_models": list(outcome["actual_models"]),
        "used_signals": decision_result.get("used_signals") or {},
        "matched_signals": decision_result.get("matched_signals") or {},
        "unmatched_signals": decision_result.get("unmatched_signals") or {},
        "signal_confidences": data.get("signal_confidences") or {},
        "metrics": data.get("metrics") or {},
    }


def compare_expected_signals(
    *,
    expected: tuple[tuple[str, str], ...],
    forbidden: tuple[tuple[str, str], ...],
    actual: Any,
    match_mode: str,
) -> dict[str, Any]:
    if not isinstance(actual, dict):
        actual = {}
    actual_pairs = {
        (str(signal_type), str(name))
        for signal_type, names in actual.items()
        if isinstance(names, list)
        for name in names
    }
    expected_pairs = set(expected)
    forbidden_pairs = set(forbidden)
    missing = sorted(
        f"{signal_type}:{name}" for signal_type, name in expected_pairs - actual_pairs
    )
    unexpected = (
        sorted(
            f"{signal_type}:{name}"
            for signal_type, name in actual_pairs - expected_pairs
        )
        if match_mode == "exact"
        else []
    )
    forbidden_matches = sorted(
        f"{signal_type}:{name}" for signal_type, name in forbidden_pairs & actual_pairs
    )
    return {
        "matched": not missing and not unexpected and not forbidden_matches,
        "missing": missing,
        "unexpected": unexpected,
        "forbidden": forbidden_matches,
    }


def find_missing_expected_signals(
    expected: tuple[tuple[str, str], ...], actual: Any
) -> list[str]:
    """Compatibility helper for callers that only need subset matching."""
    return compare_expected_signals(
        expected=expected,
        forbidden=(),
        actual=actual,
        match_mode="contains",
    )["missing"]


def compare_expected_plugins(
    *,
    expected: tuple[str, ...],
    forbidden: tuple[str, ...],
    actual: tuple[str, ...],
    match_mode: str,
) -> dict[str, Any]:
    expected_set = set(expected)
    actual_set = set(actual)
    forbidden_set = set(forbidden)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set) if match_mode == "exact" else []
    forbidden_matches = sorted(forbidden_set & actual_set)
    return {
        "matched": not missing and not unexpected and not forbidden_matches,
        "missing": missing,
        "unexpected": unexpected,
        "forbidden": forbidden_matches,
    }


def expected_alias_matches(
    expected_alias: str | None, actual_models: tuple[str, ...]
) -> bool:
    if expected_alias is None:
        return True
    if len(actual_models) == 1:
        return actual_models[0] == expected_alias
    return expected_alias in actual_models


def compare_eval_trace(
    raw_trace: Any,
    *,
    expected_decision: str,
    allowed_decisions: frozenset[str] | None,
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(raw_trace, list) or not raw_trace:
        return {
            "matched": False,
            "decisions": [],
            "errors": ["eval_trace is missing or empty"],
        }
    decision_names: list[str] = []
    matching_expected = 0
    for index, trace in enumerate(raw_trace):
        if not isinstance(trace, dict):
            errors.append(f"eval_trace[{index}] is not an object")
            continue
        name = str(trace.get("decision_name") or "").strip()
        if not name:
            errors.append(f"eval_trace[{index}] has no decision_name")
            continue
        decision_names.append(name)
        if name == expected_decision and bool(trace.get("matched")):
            matching_expected += 1
    if len(decision_names) != len(set(decision_names)):
        errors.append("eval_trace contains duplicate decision names")
    if matching_expected != 1:
        errors.append(
            f"expected exactly one matched trace for {expected_decision!r}, "
            f"got {matching_expected}"
        )
    if allowed_decisions is not None and set(decision_names) != set(allowed_decisions):
        errors.append(
            "eval_trace decision set differs from the selected recipe: "
            f"got={sorted(set(decision_names))}, want={sorted(allowed_decisions)}"
        )
    return {
        "matched": not errors,
        "decisions": decision_names,
        "errors": errors,
    }


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
