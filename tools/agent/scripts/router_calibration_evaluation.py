"""Probe request evaluation and exact routing comparisons."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from typing import Any

from router_calibration_http import ensure_success, http_json, normalize_router_url
from router_calibration_probe import (
    MAX_GENERATED_TEXT_BYTES,
    Probe,
    ProbeImageFixture,
    message_content_text_bytes,
)

MAX_REQUEST_BYTES = 10 << 20


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


def _build_request_payload(
    probe: Probe,
    max_request_bytes: int = MAX_REQUEST_BYTES,
) -> dict[str, Any]:
    if probe.messages:
        payload: dict[str, Any] = {"messages": []}
    else:
        payload = {"text": ""}
    if probe.model:
        payload["model"] = probe.model
    if probe.tools:
        payload["tools"] = list(probe.tools)
    if probe.tool_choice is not None:
        payload["tool_choice"] = copy.deepcopy(probe.tool_choice)
    request_value = payload["messages"] if probe.messages else payload["text"]
    value_limit = _request_value_byte_limit(
        probe.probe_id,
        payload,
        request_value,
        max_request_bytes,
    )
    if probe.messages:
        payload["messages"] = materialize_probe_messages(
            probe, max_json_bytes=value_limit
        )
    else:
        payload["text"] = materialize_probe_text(probe, max_json_bytes=value_limit)
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(encoded) > max_request_bytes:
        raise ValueError(
            f"{probe.probe_id} materialized request exceeds {max_request_bytes} bytes"
        )
    return payload


def _request_value_byte_limit(
    probe_id: str,
    payload: dict[str, Any],
    empty_value: Any,
    max_request_bytes: int,
) -> int:
    if max_request_bytes < 1:
        raise ValueError("max_request_bytes must be positive")
    encoded_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    encoded_value = json.dumps(empty_value, ensure_ascii=False).encode("utf-8")
    envelope_bytes = len(encoded_payload) - len(encoded_value)
    value_limit = max_request_bytes - envelope_bytes
    if value_limit < 1:
        raise ValueError(
            f"{probe_id} materialized request exceeds {max_request_bytes} bytes"
        )
    return value_limit


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
    selected_model = str(data.get("selected_model") or "").strip()
    selection_status = str(data.get("selection_status") or "").strip()
    selection_method = str(data.get("selection_method") or "").strip()
    signal_errors = data.get("signal_errors") or {}
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
    selection_comparison = compare_eval_selection(
        algorithm=probe.expected_algorithm,
        selected_model=selected_model,
        status=selection_status,
        method=selection_method,
        recommended_models=actual_models,
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
        "signal_errors": not signal_errors,
        "selection": selection_comparison["matched"],
    }
    return {
        "decision_result": decision_result,
        "actual_decision": actual_decision,
        "actual_models": actual_models,
        "actual_model": actual_model,
        "selected_model": selected_model,
        "selection_status": selection_status,
        "selection_method": selection_method,
        "signal_errors": signal_errors,
        "actual_recipe": actual_recipe,
        "expected_recipe": expected_recipe,
        "actual_algorithm": actual_algorithm,
        "actual_plugins": actual_plugins,
        "signal_comparison": signal_comparison,
        "plugin_comparison": plugin_comparison,
        "trace_comparison": trace_comparison,
        "selection_comparison": selection_comparison,
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
    selection = outcome["selection_comparison"]
    decision_result = outcome["decision_result"]
    return {
        "id": probe.probe_id,
        "decision_id": probe.decision_id,
        "variant_id": probe.variant_id,
        "expected_decision": probe.expected_decision,
        "model": probe.model,
        "actual_model": outcome["actual_model"],
        "selected_model": outcome["selected_model"],
        "selection_status": outcome["selection_status"],
        "selection_method": outcome["selection_method"],
        "signal_errors": outcome["signal_errors"],
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
        "actual_decision": outcome["actual_decision"],
        "matched": outcome["matched"],
        "model_matched": checks["model"],
        "recipe_matched": checks["recipe"],
        "algorithm_matched": checks["algorithm"],
        "plugins_matched": checks["plugins"],
        "signals_matched": checks["signals"],
        "alias_matched": checks["alias"],
        "trace_matched": checks["trace"],
        "signal_errors_matched": checks["signal_errors"],
        "selection_matched": checks["selection"],
        "selection_errors": selection["errors"],
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


def compare_eval_selection(
    *,
    algorithm: str | None,
    selected_model: str,
    status: str,
    method: str,
    recommended_models: tuple[str, ...],
) -> dict[str, Any]:
    """Require an honest final-selection contract when the probe names an algorithm."""
    normalized_algorithm = str(algorithm or "").strip()
    if not normalized_algorithm:
        return {"matched": True, "errors": []}

    expected_statuses = {
        # Router Learning can adapt or protect these base selectors only in the
        # mutating request path. Eval must report that deferral honestly instead
        # of inventing a final model.
        "static": {"selected", "execution_required"},
        "multi_factor": {"selected", "execution_required"},
        "latency_aware": {"selected", "execution_required"},
        # Multi-model algorithms expose a configured final-output model when
        # one exists; otherwise their final model is known only after execution.
        "workflows": {"planned_final", "execution_required"},
        "fusion": {"planned_final", "execution_required"},
        "remom": {"planned_final", "execution_required"},
        "confidence": {"execution_required"},
    }.get(normalized_algorithm)
    errors: list[str] = []
    if not status:
        errors.append("selection_status is missing")
    elif expected_statuses is not None and status not in expected_statuses:
        errors.append(
            f"selection_status={status!r}, want one of "
            f"{sorted(expected_statuses)!r} "
            f"for algorithm {normalized_algorithm!r}"
        )
    if not method:
        errors.append("selection_method is missing")
    elif method != normalized_algorithm and not (
        method == "single" and status == "selected"
    ):
        errors.append(f"selection_method={method!r}, want {normalized_algorithm!r}")
    if status in {"selected", "planned_final"} and not selected_model:
        errors.append(f"selected_model is required for {status}")
    if (
        status == "selected"
        and selected_model
        and selected_model not in recommended_models
    ):
        errors.append("selected_model is not a recommended decision candidate")
    if status == "execution_required" and selected_model:
        errors.append("execution_required must not fabricate selected_model")
    return {"matched": not errors, "errors": errors}


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


def materialize_probe_text(
    probe: Probe,
    max_json_bytes: int = MAX_REQUEST_BYTES,
) -> str:
    """Build adversarial long inputs without duplicating the trigger itself."""
    _validate_materialized_text_size(probe, max_json_bytes)
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


def materialize_probe_messages(
    probe: Probe,
    max_json_bytes: int = MAX_REQUEST_BYTES,
) -> list[dict[str, Any]]:
    """Expand compact text and image fixture references for one request."""
    _validate_materialized_message_size(probe, max_json_bytes)
    messages = copy.deepcopy(list(probe.messages))
    plan = _generated_text_plan(probe, messages)
    if plan is not None:
        content, content_index, character, generated_bytes = plan
        content.insert(
            content_index,
            {"type": "text", "text": character * generated_bytes},
        )
    _materialize_image_fixtures(probe, messages)
    return messages


def _validate_materialized_text_size(probe: Probe, max_json_bytes: int) -> None:
    if max_json_bytes < 1:
        raise ValueError("max_json_bytes must be positive")
    query_bytes = _json_string_content_bytes(probe.query or "") * probe.repeat
    separators = max(probe.repeat - 1, 0) * 2
    total = query_bytes + separators
    if probe.padding is not None:
        total += (
            _json_string_content_bytes(probe.padding.text) * probe.padding.repeat
            + probe.padding.repeat * 2
        )
    if total > max_json_bytes:
        raise ValueError(
            f"{probe.probe_id} materialized text exceeds {max_json_bytes} bytes"
        )


def _json_string_content_bytes(value: str) -> int:
    encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
    return len(encoded) - 2


def _validate_materialized_message_size(probe: Probe, max_json_bytes: int) -> None:
    if max_json_bytes < 1:
        raise ValueError("max_json_bytes must be positive")
    messages = copy.deepcopy(list(probe.messages))
    generated_extra = 0
    plan = _generated_text_plan(probe, messages)
    if plan is not None:
        content, content_index, character, generated_bytes = plan
        content.insert(content_index, {"type": "text", "text": ""})
        generated_extra = _json_string_content_bytes(character) * generated_bytes
    total = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
    total = _bounded_message_size_add(
        probe.probe_id, total, generated_extra, max_json_bytes
    )
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_index, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            raw_type = item.get("type")
            item_type = str(raw_type or "").strip().lower()
            label = (
                f"{probe.probe_id} messages[{message_index}].content"
                f"[{content_index}]"
            )
            if item_type in {"image_url", "input_image"}:
                raise ValueError(
                    f"{label} must use a declared image_fixture instead of an "
                    "inline or remote image"
                )
            if item_type != "image_fixture":
                continue
            if raw_type != "image_fixture":
                raise ValueError(f"{label}.type must be exactly 'image_fixture'")
            fixture = _resolve_image_fixture(probe, item, message_index, content_index)
            compact_bytes = len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
            empty_materialized = {
                "type": "image_url",
                "image_url": {"url": ""},
            }
            empty_bytes = len(
                json.dumps(empty_materialized, ensure_ascii=False).encode("utf-8")
            )
            url_bytes = len(f"data:{fixture.media_type};base64,".encode()) + len(
                fixture.data_base64
            )
            total = _bounded_message_size_add(
                probe.probe_id,
                total,
                empty_bytes - compact_bytes + url_bytes,
                max_json_bytes,
            )


def _bounded_message_size_add(probe_id: str, total: int, extra: int, limit: int) -> int:
    if total > limit or extra > limit - total:
        raise ValueError(f"{probe_id} materialized messages exceed {limit} bytes")
    return total + extra


def _generated_text_plan(
    probe: Probe,
    messages: list[dict[str, Any]],
) -> tuple[list[Any], int, str, int] | None:
    generated = probe.generated_text
    if generated is None:
        return None
    label = f"{probe.probe_id} generated_text"
    if generated.message_index < 0 or generated.message_index >= len(messages):
        raise ValueError(f"{label}.message_index is out of range")
    content = messages[generated.message_index].get("content")
    if not isinstance(content, list):
        raise ValueError(f"{label} requires the selected message content to be a list")
    if generated.content_index < 0 or generated.content_index > len(content):
        raise ValueError(f"{label}.content_index is out of range")
    character = generated.character
    if len(character) != 1 or not character.isascii() or not character.isprintable():
        raise ValueError(f"{label}.character must be one printable ASCII character")
    if (
        generated.target_text_bytes < 1
        or generated.target_text_bytes > MAX_GENERATED_TEXT_BYTES
    ):
        raise ValueError(
            f"{label}.target_text_bytes must be between 1 and "
            f"{MAX_GENERATED_TEXT_BYTES}"
        )
    current_text_bytes = message_content_text_bytes(content)
    generated_bytes = generated.target_text_bytes - current_text_bytes
    if generated_bytes < 1:
        raise ValueError(
            f"{label}.target_text_bytes must exceed the selected message's "
            f"{current_text_bytes} explicit text bytes"
        )
    return content, generated.content_index, character, generated_bytes


def _materialize_image_fixtures(probe: Probe, messages: list[dict[str, Any]]) -> None:
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_index, item in enumerate(content):
            if not isinstance(item, dict) or item.get("type") != "image_fixture":
                continue
            fixture = _resolve_image_fixture(probe, item, message_index, content_index)
            content[content_index] = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{fixture.media_type};base64,{fixture.data_base64}"
                },
            }


def _resolve_image_fixture(
    probe: Probe,
    item: dict[str, Any],
    message_index: int,
    content_index: int,
) -> ProbeImageFixture:
    fixture_name = item.get("fixture")
    label = (
        f"{probe.probe_id} messages[{message_index}].content[{content_index}].fixture"
    )
    if not isinstance(fixture_name, str) or not fixture_name.strip():
        raise TypeError(f"{label} must be a non-empty string")
    fixture_name = fixture_name.strip()
    fixture = probe.image_fixtures.get(fixture_name)
    if fixture is None:
        raise ValueError(f"{label} references unknown image fixture {fixture_name!r}")
    return fixture


def probe_padding_metadata(probe: Probe) -> dict[str, Any] | None:
    if probe.padding is None:
        return None
    return {
        "text": probe.padding.text,
        "repeat": probe.padding.repeat,
        "placement": probe.padding.placement,
    }


def probe_generated_text_metadata(probe: Probe) -> dict[str, Any] | None:
    generated = probe.generated_text
    if generated is None:
        return None
    return {
        "message_index": generated.message_index,
        "content_index": generated.content_index,
        "target_text_bytes": generated.target_text_bytes,
        "character": generated.character,
    }


def probe_materialized_messages_metadata(probe: Probe) -> dict[str, Any] | None:
    """Return a deterministic receipt without serializing expanded fixture data."""
    if probe.generated_text is None and not probe_uses_image_fixture(probe):
        return None
    messages = materialize_probe_messages(probe)
    encoded = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "json_bytes": len(encoded),
        "text_bytes": sum(message_text_bytes(message) for message in messages),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def probe_uses_image_fixture(probe: Probe) -> bool:
    for message in probe.messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(
            isinstance(item, dict) and item.get("type") == "image_fixture"
            for item in content
        ):
            return True
    return False


def message_text_bytes(message: dict[str, Any]) -> int:
    content = message.get("content")
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    if isinstance(content, list):
        return message_content_text_bytes(content)
    return 0


def probe_playground_metadata(probe: Probe) -> dict[str, Any]:
    result: dict[str, Any] = {"enabled": probe.playground.enabled}
    if probe.playground.reason:
        result["reason"] = probe.playground.reason
    return result


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
