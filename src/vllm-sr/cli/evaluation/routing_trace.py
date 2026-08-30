"""Normalize Router eval traces into a bounded, content-minimized contract."""

from __future__ import annotations

import re
from typing import Any

from cli.evaluation.canonical import digest_value
from cli.evaluation.evidence import (
    RoutingDecisionTrace,
    RoutingDiagnostic,
    RoutingSignalDiagnostic,
    RoutingTraceNode,
)

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.:/+ -]+$")
_MAX_TRACE_DEPTH = 8
_MAX_TRACE_CHILDREN = 32
_MAX_TRACES = 64
_MAX_SIGNALS = 128


def _safe_token(value: object, *, limit: int = 160) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > limit or "://" in value or "@" in value:
        return None
    return value if _SAFE_TOKEN.fullmatch(value) else None


def _bounded_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if 0 <= number <= 1 else None
    return None


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _trace_node(value: object, depth: int = 0) -> RoutingTraceNode | None:
    if not isinstance(value, dict) or depth >= _MAX_TRACE_DEPTH:
        return None
    children = value.get("children")
    normalized_children = (
        tuple(
            node
            for row in children[:_MAX_TRACE_CHILDREN]
            if (node := _trace_node(row, depth + 1)) is not None
        )
        if isinstance(children, list)
        else ()
    )
    return RoutingTraceNode(
        node_type=_safe_token(value.get("node_type"), limit=64) or "unknown",
        signal_type=_safe_token(value.get("signal_type"), limit=128),
        signal_name=_safe_token(value.get("signal_name"), limit=128),
        label=_safe_token(value.get("label"), limit=128),
        matched=value.get("matched") is True,
        confidence=_bounded_number(value.get("confidence")),
        confidence_scored=value.get("confidence_scored") is True,
        children=normalized_children,
    )


def _decision_traces(payload: dict[str, Any]) -> tuple[RoutingDecisionTrace, ...]:
    rows = payload.get("eval_trace")
    if not isinstance(rows, list):
        return ()
    traces: list[RoutingDecisionTrace] = []
    for row in rows[:_MAX_TRACES]:
        if not isinstance(row, dict):
            continue
        name = _safe_token(row.get("decision_name"), limit=128)
        if not name:
            continue
        traces.append(
            RoutingDecisionTrace(
                decision_name=name,
                matched=row.get("matched") is True,
                confidence=_bounded_number(row.get("confidence")),
                root_trace=_trace_node(row.get("root_trace")),
            )
        )
    return tuple(traces)


def _signals(payload: dict[str, Any]) -> tuple[RoutingSignalDiagnostic, ...]:
    confidence = payload.get("signal_confidences")
    values = payload.get("signal_values")
    errors = payload.get("signal_errors")
    confidence = confidence if isinstance(confidence, dict) else {}
    values = values if isinstance(values, dict) else {}
    errors = errors if isinstance(errors, dict) else {}
    keys = sorted(set(confidence) | set(values) | set(errors))
    rows: list[RoutingSignalDiagnostic] = []
    for raw_key in keys[:_MAX_SIGNALS]:
        key = _safe_token(raw_key)
        if key:
            rows.append(
                RoutingSignalDiagnostic(
                    key=key,
                    confidence=_number(confidence.get(raw_key)),
                    value=_number(values.get(raw_key)),
                    has_error=raw_key in errors,
                )
            )
    return tuple(rows)


def normalize_routing_diagnostic(
    case_id: str, payload: dict[str, Any]
) -> RoutingDiagnostic:
    decision = payload.get("decision_result")
    decision = decision if isinstance(decision, dict) else {}
    plugins = decision.get("plugins")
    recommended = payload.get("recommended_models")
    return RoutingDiagnostic(
        case_id=case_id,
        recipe=_safe_token(payload.get("recipe")),
        decision_name=_safe_token(decision.get("decision_name")),
        algorithm=_safe_token(decision.get("algorithm")),
        plugins=tuple(
            token
            for value in (plugins if isinstance(plugins, list) else [])
            if (token := _safe_token(value)) is not None
        ),
        recommended_models=tuple(
            token
            for value in (recommended if isinstance(recommended, list) else [])
            if (token := _safe_token(value, limit=256)) is not None
        ),
        selected_model=_safe_token(payload.get("selected_model"), limit=256),
        selection_status=_safe_token(payload.get("selection_status"), limit=64),
        selection_method=_safe_token(payload.get("selection_method"), limit=128),
        routing_decision=_safe_token(payload.get("routing_decision")),
        traces=_decision_traces(payload),
        signals=_signals(payload),
    )


def routing_trace_digest(diagnostic: RoutingDiagnostic) -> str:
    return digest_value(diagnostic)
