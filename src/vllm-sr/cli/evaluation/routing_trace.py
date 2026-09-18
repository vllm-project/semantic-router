"""Normalize Router eval traces into a bounded, content-minimized contract."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from cli.evaluation.canonical import canonical_json_bytes, digest_value
from cli.evaluation.contracts import RunManifest, VisibleCaseSet
from cli.evaluation.evidence import (
    ExecutionRecord,
    RoutingDecisionTrace,
    RoutingDiagnostic,
    RoutingSignalDiagnostic,
    RoutingTraceNode,
)

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.:/+ -]+$")
_ERROR_CHARS = re.compile(r"[^A-Za-z0-9_.:/+ -]")
_MAX_ERROR_LENGTH = 200
_MAX_TRACE_DEPTH = 8
_MAX_TRACE_CHILDREN = 32
_MAX_TRACES = 64
_MAX_SIGNALS = 128
ROUTING_TRACE_MAX_TOKENS = 128
ROUTING_TRACE_MAX_NODES = 256
ROUTING_TRACE_MAX_LINE_BYTES = 256 * 1024


class _RoutingTraceBudget:
    def __init__(self) -> None:
        self.nodes = 0
        self.truncated = False


def _safe_token(value: object, *, limit: int = 160) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > limit or "://" in value or "@" in value:
        return None
    return value if _SAFE_TOKEN.fullmatch(value) else None


def _bounded_error(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    sanitized = _ERROR_CHARS.sub("", value).strip()[:_MAX_ERROR_LENGTH]
    return sanitized or None


def _bounded_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if 0 <= number <= 1 else None
    return None


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _trace_node(
    value: object,
    budget: _RoutingTraceBudget,
    depth: int = 0,
) -> RoutingTraceNode | None:
    if not isinstance(value, dict):
        return None
    if depth >= _MAX_TRACE_DEPTH or budget.nodes >= ROUTING_TRACE_MAX_NODES:
        budget.truncated = True
        return None
    budget.nodes += 1
    children = value.get("children")
    normalized_children: list[RoutingTraceNode] = []
    if isinstance(children, list):
        if len(children) > _MAX_TRACE_CHILDREN:
            budget.truncated = True
        for row in children[:_MAX_TRACE_CHILDREN]:
            node = _trace_node(row, budget, depth + 1)
            if node is not None:
                normalized_children.append(node)
    return RoutingTraceNode(
        node_type=_safe_token(value.get("node_type"), limit=64) or "unknown",
        signal_type=_safe_token(value.get("signal_type"), limit=128),
        signal_name=_safe_token(value.get("signal_name"), limit=128),
        label=_safe_token(value.get("label"), limit=128),
        state=_safe_token(value.get("state"), limit=32),
        matched=value.get("matched") is True,
        confidence=_bounded_number(value.get("confidence")),
        has_signal_error=isinstance(value.get("signal_error"), str)
        and bool(value.get("signal_error")),
        confidence_scored=value.get("confidence_scored") is True,
        children=tuple(normalized_children),
    )


def _decision_traces(
    payload: dict[str, Any], budget: _RoutingTraceBudget
) -> tuple[RoutingDecisionTrace, ...]:
    rows = payload.get("eval_trace")
    if not isinstance(rows, list):
        return ()
    if len(rows) > _MAX_TRACES:
        budget.truncated = True
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
                state=_safe_token(row.get("state"), limit=32),
                matched=row.get("matched") is True,
                confidence=_bounded_number(row.get("confidence")),
                on_unknown=_safe_token(row.get("on_unknown"), limit=32),
                root_trace=_trace_node(row.get("root_trace"), budget),
            )
        )
    return tuple(traces)


def _applied_unknown_policies(
    payload: dict[str, Any], budget: _RoutingTraceBudget
) -> tuple[tuple[str, str], ...]:
    policies = payload.get("applied_unknown_policies")
    if not isinstance(policies, dict):
        return ()
    if len(policies) > _MAX_SIGNALS:
        budget.truncated = True
    rows: list[tuple[str, str]] = []
    for raw_key in sorted(policies)[:_MAX_SIGNALS]:
        key = _safe_token(raw_key, limit=128)
        value = _safe_token(policies.get(raw_key), limit=32)
        if key and value:
            rows.append((key, value))
    return tuple(rows)


def _signals(
    payload: dict[str, Any], budget: _RoutingTraceBudget
) -> tuple[RoutingSignalDiagnostic, ...]:
    confidence = payload.get("signal_confidences")
    values = payload.get("signal_values")
    errors = payload.get("signal_errors")
    confidence = confidence if isinstance(confidence, dict) else {}
    values = values if isinstance(values, dict) else {}
    errors = errors if isinstance(errors, dict) else {}
    keys = sorted(set(confidence) | set(values) | set(errors))
    if len(keys) > _MAX_SIGNALS:
        budget.truncated = True
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


def _safe_tokens(
    value: object,
    budget: _RoutingTraceBudget,
    *,
    limit: int,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    if len(value) > ROUTING_TRACE_MAX_TOKENS:
        budget.truncated = True
    return tuple(
        token
        for item in value[:ROUTING_TRACE_MAX_TOKENS]
        if (token := _safe_token(item, limit=limit)) is not None
    )


def _fit_line_budget(diagnostic: RoutingDiagnostic) -> RoutingDiagnostic:
    if len(canonical_json_bytes(diagnostic)) <= ROUTING_TRACE_MAX_LINE_BYTES:
        return diagnostic

    current = diagnostic.model_copy(update={"truncated": True})
    # Preserve the top-level routing outcome. Remove only bounded diagnostic
    # collections, from least promotion-relevant to most structural, and keep
    # the longest prefix that fits when one collection is sufficient.
    for field in ("signals", "recommended_models", "plugins", "traces"):
        values = getattr(current, field)
        if not values:
            continue
        without = current.model_copy(update={field: ()})
        if len(canonical_json_bytes(without)) > ROUTING_TRACE_MAX_LINE_BYTES:
            current = without
            continue
        low, high = 0, len(values)
        best = without
        while low <= high:
            middle = (low + high) // 2
            candidate = current.model_copy(update={field: values[:middle]})
            if len(canonical_json_bytes(candidate)) <= ROUTING_TRACE_MAX_LINE_BYTES:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best

    if len(canonical_json_bytes(current)) > ROUTING_TRACE_MAX_LINE_BYTES:
        raise ValueError("routing diagnostic identity exceeds its byte budget")
    return current


def normalize_routing_diagnostic(
    case_id: str, payload: dict[str, Any]
) -> RoutingDiagnostic:
    budget = _RoutingTraceBudget()
    decision = payload.get("decision_result")
    decision = decision if isinstance(decision, dict) else {}
    plugins = decision.get("plugins")
    recommended = payload.get("recommended_models")
    diagnostic = RoutingDiagnostic(
        case_id=case_id,
        truncated=False,
        recipe=_safe_token(payload.get("recipe")),
        decision_name=_safe_token(decision.get("decision_name")),
        algorithm=_safe_token(decision.get("algorithm")),
        plugins=_safe_tokens(plugins, budget, limit=160),
        recommended_models=_safe_tokens(recommended, budget, limit=256),
        selected_model=_safe_token(payload.get("selected_model"), limit=256),
        selection_status=_safe_token(payload.get("selection_status"), limit=64),
        selection_method=_safe_token(payload.get("selection_method"), limit=128),
        routing_decision=_safe_token(payload.get("routing_decision")),
        traces=_decision_traces(payload, budget),
        signals=_signals(payload, budget),
        applied_unknown_policies=_applied_unknown_policies(payload, budget),
        decision_error=_bounded_error(payload.get("decision_error")),
    )
    if budget.truncated:
        diagnostic = diagnostic.model_copy(update={"truncated": True})
    return _fit_line_budget(diagnostic)


def routing_trace_digest(diagnostic: RoutingDiagnostic) -> str:
    return digest_value(diagnostic)


def require_routing_trace_binding(
    manifest: RunManifest,
    visible: VisibleCaseSet,
    records: Sequence[ExecutionRecord],
    traces: tuple[RoutingDiagnostic, ...],
) -> None:
    """Bind diagnostic multiplicity to the exact routed record evidence."""

    recorded = Counter(
        (record.case_id, record.trace_digest)
        for record in records
        if record.track_id == "routing" and record.trace_digest is not None
    )
    if traces and (manifest.mode != "live" or "routing" not in manifest.track_ids):
        raise ValueError("routing traces are outside the selected live run")
    visible_case_ids = {case.id for case in visible.cases}
    traced = Counter((trace.case_id, routing_trace_digest(trace)) for trace in traces)
    if (
        any(trace.case_id not in visible_case_ids for trace in traces)
        or traced != recorded
    ):
        raise ValueError("routing traces do not match routed evidence records")
