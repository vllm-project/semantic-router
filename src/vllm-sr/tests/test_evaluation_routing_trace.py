from __future__ import annotations

from typing import Any

from cli.evaluation.canonical import canonical_json_bytes
from cli.evaluation.evidence import RoutingTraceNode
from cli.evaluation.routing_trace import (
    ROUTING_TRACE_MAX_LINE_BYTES,
    ROUTING_TRACE_MAX_NODES,
    ROUTING_TRACE_MAX_TOKENS,
    normalize_routing_diagnostic,
)


def _node_count(node: RoutingTraceNode | None) -> int:
    if node is None:
        return 0
    return 1 + sum(_node_count(child) for child in node.children)


def _wide_tree() -> dict[str, Any]:
    leaves = [{"node_type": "leaf", "matched": True, "children": []} for _ in range(32)]
    return {
        "node_type": "root",
        "matched": True,
        "children": [
            {"node_type": "branch", "matched": True, "children": leaves}
            for _ in range(32)
        ],
    }


def test_normalizer_deterministically_truncates_all_global_collections() -> None:
    trace_row = {
        "decision_name": "route",
        "matched": True,
        "root_trace": _wide_tree(),
    }
    payload = {
        "decision_result": {
            "decision_name": "route",
            "plugins": [f"plugin-{index:03d}" for index in range(129)],
        },
        "recommended_models": [f"model-{index:03d}" for index in range(129)],
        "eval_trace": [trace_row for _ in range(65)],
        "signal_confidences": {f"signal-{index:03d}": 0.5 for index in range(129)},
    }

    first = normalize_routing_diagnostic("case-1", payload)
    second = normalize_routing_diagnostic("case-1", payload)

    assert first == second
    assert first.truncated is True
    assert len(first.plugins) == ROUTING_TRACE_MAX_TOKENS
    assert len(first.recommended_models) == ROUTING_TRACE_MAX_TOKENS
    assert len(first.traces) == 64
    assert len(first.signals) == 128
    assert sum(_node_count(trace.root_trace) for trace in first.traces) == (
        ROUTING_TRACE_MAX_NODES
    )
    assert len(canonical_json_bytes(first)) <= ROUTING_TRACE_MAX_LINE_BYTES


def test_maximal_untruncated_worker_trace_fits_server_line_budget() -> None:
    token_64 = "n" * 64
    token_128 = "d" * 128
    token_160 = "p" * 160
    token_256 = "m" * 256
    leaf = {
        "node_type": token_64,
        "signal_type": token_128,
        "signal_name": token_128,
        "label": token_128,
        "matched": True,
        "confidence": 1,
        "confidence_scored": True,
        "children": [],
    }
    traces = [
        {
            "decision_name": token_128,
            "matched": True,
            "confidence": 1,
            "root_trace": {**leaf, "children": [leaf, leaf, leaf]},
        }
        for _ in range(64)
    ]
    signal_keys = [f"{index:03d}" + "s" * 157 for index in range(128)]
    payload = {
        "recipe": token_160,
        "decision_result": {
            "decision_name": token_160,
            "algorithm": token_160,
            "plugins": [token_160 for _ in range(ROUTING_TRACE_MAX_TOKENS)],
        },
        "recommended_models": [token_256 for _ in range(ROUTING_TRACE_MAX_TOKENS)],
        "selected_model": token_256,
        "selection_status": token_64,
        "selection_method": token_128,
        "routing_decision": token_160,
        "eval_trace": traces,
        "signal_confidences": dict.fromkeys(signal_keys, 1.7976931348623157e308),
        "signal_values": dict.fromkeys(signal_keys, -1.7976931348623157e308),
    }

    diagnostic = normalize_routing_diagnostic("case-1", payload)

    assert diagnostic.truncated is False
    assert sum(_node_count(trace.root_trace) for trace in diagnostic.traces) == (
        ROUTING_TRACE_MAX_NODES
    )
    assert len(canonical_json_bytes(diagnostic)) <= ROUTING_TRACE_MAX_LINE_BYTES
