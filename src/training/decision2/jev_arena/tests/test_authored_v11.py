"""Rule and editorial-gate checks for the DEV-only authored v11 builder."""

from __future__ import annotations

import pytest

from jev_arena.authored_v11_pilot import (
    OPS,
    build_item,
    evaluate,
    lexical_gate,
    reference,
)

CASES = {
    "temporal-bulletin": (
        {
            "issued": {"Ada": 1, "Bela": 2, "Cato": 3},
            "valid_until": {"Ada": 4, "Bela": 4, "Cato": 4},
            "decision_day": 3,
        },
        "Cato",
    ),
    "shared-interval": (
        {
            "slots": {"Ada": [1, 2], "Bela": [3, 4], "Cato": [5, 6]},
            "client_window": [2, 6],
            "guide_window": [3, 6],
        },
        "Bela",
    ),
    "service-coverage": (
        {
            "coverage": {"Ada": ["x"], "Bela": ["x", "y"], "Cato": ["y"]},
            "required": ["x", "y"],
            "prices": {"Ada": 2, "Bela": 3, "Cato": 4},
            "cap": 3,
        },
        "Bela",
    ),
    "scenario-minimax": (
        {
            "loss_red": {"Ada": 2, "Bela": 3, "Cato": 5},
            "loss_blue": {"Ada": 6, "Bela": 4, "Cato": 1},
            "spend": {"Ada": 2, "Bela": 3, "Cato": 4},
            "budget": 3,
        },
        "Bela",
    ),
    "emergency-exception-chain": (
        {"triggered": True, "waiver": True, "safety_review": True},
        True,
    ),
    "universal-access": (
        {"participants": ["Ada", "Bela"], "cleared": ["Ada"], "exempt": ["Bela"]},
        True,
    ),
    "common-availability": (
        {"slots_a": ["a", "b"], "slots_b": ["b", "c"], "slots_c": ["a", "c"]},
        False,
    ),
    "exposure-budget": ({"leg_a": 3, "leg_b": 4, "budget": 6}, False),
    "weighted-compliance": (
        {"late_events": 1, "unresolved": 1, "credit": 0},
        1,
    ),
    "completion-rate-rubric": (
        {"completed": 7, "assigned": 10, "thresholds": [20, 40, 60, 80]},
        3,
    ),
    "three-way-consistency": (
        {"reading_a": "x", "reading_b": "x", "reading_c": "y"},
        2,
    ),
    "ordered-milestones": (
        {"stage_a": True, "stage_b": True, "stage_c": True, "stage_d": True},
        4,
    ),
}


def test_independent_oracles_agree_on_twelve_distinct_rules():
    for name, (facts, expected) in CASES.items():
        assert evaluate(OPS[name], facts) == expected
        assert reference(OPS[name], facts) == expected


def test_source_lead_rejects_inventory_cues():
    spec = {
        "slug": "synthetic",
        "operation_id": "emergency-exception-chain",
        "scene": (
            "Decide whether the special exception applies to this request. "
            "Use the rule below and evaluate the relevant evidence for "
            "the current operating window only."
        ),
        "facts": CASES["emergency-exception-chain"][0],
        "sources": [],
    }
    with pytest.raises(ValueError, match="Source count"):
        build_item(spec, b"0" * 32)
    spec["scene"] = (
        "The attached record shows a signed waiver. Decide whether the "
        "special exception applies using the current rule and other material."
    )
    with pytest.raises(ValueError, match="preamble"):
        build_item(spec, b"0" * 32)


def test_repeated_source_prose_is_rejected():
    body = "orange coral wave willow maple harbor violet cedar lunar river"
    specs = [
        {"sources": [{"body": body}]},
        {"sources": [{"body": body}]},
    ]
    with pytest.raises(ValueError, match="eight-word"):
        lexical_gate(specs)
