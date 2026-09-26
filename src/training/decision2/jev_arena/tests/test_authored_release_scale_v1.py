"""Small generic fixtures for the private authored release-scale builder."""

from __future__ import annotations

import pytest

from jev_arena.authored_release_scale_audit import _target_balance
from jev_arena.authored_release_scale_v1 import inspect, render_source, solve


def test_new_choice_operations_require_both_sources() -> None:
    coverage = {"a": ["cold", "signed"], "b": ["cold"]}
    costs = {"a": 8, "b": 5}
    assert (
        solve("coverage_cost", coverage, costs, {"needed": ["cold", "signed"]}) == "a"
    )
    assert (
        solve(
            "coverage_cost",
            {"a": ["cold"], "b": ["signed"]},
            costs,
            {"needed": ["cold", "signed"]},
        )
        == "HOLD"
    )
    eligibility = {
        "a": {"qualified": True, "priority": 2},
        "b": {"qualified": False, "priority": 9},
    }
    deadlines = {"a": 12, "b": 10}
    assert (
        solve("eligibility_deadline", eligibility, deadlines, {"deadline": 12}) == "a"
    )
    assert (
        solve("eligibility_deadline", eligibility, deadlines, {"deadline": 11})
        == "HOLD"
    )
    left = {
        "A": {"quality": 90, "endurance": 40, "certified": True},
        "B": {"quality": 90, "endurance": 40, "certified": False},
    }
    right = {
        "A": {"available": False, "price": 8},
        "B": {"available": True, "price": 5},
    }
    params = {
        "priority": ["A", "B"],
        "minimum_quality": 80,
        "minimum_endurance": 30,
    }
    assert solve("dual_clearance_choice", left, right, params) == "HOLD"
    changed_left = {"A": left["A"], "B": {**left["B"], "certified": True}}
    assert solve("dual_clearance_choice", changed_left, right, params) == "B"
    assert (
        solve(
            "dual_clearance_choice",
            left,
            {"A": {"available": True, "price": 8}, "B": right["B"]},
            params,
        )
        == "A"
    )


def test_new_noul_operations_are_boolean() -> None:
    assert (
        solve(
            "quorum_veto",
            {"signers": ["a", "b"]},
            {"roster": ["a", "b", "c"], "veto": ["c"]},
            {"quorum": 2},
        )
        is True
    )
    assert (
        solve(
            "quorum_veto",
            {"signers": ["a", "c"]},
            {"roster": ["a", "b", "c"], "veto": ["c"]},
            {"quorum": 2},
        )
        is False
    )
    transfers = [
        {"id": "t1", "time": 2, "from": "vault", "to": "courier"},
        {"id": "t2", "time": 4, "from": "courier", "to": "lab"},
    ]
    assert solve("custody_chain", transfers, {"t1": "courier", "t2": "lab"}, {}) is True
    assert (
        solve("custody_chain", transfers, {"t1": "courier", "t2": "desk"}, {}) is False
    )
    stock = {"east": 8, "west": 7}
    need = {"east": 6, "west": 5}
    assert solve("allocation_envelope", stock, need, {"maximum_surplus": 5}) is True
    assert (
        solve(
            "allocation_envelope", stock, {"east": 9, "west": 5}, {"maximum_surplus": 5}
        )
        is False
    )


def test_new_score_operations_obey_ordered_bands() -> None:
    hazards = {"x": {"severity": 2}, "y": {"severity": 4}}
    exposure = {
        "x": {"exposure": 2, "controlled": False},
        "y": {"exposure": 3, "controlled": False},
    }
    assert solve("risk_matrix", hazards, exposure, {"limits": [3, 8]}) == 0
    exposure["y"]["controlled"] = True
    assert solve("risk_matrix", hazards, exposure, {"limits": [3, 8]}) == 1
    claims = {"x": "yes", "y": "yes"}
    second_read = {"x": "yes", "y": "no"}
    params = {"claim": "yes", "weights": {"x": 2, "y": 3}, "limits": [2, 5]}
    assert solve("evidence_agreement", claims, second_read, params) == 1
    second_read["y"] = "yes"
    assert solve("evidence_agreement", claims, second_read, params) == 2


def test_private_document_requires_all_structured_fields() -> None:
    source = {
        "title": "Toy register",
        "form": "table",
        "data": {"unit": "A", "count": 3},
        "document": "Unit {unit} recorded {count} items.",
    }
    assert "Unit A recorded 3" in render_source(source)
    source["document"] = "Unit {unit} recorded items."
    with pytest.raises(ValueError, match="Every source fact"):
        render_source(source)
    ledger = {
        "title": "Revision ledger",
        "form": "ledger",
        "data": {"1": "approved", "2": "provisional"},
        "document": "The complete revision ledger reads {entries}.",
    }
    assert '"1": "approved"' in render_source(ledger)


def test_substitution_keeps_both_sources_causally_necessary() -> None:
    case = {
        "slug": "toy",
        "operation": "net_range",
        "domain": "toy",
        "scene": "A toy shipment must be evaluated using two separate complete records.",
        "contract": "Subtract tare from gross and accept an inclusive result from seven through nine.",
        "question": "Does the toy shipment satisfy the stated inclusive net weight range?",
        "sources": [
            {
                "side": "left",
                "title": "Gross",
                "form": "ticket",
                "document": "The recorded gross weight is {gross} units.",
                "data": {"gross": 10},
            },
            {
                "side": "right",
                "title": "Tare",
                "form": "ledger",
                "document": "The recorded tare weight is {tare} units.",
                "data": {"tare": 2},
            },
        ],
        "params": {"minimum": 7, "maximum": 9},
        "criteria": {"true": "yes", "false": "no"},
        "option_order": [],
        "witnesses": {
            "left": [{"gross": 5}, {"gross": 9}],
            "right": [{"tare": 1}, {"tare": 6}],
        },
        "variant": {"side": "right", "data": {"tare": 4}},
        "variant_witnesses": {
            "left": [{"gross": 8}, {"gross": 12}],
            "right": [{"tare": 1}, {"tare": 5}],
        },
        "provenance": {
            "origin": "test",
            "rights": "test",
            "redistribution": "test",
            "source_family": "toy",
        },
    }
    proof = inspect(case)
    assert proof["original"] is True and proof["variant"] is False
    case["variant_witnesses"]["left"] = [{"gross": 8}, {"gross": 9}]
    with pytest.raises(ValueError, match="unnecessary"):
        inspect(case)


def test_aggregate_preflight_detects_answer_shortcuts() -> None:
    cases = [
        {
            "slug": f"choice-{i}",
            "option_order": ["A", "B", "C", "HOLD"],
            "criteria": {"A": "a", "B": "b", "C": "c", "HOLD": "hold"},
        }
        for i in range(4)
    ]
    proofs = (
        [{"slug": f"choice-{i}", "type": "choice", "original": "A"} for i in range(4)]
        + [{"slug": f"noul-{i}", "type": "noul", "original": True} for i in range(3)]
        + [{"slug": f"score-{i}", "type": "score", "original": 1} for i in range(3)]
    )
    counts, reasons = _target_balance(cases, proofs)
    assert counts["noul"] == {"true": 3}
    assert "noul_target_shortcut" in reasons
    assert "score_target_shortcut" in reasons
    assert "choice_position_imbalance" in reasons
    assert "unused_choice_hold_option" in reasons
