"""Small generic fixtures for the private authored release-scale builder."""

from __future__ import annotations

import pytest

from jev_arena.authored_release_scale_v1 import render_source, solve


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
