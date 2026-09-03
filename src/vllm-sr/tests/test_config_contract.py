from typing import get_args

import pytest
from cli.algorithms import AlgorithmConfig
from cli.config_contract import (
    LEGACY_SIGNAL_KEY_TO_CANONICAL,
    QUORUM_FAILURE_POLICY_VALUES,
    UNKNOWN_POLICY_VALUES,
    QuorumFailurePolicy,
    UnknownPolicy,
    build_projection_reference_index,
    build_signal_reference_index,
    signal_reference_exists,
)
from cli.models import Decision, Projections, Signals


def test_unknown_policy_literal_matches_allowed_values():
    assert get_args(UnknownPolicy) == UNKNOWN_POLICY_VALUES
    assert UNKNOWN_POLICY_VALUES == ("no_match", "match", "fail_request")


def test_quorum_failure_policy_literal_matches_allowed_values():
    assert get_args(QuorumFailurePolicy) == QUORUM_FAILURE_POLICY_VALUES
    assert QUORUM_FAILURE_POLICY_VALUES == ("fail", "fallback", "best_available")


def test_legacy_signal_inventory_covers_flat_authz_and_context_blocks():
    assert LEGACY_SIGNAL_KEY_TO_CANONICAL["role_bindings"] == "role_bindings"
    assert LEGACY_SIGNAL_KEY_TO_CANONICAL["context_rules"] == "context"
    assert LEGACY_SIGNAL_KEY_TO_CANONICAL["events"] == "events"
    assert "session_metrics" not in LEGACY_SIGNAL_KEY_TO_CANONICAL


def test_build_signal_reference_index_expands_complexity_levels_and_authz_names():
    signals = Signals(
        complexity=[
            {
                "name": "difficulty",
                "easy": {"candidates": ["simple"]},
                "hard": {"candidates": ["complex"]},
            }
        ],
        role_bindings=[
            {
                "name": "admin-access",
                "role": "admin",
                "subjects": [{"kind": "User", "name": "alice"}],
            }
        ],
        events=[
            {
                "name": "critical_event",
                "event_types": ["payment_failed"],
                "severities": ["critical"],
            }
        ],
    )

    signal_names = build_signal_reference_index(signals)

    assert signal_names["complexity"] == {
        "difficulty:easy",
        "difficulty:medium",
        "difficulty:hard",
    }
    assert signal_names["authz"] == {"admin-access"}
    assert signal_names["event"] == {"critical_event"}


def test_signal_reference_exists_is_scoped_by_family_and_exact_runtime_name():
    signal_names = {
        "keyword": {"security"},
        "authz": {"admin-access"},
    }

    assert signal_reference_exists(signal_names, "keyword", "security")
    assert not signal_reference_exists(signal_names, "keyword", "security:match")
    assert signal_reference_exists(signal_names, "authz", "admin-access")
    assert not signal_reference_exists(signal_names, "complexity", "security:match")
    assert not signal_reference_exists(signal_names, "authz", "security")


def test_projection_reference_index_collects_mapping_outputs():
    projections = Projections(
        mappings=[
            {
                "name": "difficulty_band",
                "source": "difficulty_score",
                "method": "threshold_bands",
                "outputs": [
                    {"name": "balance_simple", "lt": 0.15},
                    {"name": "balance_medium", "gte": 0.15, "lt": 0.45},
                ],
            }
        ]
    )

    assert build_projection_reference_index(projections) == {
        "balance_simple",
        "balance_medium",
    }


def test_decision_accepts_typed_output_contract_spec():
    decision = Decision(
        name="gpqa",
        description="strict choice route",
        priority=100,
        rules={"operator": "AND", "conditions": []},
        output_contract="Return exactly one answer letter: A, B, C, or D.",
        output_contract_spec={
            "type": "choice",
            "choice_set": {"values": ["A", "B", "C", "D"]},
            "render": {"mode": "value"},
            "extract": {"mode": "exact", "sources": ["content"]},
        },
        modelRefs=[{"model": "model-a", "use_reasoning": False}],
    )

    assert decision.output_contract_spec is not None
    assert decision.output_contract_spec.choice_set is not None
    assert decision.output_contract_spec.choice_set.values == ["A", "B", "C", "D"]


def test_decision_without_rules_is_a_match_all_fallback():
    decision = Decision(
        name="fallback",
        description="terminal route",
        priority=1,
        modelRefs=[{"model": "model-a", "use_reasoning": False}],
    )

    assert decision.rules.operator == "AND"
    assert decision.rules.conditions == []


def test_decision_enforces_minimum_candidates_after_materialization():
    with pytest.raises(ValueError, match="minimum_candidates=2"):
        Decision(
            name="panel",
            priority=1,
            modelRefs=[{"model": "model-a", "use_reasoning": False}],
            algorithm=AlgorithmConfig(type="fusion", minimum_candidates=2),
        )

    model_free = Decision(
        name="model-free-panel",
        priority=1,
        modelRefs=[],
        algorithm=AlgorithmConfig(type="fusion", minimum_candidates=2),
    )
    assert model_free.modelRefs == []


def test_decision_accepts_terminal_action_output_contract_spec():
    decision = Decision(
        name="terminal",
        description="terminal action route",
        priority=100,
        rules={"operator": "AND", "conditions": []},
        output_contract="Return one terminal action JSON object.",
        output_contract_spec={
            "type": "structured_json",
            "json_schema": {"schema_ref": "terminal_action_v1"},
            "extract": {
                "mode": "json_object",
                "sources": ["content", "candidate_responses"],
            },
        },
        modelRefs=[{"model": "model-a", "use_reasoning": False}],
    )

    assert decision.output_contract_spec is not None
    assert decision.output_contract_spec.json_schema is not None
    assert decision.output_contract_spec.json_schema.schema_ref == "terminal_action_v1"
