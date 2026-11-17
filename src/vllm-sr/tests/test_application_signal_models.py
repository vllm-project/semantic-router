"""The application signals' canonical rules and the conditions that read them."""

import pytest
from cli.models import Condition, Signals, UserConfig
from cli.models_application import (
    JailbreakRule,
    PIIRule,
    SafetyRule,
    TurnRelationRule,
)
from pydantic import ValidationError


def test_a_safety_rule_names_harm_categories_with_a_predicate_and_stage():
    rule = SafetyRule.model_validate(
        {
            "name": "unsafe-answer",
            "labels": ["Harassment", "Fraud/Deception"],
            "predicate": {"gte": 0.46},
            "stage": "response",
            "deployment": "guard",
            "timeout_ms": 300,
        }
    )
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "recipes": [
                {"name": "default", "signals": {"safety": [rule.model_dump()]}}
            ],
        }
    )
    assert config.default_recipe.signals.safety == [rule]


@pytest.mark.parametrize(
    "patch",
    [
        {"threshold": 0.5},
        {"unsafe_labels": ["unsafe"]},
        {"model": "guard"},
        {"hazard": {"labels": ["Harassment"], "categories": ["Harassment"]}},
        {"name": " padded "},
        {"name": "unsafe:privacy"},
        {"stage": "both"},
        {"predicate": {"gte": float("nan")}},
        {"deployment": " "},
    ],
)
def test_a_safety_rule_refuses_removed_and_invalid_fields(patch):
    with pytest.raises(ValidationError):
        SafetyRule.model_validate({"name": "unsafe", **patch})


@pytest.mark.parametrize("model", [JailbreakRule, PIIRule])
def test_a_guard_rule_says_what_a_partly_scanned_input_reads_as(model):
    for value in ("best_effort", "match", "no_match", "unknown"):
        rule = model.model_validate({"name": "guard", "on_unscanned": value})
        assert rule.on_unscanned == value
    with pytest.raises(ValidationError):
        model.model_validate({"name": "guard", "on_unscanned": "allow"})


@pytest.mark.parametrize("model", [JailbreakRule, PIIRule, SafetyRule])
def test_a_guard_rule_takes_a_min_coverage_for_best_effort(model):
    for fields in ({"min_coverage": 0.5}, {"min_coverage": 1}):
        assert (
            model.model_validate({"name": "guard", **fields}).min_coverage
            == fields["min_coverage"]
        )
    for value in (-0.1, 1.5):
        with pytest.raises(ValidationError):
            model.model_validate({"name": "guard", "min_coverage": value})
    if model is not SafetyRule:
        assert (
            model.model_validate(
                {"name": "guard", "on_unscanned": "best_effort", "min_coverage": 0.5}
            ).min_coverage
            == 0.5
        )
        with pytest.raises(ValidationError, match="min_coverage applies to"):
            model.model_validate(
                {"name": "guard", "on_unscanned": "match", "min_coverage": 0.5}
            )


def test_a_turn_relation_rule_reads_previous_turns():
    rule = TurnRelationRule.model_validate(
        {"name": "reask", "labels": ["reask"], "input": {"previous_turns": 4}}
    )
    assert rule.input.previous_turns == 4
    with pytest.raises(ValidationError):
        TurnRelationRule.model_validate(
            {"name": "reask", "input": {"previous_turns": 33}}
        )


def test_rule_names_are_unique_within_a_type():
    Signals(turn_relation=[{"name": "x"}], refusal=[{"name": "x"}])
    with pytest.raises(ValidationError):
        Signals(refusal=[{"name": "x"}, {"name": "x"}])


@pytest.mark.parametrize(
    "condition_type", ["complexity", "safety", "turn_relation", "systemone"]
)
def test_application_and_systemone_conditions_name_a_label(condition_type):
    condition = Condition(type=condition_type, name="rule", label="hard")
    assert condition.label == "hard"


def test_application_conditions_refuse_on_error():
    with pytest.raises(ValidationError):
        Condition(type="complexity", name="rule", label="hard", on_error="match")
