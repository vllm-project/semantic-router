"""CLI contract tests for the topic_continuity signal family."""

import pytest
from cli.models import Signals, TopicContinuityRule
from pydantic import ValidationError


def test_defaults_validate():
    rule = TopicContinuityRule(name="topic_boundary")
    assert rule.include_assistant is None
    assert rule.thresholds is None


def test_explicit_false_and_zero_are_kept():
    rule = TopicContinuityRule(
        name="strict", include_assistant=False, thresholds={"change": 0}
    )
    assert rule.include_assistant is False
    assert rule.thresholds.change == 0


@pytest.mark.parametrize(
    "fields",
    [
        {"name": " padded "},
        {"name": "r", "thresholds": {"continuation": 1}},
        {"name": "r", "thresholds": {"change": 0.5}},
        {"name": "r", "limits": {"max_prior_turns": 33}},
        {"name": "r", "limits": {"max_turn_bytes": 100}},
        {"name": "r", "limits": {"max_prior_turns": 32, "max_turn_bytes": 65536}},
        {"name": "r", "limits": {"max_input_bytes": 2048}},
        {"name": "r", "method": "lexical"},
    ],
)
def test_invalid_rules_are_rejected(fields):
    with pytest.raises(ValidationError):
        TopicContinuityRule(**fields)


def test_explicit_total_above_derived_cap_is_accepted():
    TopicContinuityRule(
        name="r",
        limits={
            "max_prior_turns": 32,
            "max_turn_bytes": 65536,
            "max_input_bytes": 1048576,
        },
    )


def test_rule_count_and_unique_names():
    Signals(topic_continuity=[{"name": f"r{i}"} for i in range(8)])
    with pytest.raises(ValidationError):
        Signals(topic_continuity=[{"name": f"r{i}"} for i in range(9)])
    with pytest.raises(ValidationError):
        Signals(topic_continuity=[{"name": "x"}, {"name": "x"}])
