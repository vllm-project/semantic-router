import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

models = importlib.import_module("cli.models")
UserConfig = models.UserConfig
Rules = models.Rules


def _decision(**overrides):
    decision = {
        "name": "vision_request",
        "priority": 1000,
        "rules": {
            "operator": "AND",
            "conditions": [{"type": "conversation", "name": "image_input"}],
        },
        "modelRefs": [{"model": "vision-model"}],
    }
    decision.update(overrides)
    return decision


def test_decision_without_description_parses():
    config = UserConfig(version="0.3", routing={"decisions": [_decision()]})

    assert config.decisions[0].description is None


def test_decision_with_description_still_parses():
    config = UserConfig(
        version="0.3",
        routing={"decisions": [_decision(description="Routes vision requests.")]},
    )

    assert config.decisions[0].description == "Routes vision requests."


def test_decision_route_action_requires_jailbreak_condition():
    with pytest.raises(ValueError, match="jailbreak condition"):
        UserConfig(
            version="0.3",
            routing={
                "decisions": [
                    _decision(action={"type": "route", "destination": "safe-model"})
                ]
            },
        )


def test_decision_route_action_with_jailbreak_condition_parses():
    decision = _decision(
        rules={
            "operator": "AND",
            "conditions": [{"type": "jailbreak", "name": "prompt_injection"}],
        },
        action={"type": "route", "destination": "safe-model"},
    )
    config = UserConfig(version="0.3", routing={"decisions": [decision]})

    assert config.decisions[0].action.destination == "safe-model"


def test_decision_route_action_accepts_root_leaf_jailbreak_rule():
    decision = _decision(
        rules={"type": "jailbreak", "name": "prompt_injection"},
        action={"type": "route", "destination": "safe-model"},
    )
    config = UserConfig(version="0.3", routing={"decisions": [decision]})

    assert config.decisions[0].action.destination == "safe-model"
    assert config.decisions[0].rules.conditions[0].type == "jailbreak"


def test_decision_route_action_accepts_nested_jailbreak_rule():
    decision = _decision(
        rules={
            "operator": "AND",
            "conditions": [
                {
                    "operator": "OR",
                    "conditions": [{"type": "jailbreak", "name": "prompt_injection"}],
                }
            ],
        },
        action={"type": "route", "destination": "safe-model"},
    )
    config = UserConfig(version="0.3", routing={"decisions": [decision]})

    assert config.decisions[0].action.destination == "safe-model"


def test_decision_route_action_rejects_unknown_type():
    with pytest.raises(ValueError):
        _ = UserConfig(
            version="0.3",
            routing={
                "decisions": [
                    _decision(action={"type": "block", "destination": "safe-model"})
                ]
            },
        )


def test_rules_reject_on_unknown_with_condition_on_error():
    conflicting = {
        "operator": "AND",
        "on_unknown": "no_match",
        "conditions": [
            {
                "type": "classifier",
                "name": "risk",
                "label": "RISKY",
                "predicate": {"gte": 0.5},
                "on_error": "no_match",
            }
        ],
    }
    with pytest.raises(ValueError, match="on_error has no effect"):
        Rules(**conflicting)

    nested = {
        "operator": "AND",
        "on_unknown": "match",
        "conditions": [
            {"operator": "OR", "conditions": [conflicting["conditions"][0]]}
        ],
    }
    with pytest.raises(ValueError, match="on_error has no effect"):
        Rules(**nested)

    del conflicting["on_unknown"]
    assert Rules(**conflicting).conditions[0].on_error == "no_match"
