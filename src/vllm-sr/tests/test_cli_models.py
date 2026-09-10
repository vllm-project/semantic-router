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

    root = {
        "operator": "AND",
        "on_unknown": "no_match",
        "on_error": "no_match",
        "conditions": [{"type": "keyword", "name": "x"}],
    }
    with pytest.raises(ValueError, match="on_error has no effect"):
        Rules(**root)
    del root["on_unknown"]
    with pytest.raises(ValueError, match="only applies to leaf"):
        Rules(**root)

    del conflicting["on_unknown"]
    assert Rules(**conflicting).conditions[0].on_error == "no_match"
    assert Rules(operator="AND", on_error="", on_unknown="no_match").on_error == ""


def test_rules_reject_unknown_keys_on_every_shape():
    with pytest.raises(ValueError, match="on_errror"):
        Rules(operator="AND", on_errror="no_match", conditions=[])
    with pytest.raises(ValueError, match="on_errror"):
        Rules(type="keyword", name="x", on_errror="no_match")
    with pytest.raises(ValueError, match="on_errror"):
        Rules(
            operator="AND",
            conditions=[{"type": "keyword", "name": "x", "on_errror": "no_match"}],
        )
    with pytest.raises(ValueError, match="child conditions"):
        Rules(type="keyword", name="x", conditions=[{"type": "keyword", "name": "y"}])


def test_custom_evaluation_and_model_evidence_round_trip():
    config = UserConfig.model_validate(
        {
            "version": "0.3",
            "evaluation": {
                "benchmarks": [
                    {
                        "id": "acme/legal-reasoning@1.0.0",
                        "display_name": "Acme Legal Reasoning",
                        "domain": "reasoning",
                        "default_profile": "default",
                        "profiles": [
                            {
                                "id": "default",
                                "display_name": "Default",
                                "description": "Frozen public test split.",
                            }
                        ],
                        "metrics": [
                            {
                                "id": "accuracy",
                                "unit": "percent",
                                "direction": "higher_is_better",
                                "range": [0, 100],
                            }
                        ],
                    }
                ],
                "indices": [
                    {
                        "id": "acme/legal@1.0.0",
                        "display_name": "Acme Legal",
                        "aggregation": "weighted_mean",
                        "scale": [0, 100],
                        "missing": {"policy": "require_all"},
                        "components": [
                            {
                                "benchmark": "acme/legal-reasoning@1.0.0",
                                "metric": "accuracy",
                                "weight": 1,
                                "normalization": {
                                    "type": "linear_clamp",
                                    "min": 0,
                                    "max": 100,
                                },
                            }
                        ],
                    }
                ],
                "records": [
                    {
                        "model": "acme-model",
                        "benchmark": "acme/legal-reasoning@1.0.0",
                        "metrics": {"accuracy": 82},
                    }
                ],
            },
            "routing": {"modelCards": [{"name": "acme-model"}]},
        }
    )

    dumped = config.model_dump(by_alias=True, exclude_none=True)
    assert dumped["evaluation"]["indices"][0]["id"] == "acme/legal@1.0.0"
    assert dumped["evaluation"]["records"][0]["model"] == "acme-model"
    assert dumped["evaluation"]["records"][0]["metrics"]["accuracy"] == 82


def test_custom_evaluation_requires_versioned_namespace():
    with pytest.raises(ValueError):
        UserConfig.model_validate(
            {
                "version": "0.3",
                "evaluation": {
                    "benchmarks": [
                        {
                            "id": "legal-reasoning",
                            "display_name": "Legal Reasoning",
                            "domain": "reasoning",
                            "default_profile": "default",
                            "profiles": [
                                {
                                    "id": "default",
                                    "display_name": "Default",
                                    "description": "Frozen split.",
                                }
                            ],
                            "metrics": [
                                {
                                    "id": "accuracy",
                                    "unit": "percent",
                                    "direction": "higher_is_better",
                                    "range": [0, 100],
                                }
                            ],
                        }
                    ]
                },
            }
        )
