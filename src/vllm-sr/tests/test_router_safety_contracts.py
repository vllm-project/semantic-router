"""Safety policy labels; the Router validates the model that answers them."""

import pytest
from cli.models import UserConfig
from cli.models_safety import SafetyRule
from pydantic import ValidationError


def test_local_safety_and_multilabel_hazard():
    rule = SafetyRule(
        name="unsafe",
        threshold=0.5,
        hazard={
            "labels": ["privacy", "violence"],
            "categories": ["privacy"],
            "threshold": 0.6,
        },
    )
    assert rule.model == ""
    config = UserConfig.model_validate(
        {"version": "v0.3", "routing": {"signals": {"safety": [rule.model_dump()]}}}
    )
    assert config.routing.signals.safety == [rule]


def test_safety_and_its_hazard_part_name_their_deployments():
    rule = SafetyRule.model_validate(
        {
            "name": "unsafe",
            "deployment": "guard",
            "threshold": 0.5,
            "hazard": {
                "deployment": "hazard",
                "labels": ["privacy", "violence"],
                "categories": ["privacy"],
                "threshold": 0.6,
            },
        }
    )
    assert rule.deployment == "guard" and rule.hazard.deployment == "hazard"


@pytest.mark.parametrize(
    "patch",
    [
        {"threshold": float("nan")},
        {"unsafe_labels": ["safe", "unsafe"]},
        {"labels": ["safe", "safe"]},
        {"model": " spaced "},
        {
            "hazard": {
                "labels": ["privacy", "violence"],
                "categories": ["unknown"],
                "threshold": 0.5,
            }
        },
    ],
)
def test_safety_rejects_invalid_label_contract(patch):
    with pytest.raises(ValidationError):
        SafetyRule.model_validate({"name": "unsafe", "threshold": 0.5, **patch})
