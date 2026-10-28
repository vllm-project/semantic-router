"""Safety policy contract matches native and external sequence heads."""

import pytest
from cli.models import UserConfig
from cli.models_safety import SafetyRule
from cli.validator_safety import validate_safety_contracts
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
    assert not validate_safety_contracts(config)


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


def test_recipe_safety_external_references_are_validated():
    config = UserConfig.model_validate(
        {
            "version": "v0.3",
            "recipes": [
                {
                    "name": "private",
                    "routing": {
                        "signals": {
                            "safety": [
                                {"name": "unsafe", "model": "missing", "threshold": 0.5}
                            ]
                        }
                    },
                }
            ],
        }
    )
    errors = validate_safety_contracts(config)
    assert len(errors) == 1
    assert "unknown external model" in errors[0].message
    assert "recipes.private" in errors[0].field


def test_native_safety_checks_context_override():
    config = UserConfig.model_validate(
        {
            "version": "v0.3",
            "routing": {"signals": {"safety": [{"name": "unsafe", "threshold": 0.5}]}},
            "global": {
                "model_catalog": {
                    "modules": {"safety": {"safety": {"max_sequence_length": -1}}}
                }
            },
        }
    )
    assert any(
        "max_sequence_length" in e.message for e in validate_safety_contracts(config)
    )
