"""Cross-object policy classifier and predicate validation tests."""

import pytest
from cli.models import Decision, NumericPredicate, UserConfig
from cli.validator import validate_user_config
from pydantic import ValidationError as PydanticValidationError


@pytest.mark.parametrize(
    "predicate",
    [
        {},
        {"gt": 1, "gte": 1},
        {"lt": 1, "lte": 1},
        {"gt": 5, "lt": 3},
        {"gt": 5, "lte": 5},
    ],
)
def test_numeric_predicate_rejects_invalid_shapes(predicate):
    with pytest.raises(PydanticValidationError):
        NumericPredicate.model_validate(predicate)


def test_decision_annotations_enforce_transport_bounds():
    base = {
        "name": "route",
        "description": "route",
        "priority": 1,
        "rules": {},
        "modelRefs": [],
    }
    with pytest.raises(PydanticValidationError):
        Decision.model_validate(
            {
                **base,
                "annotations": {f"k{i}": i for i in range(33)},
            }
        )
    with pytest.raises(PydanticValidationError):
        Decision.model_validate(
            {
                **base,
                "annotations": {"payload": "x" * 4097},
            }
        )


def test_prompt_candidates_reject_effective_lora_identity_collision():
    with pytest.raises(PydanticValidationError):
        Decision.model_validate(
            {
                "name": "prompt-route",
                "description": "route",
                "priority": 1,
                "rules": {},
                "modelRefs": [
                    {
                        "model": "base",
                        "lora_name": "specialist",
                        "use_reasoning": False,
                    },
                    {"model": "specialist", "use_reasoning": False},
                ],
                "algorithm": {
                    "type": "prompt",
                    "prompt": {
                        "model": "helper",
                        "instructions": "Choose.",
                    },
                },
            }
        )


def test_classifier_contract_rejects_multiple_local_rules():
    config = UserConfig.model_validate(
        {
            "version": "v0.3",
            "routing": {
                "signals": {
                    "classifiers": [
                        {
                            "name": "risk-a",
                            "type": "local",
                            "model_path": "models/risk-a",
                            "labels": ["SAFE", "RISKY"],
                        },
                        {
                            "name": "risk-b",
                            "type": "local",
                            "model_path": "models/risk-b",
                            "labels": ["SAFE", "RISKY"],
                        },
                    ]
                }
            },
        }
    )
    assert any(
        "only one local classifier" in error.message
        for error in validate_user_config(config)
    )


@pytest.mark.parametrize(
    ("external", "expected"),
    [
        ([], "unknown external model"),
        (
            [{"name": "judge", "model_role": "generation"}],
            "model_role=classification",
        ),
        (
            [{"name": "judge", "model_role": "classification"}],
            "requires llm_model_name",
        ),
    ],
)
def test_llm_classifier_requires_classification_external_model(
    external,
    expected,
):
    config = UserConfig.model_validate(
        {
            "version": "v0.3",
            "routing": {
                "signals": {
                    "classifiers": [
                        {
                            "name": "risk",
                            "type": "llm",
                            "model": "judge",
                            "labels": ["SAFE", "RISKY"],
                            "instructions": "Classify.",
                        }
                    ]
                }
            },
            "global": {"model_catalog": {"external": external}},
        }
    )
    assert any(expected in error.message for error in validate_user_config(config))


def test_classifier_decision_rejects_unknown_label_and_local_threshold():
    config = UserConfig.model_validate(
        {
            "version": "v0.3",
            "routing": {
                "signals": {
                    "classifiers": [
                        {
                            "name": "risk",
                            "type": "local",
                            "model_path": "models/risk",
                            "labels": ["SAFE", "RISKY"],
                        }
                    ]
                },
                "decisions": [
                    {
                        "name": "risk-route",
                        "description": "risk",
                        "priority": 1,
                        "rules": {
                            "type": "classifier",
                            "name": "risk",
                            "label": "UNKNOWN",
                            "predicate": {"gte": 0.4},
                        },
                        "modelRefs": [],
                    }
                ],
            },
        }
    )
    messages = [error.message for error in validate_user_config(config)]
    assert any("is not declared" in message for message in messages)
    assert any("predicate.gte >= 0.5" in message for message in messages)


def test_classifier_names_and_labels_reject_reserved_delimiters():
    with pytest.raises(PydanticValidationError):
        UserConfig.model_validate(
            {
                "version": "v0.3",
                "routing": {
                    "signals": {
                        "classifiers": [
                            {
                                "name": "risk:score",
                                "type": "local",
                                "model_path": "models/risk",
                                "labels": ["SAFE", "RISKY"],
                            }
                        ]
                    }
                },
            }
        )
    with pytest.raises(PydanticValidationError):
        UserConfig.model_validate(
            {
                "version": "v0.3",
                "routing": {
                    "signals": {
                        "classifiers": [
                            {
                                "name": "risk",
                                "type": "local",
                                "model_path": "models/risk",
                                "labels": [" SAFE ", "RISKY"],
                            }
                        ]
                    }
                },
            }
        )
