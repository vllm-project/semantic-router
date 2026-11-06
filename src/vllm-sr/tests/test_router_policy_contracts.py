"""Cross-object policy and predicate validation tests."""

import pytest
from cli.models import Decision, NumericPredicate
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
