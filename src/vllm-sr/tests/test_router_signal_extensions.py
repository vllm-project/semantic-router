"""Tests for router signal and decision extension schema."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.models import (  # noqa: E402
    Condition,
    MetadataPredicate,
    MetadataRule,
    Rules,
    UserConfig,
)
from cli.validator import validate_user_config  # noqa: E402


def test_metadata_rule_exact_match():
    rule = MetadataRule(
        name="consent-denied",
        key="consent",
        predicate=MetadataPredicate(equals="denied"),
    )
    assert rule.predicate.equals == "denied"


def test_metadata_predicate_rejects_multiple_comparators():
    with pytest.raises(ValidationError):
        MetadataPredicate(equals="denied", exists=True)


def test_numeric_predicate_rejects_nan():
    with pytest.raises(ValidationError):
        Condition(
            type="structure",
            name="bytes",
            predicate={"gte": float("nan")},
        )


def test_systemone_condition_score_shape():
    condition = Condition(
        type="systemone",
        name="risk",
        label="RISKY",
        predicate={"gte": 0.5},
        on_error="no_match",
    )
    assert condition.predicate.gte == 0.5


def test_unlabeled_condition_type_rejects_label():
    with pytest.raises(ValidationError):
        Condition(type="structure", name="bytes", label="RISKY")


def test_non_systemone_condition_rejects_on_error():
    with pytest.raises(ValidationError):
        Condition(
            type="structure",
            name="bytes",
            predicate={"gte": 10},
            on_error="match",
        )


def test_bare_systemone_leaf_preserves_predicate_fields():
    rules = Rules.model_validate(
        {
            "type": "systemone",
            "name": "risk",
            "label": "RISKY",
            "predicate": {"gte": 0.8},
            "on_error": "match",
        }
    )
    condition = rules.conditions[0]
    assert condition.label == "RISKY"
    assert condition.predicate.gte == 0.8
    assert condition.on_error == "match"


def test_rules_accept_terminal_on_unknown():
    rules = Rules.model_validate(
        {
            "type": "systemone",
            "name": "risk",
            "label": "RISKY",
            "predicate": {"gte": 0.8},
            "on_unknown": "fail_request",
        }
    )
    assert rules.on_unknown == "fail_request"


def test_nested_condition_rejects_on_unknown():
    with pytest.raises(ValidationError):
        Rules.model_validate(
            {
                "operator": "AND",
                "conditions": [
                    {
                        "type": "systemone",
                        "name": "risk",
                        "label": "RISKY",
                        "predicate": {"gte": 0.8},
                        "on_unknown": "match",
                    }
                ],
            }
        )


def test_user_config_accepts_recipe_entrypoints():
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "entrypoints": [
                {"model_names": ["vllm-sr/auto"], "recipe": "default", "default": True},
                {"model_names": ["vllm-sr/privacy"], "recipe": "privacy"},
            ],
            "recipes": [
                {"name": "default"},
                {
                    "name": "privacy",
                    "signals": {
                        "metadata": [
                            {
                                "name": "consent-denied",
                                "key": "consent",
                                "predicate": {"equals": "denied"},
                            }
                        ]
                    },
                },
            ],
        }
    )
    assert config.entrypoints[1].recipe == "privacy"
    assert config.recipes[1].signals.metadata[0].name == "consent-denied"


def test_user_config_rejects_unknown_recipe_entrypoint():
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "entrypoints": [{"model_names": ["vllm-sr/missing"], "recipe": "missing"}],
        }
    )
    errors = validate_user_config(config)
    assert any("unknown recipe" in error.message.lower() for error in errors)


def test_general_validators_cover_recipe_decisions():
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "providers": {"models": [{"name": "model-a"}]},
            "recipes": [
                {"name": "default"},
                {
                    "name": "invalid-recipe",
                    "decisions": [
                        {
                            "name": "invalid-route",
                            "description": "invalid",
                            "priority": 1,
                            "rules": {
                                "operator": "AND",
                                "conditions": [{"type": "latency", "name": "p95"}],
                            },
                            "modelRefs": [{"model": "missing-model"}],
                            "algorithm": {
                                "type": "static",
                                "multi_factor": {"weights": {"quality": 1}},
                            },
                        }
                    ],
                },
            ],
            "entrypoints": [
                {"model_names": ["vllm-sr/auto"], "recipe": "default", "default": True}
            ],
        }
    )
    errors = validate_user_config(config)
    fields = {error.field for error in errors}
    assert "recipes.invalid-recipe.decisions.invalid-route.modelRefs" in fields
    assert (
        "recipes.invalid-recipe.decisions.invalid-route.algorithm.multi_factor"
        in fields
    )
    assert "recipes.invalid-recipe.decisions.invalid-route.rules.conditions" in fields


def test_user_config_rejects_recipe_owned_model_cards():
    with pytest.raises(ValidationError):
        UserConfig.model_validate(
            {
                "version": "v0.5",
                "recipes": [
                    {"name": "default"},
                    {"name": "with-models", "modelCards": [{"name": "model-a"}]},
                ],
                "entrypoints": [
                    {
                        "model_names": ["vllm-sr/auto"],
                        "recipe": "default",
                        "default": True,
                    }
                ],
            }
        )


def test_user_config_allows_recipe_local_decision_names():
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "recipes": [
                {
                    "name": "default",
                    "decisions": [
                        {
                            "name": "shared-name",
                            "description": "default",
                            "priority": 1,
                            "rules": {},
                            "modelRefs": [],
                        }
                    ],
                },
                {
                    "name": "other",
                    "decisions": [
                        {
                            "name": "shared-name",
                            "description": "recipe",
                            "priority": 1,
                            "rules": {},
                            "modelRefs": [],
                        }
                    ],
                },
            ],
            "entrypoints": [
                {"model_names": ["vllm-sr/auto"], "recipe": "default", "default": True}
            ],
        }
    )
    assert config.recipes[0].decisions[0].name == "shared-name"


def test_user_config_accepts_recipes_only_default_profile():
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "recipes": [
                {
                    "name": "default",
                    "signals": {
                        "metadata": [
                            {
                                "name": "canary",
                                "key": "cohort",
                                "predicate": {"equals": "canary"},
                            }
                        ]
                    },
                }
            ],
            "entrypoints": [
                {"model_names": ["vllm-sr/default-route"], "recipe": "default"}
            ],
        }
    )
    assert config.recipes[0].name == "default"


def test_user_config_allows_identical_signal_redeclaration():
    metadata = {
        "name": "canary",
        "key": "cohort",
        "predicate": {"equals": "canary"},
    }
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "recipes": [
                {"name": "default", "signals": {"metadata": [metadata]}},
                {"name": "other", "signals": {"metadata": [metadata]}},
            ],
        }
    )
    assert config.recipes[0].signals.metadata[0].name == "canary"


def test_user_config_allows_conflicting_names_across_isolated_recipes():
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "recipes": [
                {
                    "name": "default",
                    "signals": {
                        "metadata": [
                            {
                                "name": "canary",
                                "key": "cohort",
                                "predicate": {"equals": "canary"},
                            }
                        ]
                    },
                },
                {
                    "name": "other",
                    "signals": {
                        "metadata": [
                            {
                                "name": "canary",
                                "key": "cohort",
                                "predicate": {"equals": "beta"},
                            }
                        ]
                    },
                },
            ],
            "entrypoints": [
                {"model_names": ["vllm-sr/auto"], "recipe": "default", "default": True}
            ],
        }
    )
    predicate = config.recipes[1].signals.metadata[0].predicate
    assert predicate.equals == "beta"


def test_user_config_allows_recipe_local_projection_names():
    partition = {
        "name": "difficulty",
        "semantics": "difficulty",
        "members": ["easy", "hard"],
        "default": "easy",
    }
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "recipes": [
                {"name": "default", "projections": {"partitions": [partition]}},
                {"name": "other", "projections": {"partitions": [partition]}},
            ],
        }
    )
    assert config.recipes[0].projections.partitions[0].name == "difficulty"


@pytest.mark.parametrize(
    "model_name",
    ["model-a", "adapter-a"],
)
def test_user_config_rejects_entrypoint_name_collisions(model_name):
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "providers": {
                "models": [{"name": "model-a", "loras": [{"name": "adapter-a"}]}]
            },
            "recipes": [{"name": "default"}],
            "entrypoints": [{"model_names": [model_name], "recipe": "default"}],
        }
    )
    errors = validate_user_config(config)
    assert any("conflicts" in error.message.lower() for error in errors)


def test_user_config_normalizes_entrypoint_names():
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "entrypoints": [
                {
                    "model_names": [" virtual-route ", "virtual-route", ""],
                    "recipe": " default ",
                }
            ],
        }
    )
    assert config.entrypoints[0].model_names == ["virtual-route"]
    assert config.entrypoints[0].recipe == "default"


def test_user_config_rejects_empty_entrypoint_names():
    with pytest.raises(ValidationError):
        UserConfig.model_validate(
            {
                "version": "v0.5",
                "entrypoints": [{"model_names": [], "recipe": "default"}],
            }
        )
