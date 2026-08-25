"""Tests for router signal and decision extension schema."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.algorithms import (  # noqa: E402
    AlgorithmConfig,
    ModelRef,
    PromptSelectionConfig,
)
from cli.models import (  # noqa: E402
    ClassifierSignal,
    Condition,
    Decision,
    MetadataPredicate,
    MetadataRule,
    Rules,
    UserConfig,
)
from cli.validator import validate_user_config  # noqa: E402


def _user_config(document: dict) -> UserConfig:
    global_config = document.setdefault("global", {})
    services = global_config.setdefault("services", {})
    services.setdefault(
        "backend_egress",
        {"policy_file": "/app/config/backend-egress-policy.yaml"},
    )
    return UserConfig.model_validate(document)


def _recipe(name, routing):
    return {
        "name": name,
        "routing": routing,
    }


def _entrypoint(name, recipe, assignments=None):
    return {
        "model_names": [name],
        "recipe": recipe,
        "assignments": assignments or {},
    }


def _provider_model(name="model-a"):
    return {
        "name": name,
        "provider_model_id": name,
        "backend_refs": [
            {
                "provider": "openai-compatible",
                "base_url": "http://model:8000/v1",
            }
        ],
    }


def _model_card(name="model-a", *, loras=None):
    return {
        "name": name,
        "loras": [{"name": lora} for lora in (loras or [])],
    }


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


def test_classifier_signal_llm_shape():
    signal = ClassifierSignal(
        name="risk",
        type="llm",
        model="risk-judge",
        labels=["SAFE", "RISKY"],
        instructions="Choose a label.",
    )
    assert signal.labels == ["SAFE", "RISKY"]


def test_classifier_signal_local_shape():
    signal = ClassifierSignal(
        name="risk",
        type="local",
        model_path="models/risk",
        labels=["SAFE", "RISKY"],
    )
    assert signal.model_path == "models/risk"


def test_classifier_signal_rejects_backend_mixed_fields():
    with pytest.raises(ValidationError):
        ClassifierSignal(
            name="risk",
            type="local",
            model="risk-judge",
            model_path="models/risk",
            labels=["SAFE", "RISKY"],
        )


def test_classifier_signal_rejects_whitespace_name():
    with pytest.raises(ValidationError):
        ClassifierSignal(
            name=" risk ",
            type="local",
            model_path="models/risk",
            labels=["SAFE", "RISKY"],
        )


def test_user_config_allows_recipe_local_classifier_names():
    config = _user_config(
        {
            "version": "v0.3",
            "recipes": [
                _recipe(
                    "default",
                    {
                        "signals": {
                            "classifiers": [
                                {
                                    "name": "Risk",
                                    "type": "local",
                                    "model_path": "models/risk",
                                    "labels": ["SAFE", "RISKY"],
                                }
                            ]
                        }
                    },
                ),
                _recipe(
                    "other",
                    {
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
                ),
            ],
        }
    )
    assert config.recipes[1].routing.signals.classifiers[0].name == "risk"


def test_user_config_rejects_case_colliding_classifier_names_within_recipe():
    with pytest.raises(ValidationError):
        _user_config(
            {
                "version": "v0.3",
                "recipes": [
                    _recipe(
                        "default",
                        {
                            "signals": {
                                "classifiers": [
                                    {
                                        "name": "Risk",
                                        "type": "local",
                                        "model_path": "models/risk",
                                        "labels": ["SAFE", "RISKY"],
                                    },
                                    {
                                        "name": "risk",
                                        "type": "llm",
                                        "model": "judge",
                                        "labels": ["SAFE", "RISKY"],
                                        "instructions": "Classify.",
                                    },
                                ]
                            }
                        },
                    )
                ],
            }
        )


def test_classifier_condition_score_shape():
    condition = Condition(
        type="classifier",
        name="risk",
        label="RISKY",
        predicate={"gte": 0.5},
        on_error="no_match",
    )
    assert condition.predicate.gte == 0.5


def test_classifier_condition_requires_label_and_predicate():
    with pytest.raises(ValidationError):
        Condition(type="classifier", name="risk")


def test_non_classifier_condition_rejects_label():
    with pytest.raises(ValidationError):
        Condition(type="embedding", name="risk", label="RISKY")


def test_non_classifier_condition_rejects_on_error():
    with pytest.raises(ValidationError):
        Condition(
            type="structure",
            name="bytes",
            predicate={"gte": 10},
            on_error="match",
        )


def test_bare_classifier_leaf_preserves_predicate_fields():
    rules = Rules.model_validate(
        {
            "type": "classifier",
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


def test_prompt_decision_rejects_duplicate_base_models():
    with pytest.raises(ValidationError):
        Decision(
            name="prompt-route",
            description="duplicate candidates",
            priority=1,
            rules=Rules(),
            modelRefs=[
                ModelRef(model="model-a"),
                ModelRef(model="model-a", lora_name="specialist"),
            ],
            algorithm=AlgorithmConfig(
                type="prompt",
                prompt=PromptSelectionConfig(
                    model="router-small",
                    instructions="Choose.",
                ),
            ),
        )


def test_user_config_accepts_recipe_entrypoints():
    config = _user_config(
        {
            "version": "v0.3",
            "entrypoints": [_entrypoint("vllm-sr/privacy", "privacy")],
            "recipes": [
                _recipe(
                    "privacy",
                    {
                        "signals": {
                            "metadata": [
                                {
                                    "name": "consent-denied",
                                    "key": "consent",
                                    "predicate": {"equals": "denied"},
                                }
                            ]
                        }
                    },
                )
            ],
        }
    )
    assert config.entrypoints[0].recipe == "privacy"
    assert config.recipes[0].routing.signals.metadata[0].name == "consent-denied"


def test_user_config_rejects_unknown_recipe_entrypoint():
    config = _user_config(
        {
            "version": "v0.3",
            "entrypoints": [_entrypoint("vllm-sr/missing", "missing")],
        }
    )
    errors = validate_user_config(config)
    assert any("unknown recipe" in error.message.lower() for error in errors)


def test_prompt_recipe_rejects_physical_model_selection():
    with pytest.raises(ValidationError, match="Entrypoint assignments"):
        _user_config(
            {
                "version": "v0.3",
                "recipes": [
                    _recipe(
                        "prompt-recipe",
                        {
                            "decisions": [
                                {
                                    "name": "prompt-route",
                                    "description": "prompt",
                                    "priority": 1,
                                    "rules": {},
                                    "algorithm": {
                                        "type": "prompt",
                                        "prompt": {
                                            "model": "helper",
                                            "instructions": "Choose.",
                                        },
                                    },
                                }
                            ]
                        },
                    )
                ],
            }
        )


def test_prompt_recipe_uses_entrypoint_assignments():
    config = _user_config(
        {
            "version": "v0.3",
            "providers": {
                "models": [_provider_model("model-a"), _provider_model("model-b")]
            },
            "routing": {"modelCards": [_model_card("model-a"), _model_card("model-b")]},
            "recipes": [
                _recipe(
                    "prompt-recipe",
                    {
                        "decisions": [
                            {
                                "name": "prompt-route",
                                "description": "prompt",
                                "priority": 1,
                                "rules": {},
                                "algorithm": {
                                    "type": "prompt",
                                    "prompt": {"instructions": "Choose."},
                                },
                            }
                        ]
                    },
                )
            ],
            "entrypoints": [
                _entrypoint(
                    "vllm-sr/prompt",
                    "prompt-recipe",
                    {
                        "prompt-route": {
                            "models": [
                                {"model": "model-a"},
                                {"model": "model-b"},
                            ]
                        }
                    },
                )
            ],
            "global": {
                "integrations": {
                    "looper": {"endpoint": "http://envoy:8899/v1/chat/completions"}
                }
            },
        }
    )
    assert not any(
        "algorithm.prompt" in error.field for error in validate_user_config(config)
    )


def test_general_validators_cover_recipe_decisions():
    config = _user_config(
        {
            "version": "v0.3",
            "recipes": [
                _recipe(
                    "invalid-recipe",
                    {
                        "decisions": [
                            {
                                "name": "invalid-route",
                                "description": "invalid",
                                "priority": 1,
                                "rules": {
                                    "operator": "AND",
                                    "conditions": [{"type": "latency", "name": "p95"}],
                                },
                                "algorithm": {
                                    "type": "static",
                                    "hybrid": {
                                        "elo_weight": 1,
                                        "router_dc_weight": 0,
                                        "automix_weight": 0,
                                    },
                                },
                            }
                        ]
                    },
                )
            ],
        }
    )
    errors = validate_user_config(config)
    fields = {error.field for error in errors}
    assert (
        "recipes.invalid-recipe.routing.decisions.invalid-route.algorithm.hybrid"
        in fields
    )
    assert (
        "recipes.invalid-recipe.routing.decisions.invalid-route.rules.conditions"
        in fields
    )


def test_user_config_rejects_recipe_owned_model_cards():
    with pytest.raises(ValidationError):
        _user_config(
            {
                "version": "v0.3",
                "recipes": [
                    _recipe(
                        "with-models",
                        {"modelCards": [{"name": "model-a"}]},
                    )
                ],
            }
        )


def test_user_config_allows_recipe_local_decision_names():
    config = _user_config(
        {
            "version": "v0.3",
            "recipes": [
                _recipe(
                    "default",
                    {
                        "decisions": [
                            {
                                "name": "shared-name",
                                "description": "default",
                                "priority": 1,
                                "rules": {},
                            }
                        ]
                    },
                ),
                _recipe(
                    "other",
                    {
                        "decisions": [
                            {
                                "name": "shared-name",
                                "description": "recipe",
                                "priority": 1,
                                "rules": {},
                            }
                        ]
                    },
                ),
            ],
        }
    )
    assert config.recipes[1].routing.decisions[0].name == "shared-name"


def test_user_config_accepts_recipes_only_default_profile():
    config = _user_config(
        {
            "version": "v0.3",
            "recipes": [
                _recipe(
                    "default",
                    {
                        "signals": {
                            "metadata": [
                                {
                                    "name": "canary",
                                    "key": "cohort",
                                    "predicate": {"equals": "canary"},
                                }
                            ]
                        }
                    },
                )
            ],
            "entrypoints": [_entrypoint("vllm-sr/default-route", "default")],
        }
    )
    assert config.recipes[0].name == "default"


def test_user_config_allows_identical_signal_redeclaration():
    metadata = {
        "name": "canary",
        "key": "cohort",
        "predicate": {"equals": "canary"},
    }
    config = _user_config(
        {
            "version": "v0.3",
            "recipes": [
                _recipe("default", {"signals": {"metadata": [metadata]}}),
                _recipe("other", {"signals": {"metadata": [metadata]}}),
            ],
        }
    )
    assert config.recipes[1].routing.signals.metadata[0].name == "canary"


def test_user_config_allows_conflicting_names_across_isolated_recipes():
    config = _user_config(
        {
            "version": "v0.3",
            "recipes": [
                _recipe(
                    "default",
                    {
                        "signals": {
                            "metadata": [
                                {
                                    "name": "canary",
                                    "key": "cohort",
                                    "predicate": {"equals": "canary"},
                                }
                            ]
                        }
                    },
                ),
                _recipe(
                    "other",
                    {
                        "signals": {
                            "metadata": [
                                {
                                    "name": "canary",
                                    "key": "cohort",
                                    "predicate": {"equals": "beta"},
                                }
                            ]
                        }
                    },
                ),
            ],
        }
    )
    predicate = config.recipes[1].routing.signals.metadata[0].predicate
    assert predicate.equals == "beta"


def test_user_config_allows_recipe_local_projection_names():
    partition = {
        "name": "difficulty",
        "semantics": "difficulty",
        "members": ["easy", "hard"],
        "default": "easy",
    }
    config = _user_config(
        {
            "version": "v0.3",
            "recipes": [
                _recipe("default", {"projections": {"partitions": [partition]}}),
                _recipe("other", {"projections": {"partitions": [partition]}}),
            ],
        }
    )
    assert config.recipes[1].routing.projections.partitions[0].name == "difficulty"


@pytest.mark.parametrize(
    "model_name",
    ["model-a", "adapter-a"],
)
def test_user_config_rejects_entrypoint_name_collisions(model_name):
    config = _user_config(
        {
            "version": "v0.3",
            "providers": {"models": [_provider_model()]},
            "routing": {"modelCards": [_model_card(loras=["adapter-a"])]},
            "recipes": [_recipe("default", {})],
            "entrypoints": [_entrypoint(model_name, "default")],
        }
    )
    errors = validate_user_config(config)
    assert any("conflicts" in error.message.lower() for error in errors)


@pytest.mark.parametrize("model_name", ["vllm-sr/auto", "vllm-sr/remom"])
def test_user_config_allows_explicit_virtual_entrypoint_names(model_name):
    config = _user_config(
        {
            "version": "v0.3",
            "recipes": [_recipe("default", {})],
            "entrypoints": [_entrypoint(model_name, "default")],
            "global": {"router": {"auto_model_names": []}},
        }
    )
    assert not any(
        "conflicts" in error.message.lower() for error in validate_user_config(config)
    )


def test_user_config_normalizes_entrypoint_names():
    config = _user_config(
        {
            "version": "v0.3",
            "entrypoints": [
                {
                    "model_names": [
                        " primary-route ",
                        " virtual-route ",
                        "virtual-route",
                        "",
                    ],
                    "recipe": "default",
                    "assignments": {},
                }
            ],
        }
    )
    assert config.entrypoints[0].model_names == ["primary-route", "virtual-route"]
    assert config.entrypoints[0].recipe == "default"


def test_user_config_rejects_empty_entrypoint_names():
    with pytest.raises(ValidationError):
        _user_config(
            {
                "version": "v0.3",
                "entrypoints": [
                    {
                        "model_names": [" "],
                        "recipe": "default",
                        "assignments": {},
                    }
                ],
            }
        )
