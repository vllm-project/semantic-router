import pytest
from cli.models import (
    AssignmentSet,
    Condition,
    DecisionAdaptationsConfig,
    Entrypoint,
    KeywordSignal,
    ModelAssignment,
    UserConfig,
)
from cli.validator import validate_user_config
from pydantic import ValidationError as PydanticValidationError


def model_value(name: str, port: int, *, reasoning_family: str | None = None) -> dict:
    return {
        "name": name,
        "provider_model_id": name,
        "reasoning_family": reasoning_family,
        "backend_refs": [
            {
                "provider": "openai-compatible",
                "base_url": f"http://127.0.0.1:{port}/v1",
            }
        ],
    }


def recipe_config(
    *,
    assigned_model: str | None = None,
    recipe_name: str | None = None,
) -> UserConfig:
    selected_recipe = recipe_name or "private"
    return UserConfig.model_validate(
        {
            "version": "v0.3",
            "providers": {"models": [model_value("model-a", 8000)]},
            "routing": {"modelCards": [{"name": "model-a", "capabilities": ["chat"]}]},
            "recipes": [
                {
                    "name": "private",
                    "routing": {
                        "decisions": [
                            {
                                "name": "private-route",
                                "description": "private",
                                "priority": 1,
                                "tier": 7,
                                "rules": {"operator": "AND", "conditions": []},
                            }
                        ],
                    },
                }
            ],
            "entrypoints": [
                {
                    "model_names": ["amd/rocm-v1-private"],
                    "recipe": selected_recipe,
                    "assignments": {
                        "private-route": {
                            "models": [{"model": assigned_model or "model-a"}]
                        }
                    },
                }
            ],
        }
    )


def test_entrypoint_assignment_references_are_validated():
    errors = validate_user_config(recipe_config(assigned_model="missing"))

    assert any("unknown Model 'missing'" in error.message for error in errors)


def test_recipe_decision_tier_survives_schema_parse():
    config = recipe_config()

    assert config.recipes[0].routing.decisions[0].tier == 7


def test_recipe_requires_at_least_one_decision():
    config = recipe_config()
    config.recipes[0].routing.decisions = []

    errors = validate_user_config(config)

    assert any("at least one decision" in error.message for error in errors)


def test_minimal_v03_runtime_manifest_is_valid():
    config = UserConfig.model_validate({"version": "v0.3"})

    assert config.providers.models == []
    assert config.routing.model_cards == []


def test_entrypoint_can_derive_complete_recipe_model_refs():
    document = recipe_config().model_dump(mode="json", by_alias=True)
    decision = document["recipes"][0]["routing"]["decisions"][0]
    decision["modelRefs"] = [{"model": "model-a"}]
    document["entrypoints"][0].pop("assignments")

    config = UserConfig.model_validate(document)
    errors = validate_user_config(config)

    assert not any("assignments are required" in error.message for error in errors)


def test_entrypoint_without_assignments_rejects_incomplete_recipe_defaults():
    document = recipe_config().model_dump(mode="json", by_alias=True)
    document["entrypoints"][0].pop("assignments")

    errors = validate_user_config(UserConfig.model_validate(document))

    assert any("assignments are required" in error.message for error in errors)


def test_entrypoints_must_reference_known_recipe():
    errors = validate_user_config(recipe_config(recipe_name="missing"))

    assert any("unknown Recipe" in error.message for error in errors)


def test_entrypoint_names_cannot_collide_with_models():
    config = recipe_config()
    config.entrypoints[0].model_names = ["model-a"]

    errors = validate_user_config(config)

    assert any("conflicts with a configured model" in error.message for error in errors)


def test_decision_adaptation_mode_boundaries_apply_inside_recipes():
    with pytest.raises(PydanticValidationError, match="cannot be 'apply'"):
        DecisionAdaptationsConfig(mode="bypass", adaptation={"mode": "apply"})


def test_entrypoint_identifiers_are_trimmed_and_deduplicated():
    entrypoint = Entrypoint.model_validate(
        {
            "model_names": [
                " amd/rocm-v1-balanced ",
                "amd/rocm-v1-balanced",
            ],
            "recipe": "balanced",
            "assignments": {"route": {"models": [{"model": "model-a"}]}},
        }
    )

    assert entrypoint.model_names == ["amd/rocm-v1-balanced"]
    with pytest.raises(PydanticValidationError, match="must be strings"):
        Entrypoint.model_validate(
            {
                "model_names": ["amd/rocm-v1-balanced", 123],
                "recipe": "balanced",
            }
        )


def test_entrypoint_assignments_round_trip_in_order():
    document = recipe_config().model_dump(mode="json", by_alias=True)
    document["providers"]["models"].append(
        model_value("model-b", 8001, reasoning_family="qwen")
    )
    document["routing"]["modelCards"].append(
        {"name": "model-b", "capabilities": ["chat", "reasoning"]}
    )
    decision_name = document["recipes"][0]["routing"]["decisions"][0]["name"]
    document["entrypoints"][0]["assignments"] = {
        decision_name: {
            "models": [
                {"model": "model-b", "reasoning": {"enabled": True}},
                {"model": "model-a", "reasoning": {"enabled": False}},
            ]
        }
    }

    config = UserConfig.model_validate(document)
    refs = config.entrypoints[0].assignments[decision_name].models

    assert [ref.model for ref in refs] == ["model-b", "model-a"]
    assert refs[1].reasoning is None
    assert not any(
        "Entrypoint assignment" in error.message
        for error in validate_user_config(config)
    )


def test_entrypoint_reasoning_requires_model_family():
    document = recipe_config().model_dump(mode="json", by_alias=True)
    decision_name = document["recipes"][0]["routing"]["decisions"][0]["name"]
    assignment = document["entrypoints"][0]["assignments"][decision_name]
    assignment["models"][0]["reasoning"] = {"enabled": True}

    errors = validate_user_config(UserConfig.model_validate(document))

    assert any(
        "does not support reasoning controls" in error.message for error in errors
    )


def test_disabled_assignment_reasoning_rejects_active_fields():
    with pytest.raises(PydanticValidationError, match="disabled assignment reasoning"):
        ModelAssignment.model_validate(
            {
                "model": "model-a",
                "reasoning": {"enabled": False, "effort": "high"},
            }
        )


@pytest.mark.parametrize(
    ("assignments", "expected"),
    [
        ({"missing": [{"model": "missing"}]}, "assign every"),
        ({"private-route": []}, "at least one Model reference"),
        ({"private-route": [{"model": "missing"}]}, "unknown Model"),
        (
            {
                "private-route": [
                    {"model": "model-a"},
                    {"model": "model-a"},
                ]
            },
            "repeats the same Model",
        ),
    ],
)
def test_invalid_entrypoint_assignments_are_reported(assignments, expected):
    config = recipe_config()
    config.entrypoints[0].assignments = {
        key: AssignmentSet.model_construct(
            models=[ModelAssignment.model_validate(ref) for ref in refs]
        )
        for key, refs in assignments.items()
    }

    errors = validate_user_config(config)

    assert any(expected in error.message for error in errors)


def test_signal_names_are_isolated_across_recipes():
    config = recipe_config()
    first = config.recipes[0]
    first.routing.signals.keywords = [
        KeywordSignal(
            name="shared-keyword",
            operator="OR",
            keywords=["one"],
            case_sensitive=False,
        )
    ]
    second = first.model_copy(deep=True)
    second.name = "second"
    second.routing.signals.keywords[0].keywords = ["two"]
    config.recipes.append(second)

    errors = validate_user_config(config)

    assert not any("shared-keyword" in error.message for error in errors)


def test_signal_references_cannot_cross_recipe_boundaries():
    config = recipe_config()
    config.recipes[0].routing.decisions[0].rules.conditions = [
        Condition(type="keyword", name="other-recipe-only")
    ]

    errors = validate_user_config(config)

    assert any(
        "unknown signal 'other-recipe-only'" in error.message for error in errors
    )


def test_nullable_global_sections_do_not_crash_validation():
    config = recipe_config()
    config.global_ = {"router": None, "integrations": None}

    validate_user_config(config)
