import pytest
from pydantic import ValidationError as PydanticValidationError

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


def _user_config(document: dict) -> UserConfig:
    global_config = document.setdefault("global", {})
    services = global_config.setdefault("services", {})
    services.setdefault(
        "backend_egress",
        {"policy_file": "/app/config/backend-egress-policy.yaml"},
    )
    return UserConfig.model_validate(document)


def model_value(name: str, port: int) -> dict:
    return {
        "name": name,
        "card": {"capabilities": ["chat"]},
        "runtime": {
            "max_retries": 0,
            "request_timeout": "300s",
            "stream_timeout": "300s",
        },
        "pricing": {},
        "connections": [
            {
                "provider": "openai-compatible",
                "endpoint": f"http://127.0.0.1:{port}/v1",
                "model": name,
                "weight": "1",
            }
        ],
    }


def recipe_config(*, assigned_model: str | None = None, recipe_name: str | None = None):
    model = model_value("model-a", 8000)
    selected_recipe = recipe_name or "private"
    return _user_config(
        {
            "version": "v0.4",
            "models": [model],
            "recipes": [
                {
                    "name": "private",
                    "document": {
                        "signals": {},
                        "projections": {},
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
                    "name": "amd/rocm-v1-private",
                    "recipe": selected_recipe,
                    "assignments": {
                        "private-route": {
                            "models": [{"model": assigned_model or model["name"]}]
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

    assert config.recipes[0].document.decisions[0].tier == 7


def test_recipe_requires_at_least_one_decision():
    config = recipe_config()
    config.recipes[0].document.decisions = []

    errors = validate_user_config(config)

    assert any("at least one decision" in error.message for error in errors)


def test_runtime_manifest_requires_backend_egress_policy():
    with pytest.raises(
        PydanticValidationError,
        match="global.services.backend_egress.policy_file is required",
    ):
        UserConfig.model_validate({"version": "v0.4"})


def test_recipe_rejects_physical_model_selection():
    document = recipe_config().model_dump(mode="json", by_alias=True)
    document["recipes"][0]["document"]["decisions"][0]["modelRefs"] = [
        {"model": "model-a"}
    ]

    with pytest.raises(PydanticValidationError, match="Entrypoint assignments"):
        _user_config(document)


def test_entrypoints_must_reference_known_recipe():
    errors = validate_user_config(recipe_config(recipe_name="missing"))

    assert any("unknown Recipe" in error.message for error in errors)


def test_entrypoint_names_cannot_collide_with_models():
    config = recipe_config()
    config.entrypoints[0].name = "model-a"

    errors = validate_user_config(config)

    assert any("conflicts with a configured model" in error.message for error in errors)


def test_decision_adaptation_mode_boundaries_apply_inside_recipes():
    with pytest.raises(PydanticValidationError, match="cannot be 'apply'"):
        DecisionAdaptationsConfig(mode="bypass", adaptation={"mode": "apply"})


def test_entrypoint_identifiers_are_trimmed_and_aliases_deduplicated():
    entrypoint = Entrypoint.model_validate(
        {
            "name": "balanced",
            "aliases": [" amd/rocm-v1-balanced ", "amd/rocm-v1-balanced"],
            "recipe": "balanced",
            "assignments": {"route": {"models": [{"model": "model-a"}]}},
        }
    )

    assert entrypoint.aliases == ["amd/rocm-v1-balanced"]
    with pytest.raises(PydanticValidationError, match="must be strings"):
        Entrypoint.model_validate(
            {
                "name": "balanced",
                "aliases": ["amd/rocm-v1-balanced", 123],
                "recipe": "balanced",
                "assignments": {"route": {"models": [{"model": "model-a"}]}},
            }
        )


def test_entrypoint_assignments_round_trip_in_order():
    document = recipe_config().model_dump(mode="json", by_alias=True)
    model_b = model_value("model-b", 8001)
    model_b["card"]["reasoning"] = {
        "type": "reasoning_effort",
        "efforts": ["medium", "high"],
    }
    document["models"].append(model_b)
    decision_name = document["recipes"][0]["document"]["decisions"][0]["name"]
    document["entrypoints"][0]["assignments"] = {
        decision_name: {
            "models": [
                {"model": model_b["name"], "reasoning": {"enabled": True}},
                {
                    "model": document["models"][0]["name"],
                    "reasoning": {"enabled": False},
                },
            ]
        }
    }

    config = _user_config(document)
    refs = config.entrypoints[0].assignments[decision_name].models

    assert [ref.model for ref in refs] == [
        model_b["name"],
        document["models"][0]["name"],
    ]
    assert refs[1].reasoning is None
    assert "reasoning" not in refs[1].model_dump(mode="json", exclude_none=True)
    assert not any(
        "Entrypoint assignment" in error.message
        for error in validate_user_config(config)
    )


def test_entrypoint_reasoning_requires_model_capability():
    document = recipe_config().model_dump(mode="json", by_alias=True)
    decision_name = document["recipes"][0]["document"]["decisions"][0]["name"]
    assignment = document["entrypoints"][0]["assignments"][decision_name]
    assignment["models"][0]["reasoning"] = {"enabled": True}

    errors = validate_user_config(_user_config(document))

    assert any(
        "does not support reasoning controls" in error.message for error in errors
    )


def test_entrypoint_reasoning_effort_must_be_declared_by_model():
    document = recipe_config().model_dump(mode="json", by_alias=True)
    document["models"][0]["card"]["reasoning"] = {
        "type": "reasoning_effort",
        "efforts": ["medium"],
    }
    decision_name = document["recipes"][0]["document"]["decisions"][0]["name"]
    assignment = document["entrypoints"][0]["assignments"][decision_name]
    assignment["models"][0]["reasoning"] = {
        "enabled": True,
        "effort": "high",
    }

    errors = validate_user_config(_user_config(document))

    assert any(
        "does not support reasoning effort 'high'" in error.message for error in errors
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
    first.document.signals.keywords = [
        KeywordSignal(
            name="shared-keyword",
            operator="OR",
            keywords=["one"],
            case_sensitive=False,
        )
    ]
    second = first.model_copy(deep=True)
    second.name = "second"
    second.document.signals.keywords[0].keywords = ["two"]
    config.recipes.append(second)

    errors = validate_user_config(config)

    assert not any("shared-keyword" in error.message for error in errors)


def test_signal_references_cannot_cross_recipe_boundaries():
    config = recipe_config()
    config.recipes[0].document.decisions[0].rules.conditions = [
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
