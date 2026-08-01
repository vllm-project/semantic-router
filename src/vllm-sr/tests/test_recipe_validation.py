import pytest
from cli.algorithms import AlgorithmConfig, ReMoMAlgorithmConfig
from cli.models import (
    Condition,
    DecisionAdaptationsConfig,
    Domain,
    Entrypoint,
    KeywordSignal,
    ProjectionMapping,
    ProjectionMappingOutput,
    ProjectionScore,
    ProjectionScoreInput,
    UserConfig,
)
from cli.validator import validate_user_config
from pydantic import ValidationError as PydanticValidationError


def recipe_config(*, recipe_model: str = "model-a", recipe_name: str = "private"):
    return UserConfig.model_validate(
        {
            "version": "v0.3",
            "providers": {
                "defaults": {"default_model": "model-a"},
                "models": [
                    {
                        "name": "model-a",
                        "backend_refs": [{"endpoint": "127.0.0.1:8000"}],
                    }
                ],
            },
            "routing": {
                "modelCards": [{"name": "model-a"}],
                "decisions": [
                    {
                        "name": "default-route",
                        "description": "default",
                        "priority": 1,
                        "rules": {"operator": "AND", "conditions": []},
                        "modelRefs": [{"model": "model-a"}],
                    }
                ],
            },
            "entrypoints": [
                {"model_names": ["amd/rocm-v1-private"], "recipe": recipe_name}
            ],
            "recipes": [
                {
                    "name": "private",
                    "routing": {
                        "decisions": [
                            {
                                "name": "private-route",
                                "description": "private",
                                "priority": 1,
                                "rules": {"operator": "AND", "conditions": []},
                                "modelRefs": [{"model": recipe_model}],
                            }
                        ]
                    },
                }
            ],
        }
    )


def test_recipe_model_references_are_validated():
    errors = validate_user_config(recipe_config(recipe_model="missing-model"))

    assert any("unknown model 'missing-model'" in error.message for error in errors)


def test_entrypoints_must_reference_known_recipe():
    errors = validate_user_config(recipe_config(recipe_name="missing-recipe"))

    assert any("unknown recipe 'missing-recipe'" in error.message for error in errors)


def test_entrypoint_names_cannot_collide_with_provider_models():
    config = recipe_config()
    config.entrypoints[0].model_names = ["model-a"]

    errors = validate_user_config(config)

    assert any("conflicts with a configured model" in error.message for error in errors)


def test_recipes_only_default_profile_is_allowed():
    config = recipe_config()
    config.routing.decisions = []
    config.recipes[0].name = "default"
    config.entrypoints[0].recipe = "default"

    errors = validate_user_config(config)

    assert not any(
        "Duplicate recipe name 'default'" in error.message for error in errors
    )
    assert not any("unknown recipe 'default'" in error.message for error in errors)


def test_explicit_default_recipe_conflicts_with_top_level_strategy():
    config = recipe_config()
    config.routing.decisions = []
    config.routing.strategy = "priority"
    config.recipes[0].name = "default"
    config.entrypoints[0].recipe = "default"

    errors = validate_user_config(config)

    assert any("Duplicate recipe name 'default'" in error.message for error in errors)


def test_decision_adaptation_mode_boundaries_apply_inside_recipes():
    with pytest.raises(PydanticValidationError, match="cannot be 'apply'"):
        DecisionAdaptationsConfig(
            mode="bypass",
            adaptation={"mode": "apply"},
        )


def test_recipe_remom_synthesis_model_must_be_in_model_refs():
    config = recipe_config()
    config.recipes[0].routing.decisions[0].algorithm = AlgorithmConfig(
        type="remom",
        remom=ReMoMAlgorithmConfig(
            breadth_schedule=[2],
            synthesis_model="missing-model",
        ),
    )

    errors = validate_user_config(config)

    assert any("synthesis_model 'missing-model'" in error.message for error in errors)


def test_entrypoint_identifiers_are_trimmed_and_deduplicated():
    entrypoint = Entrypoint.model_validate(
        {
            "model_names": [" amd/rocm-v1-balanced ", "amd/rocm-v1-balanced"],
            "recipe": " balanced ",
        }
    )

    assert entrypoint.model_names == ["amd/rocm-v1-balanced"]
    assert entrypoint.recipe == "balanced"
    with pytest.raises(PydanticValidationError, match="must be strings"):
        Entrypoint.model_validate(
            {
                "model_names": ["amd/rocm-v1-balanced", 123],
                "recipe": "balanced",
            }
        )


def test_duplicate_signal_names_are_isolated_across_recipes():
    config = recipe_config()
    config.routing.signals.keywords = [
        KeywordSignal(
            name="shared-keyword",
            operator="OR",
            keywords=["shared"],
            case_sensitive=False,
        )
    ]
    config.recipes[0].routing.signals.keywords = [
        KeywordSignal(
            name="shared-keyword",
            operator="OR",
            keywords=["conflicting"],
            case_sensitive=False,
        )
    ]

    errors = validate_user_config(config)

    assert not any("shared-keyword" in error.message for error in errors)


def test_identical_signals_can_be_shared_across_recipes():
    config = recipe_config()
    signal = KeywordSignal(
        name="shared-keyword",
        operator="OR",
        keywords=["shared"],
        case_sensitive=False,
    )
    config.routing.signals.keywords = [signal]
    config.recipes[0].routing.signals.keywords = [signal]

    errors = validate_user_config(config)

    assert not any("shared-keyword" in error.message for error in errors)


def test_signal_references_cannot_cross_recipe_boundaries():
    config = recipe_config()
    config.recipes[0].routing.signals.keywords = [
        KeywordSignal(
            name="private-only",
            operator="OR",
            keywords=["private"],
            case_sensitive=False,
        )
    ]
    config.routing.decisions[0].rules.conditions = [
        Condition(type="keyword", name="private-only")
    ]

    errors = validate_user_config(config)

    assert any(
        "recipe 'default'" in error.message
        and "unknown signal 'private-only'" in error.message
        for error in errors
    )


def test_signal_references_cannot_cross_signal_families():
    config = recipe_config()
    config.routing.signals.keywords = [
        KeywordSignal(
            name="shared-name",
            operator="OR",
            keywords=["shared"],
            case_sensitive=False,
        )
    ]
    config.routing.decisions[0].rules.conditions = [
        Condition(type="authz", name="shared-name")
    ]

    errors = validate_user_config(config)

    assert any(
        "recipe 'default'" in error.message
        and "unknown signal 'shared-name'" in error.message
        for error in errors
    )


def test_decision_names_can_repeat_across_recipes():
    config = recipe_config()
    config.recipes[0].routing.decisions[0].name = "default-route"

    errors = validate_user_config(config)

    assert not any("more than one routing profile" in error.message for error in errors)


def test_default_looper_aliases_are_reserved_for_entrypoints():
    config = recipe_config()
    config.entrypoints[0].model_names = ["vllm-sr/remom"]

    errors = validate_user_config(config)

    assert any("reserved alias" in error.message for error in errors)


def test_nullable_global_sections_do_not_crash_validation():
    config = recipe_config()
    config.global_ = {"router": None, "integrations": None}

    validate_user_config(config)


def test_malformed_global_sections_are_reported():
    config = recipe_config()
    config.global_ = {"router": [], "integrations": "bad"}

    errors = validate_user_config(config)

    assert any(error.field == "global.router" for error in errors)
    assert any(error.field == "global.integrations" for error in errors)


def test_malformed_alias_values_are_reported_without_crashing():
    config = recipe_config()
    config.global_ = {
        "router": {"auto_model_names": 123},
        "integrations": {
            "looper": {
                "fusion": {"model_names": "bad"},
            }
        },
    }

    errors = validate_user_config(config)

    assert any(error.field == "global.router.auto_model_names" for error in errors)
    assert any(
        error.field == "global.integrations.looper.fusion.model_names"
        for error in errors
    )


def test_reserved_aliases_are_trimmed_before_collision_checks():
    config = recipe_config()
    config.global_ = {"router": {"auto_model_names": [" amd/custom-auto "]}}
    config.entrypoints[0].model_names = ["amd/custom-auto"]

    errors = validate_user_config(config)

    assert any("reserved alias" in error.message for error in errors)


def test_configured_model_names_keep_exact_collision_semantics():
    config = recipe_config()
    config.providers.models[0].name = " model-a "
    config.routing.model_cards[0].name = " model-a "
    config.entrypoints[0].model_names = ["model-a"]

    errors = validate_user_config(config)

    assert not any(
        error.field == "entrypoints.0.model_names"
        and "configured model" in error.message
        for error in errors
    )


def test_projection_dependencies_cannot_cross_recipe_profiles():
    config = recipe_config()
    config.routing.projections.scores = [
        ProjectionScore(name="base-score", method="weighted_sum", inputs=[])
    ]
    config.recipes[0].routing.projections.scores = [
        ProjectionScore(
            name="derived-score",
            method="weighted_sum",
            inputs=[
                ProjectionScoreInput(
                    type="projection",
                    name="base-score",
                    weight=1,
                    value_source="raw",
                )
            ],
        )
    ]

    errors = validate_user_config(config)

    assert any("undefined score" in error.message for error in errors)


def test_projection_outputs_can_repeat_across_recipes():
    config = recipe_config()
    output = ProjectionMappingOutput(name="shared-band", gte=0)
    config.routing.projections.mappings = [
        ProjectionMapping(
            name="default-band",
            source="base-score",
            method="threshold_bands",
            outputs=[output],
        )
    ]
    config.recipes[0].routing.projections.mappings = [
        ProjectionMapping(
            name="recipe-band",
            source="recipe-score",
            method="threshold_bands",
            outputs=[output],
        )
    ]

    errors = validate_user_config(config)

    assert not any("duplicate projection output" in error.message for error in errors)


def test_domain_auto_generation_is_scoped_per_recipe():
    config = recipe_config()
    config.routing.signals.domains = [
        Domain(
            name="top-level-domain",
            description="Top-level only",
            mmlu_categories=["math"],
        )
    ]
    config.recipes[0].routing.decisions[0].rules.conditions = [
        Condition(type="domain", name="recipe-generated-domain")
    ]

    errors = validate_user_config(config)

    assert not any("recipe-generated-domain" in error.message for error in errors)


def test_domain_references_cannot_cross_recipe_boundaries():
    config = recipe_config()
    config.routing.signals.domains = [
        Domain(name="top-level-domain", description="Top-level only")
    ]
    config.recipes[0].routing.signals.domains = [
        Domain(name="recipe-domain", description="Recipe-owned")
    ]
    config.routing.decisions[0].rules.conditions = [
        Condition(type="domain", name="recipe-domain")
    ]

    errors = validate_user_config(config)

    assert any(
        "recipe 'default'" in error.message and "recipe-domain" in error.message
        for error in errors
    )


def test_domain_names_can_have_different_definitions_across_recipes():
    config = recipe_config()
    config.routing.signals.domains = [
        Domain(name="shared-domain", description="Custom definition")
    ]
    config.recipes[0].routing.decisions[0].rules.conditions = [
        Condition(type="domain", name="shared-domain")
    ]

    errors = validate_user_config(config)

    assert not any("conflicting definitions" in error.message for error in errors)
