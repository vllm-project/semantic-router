from pathlib import Path

import pytest
import yaml
from cli.algorithms import (
    AlgorithmConfig,
    FusionAlgorithmConfig,
    ReMoMAlgorithmConfig,
    WorkflowPlannerConfig,
    WorkflowsAlgorithmConfig,
)
from cli.main import main
from cli.models import (
    Condition,
    DecisionAdaptationsConfig,
    Domain,
    Entrypoint,
    EvaluationRecord,
    KeywordSignal,
    Listener,
    LoRAAdapter,
    ProjectionMapping,
    ProjectionMappingOutput,
    ProjectionScore,
    ProjectionScoreInput,
    Reasoning,
    UserConfig,
)
from cli.validator import validate_user_config
from cli.validator_recipe_contracts import effective_entrypoints
from click.testing import CliRunner
from pydantic import ValidationError as PydanticValidationError


def recipe_config(*, recipe_model: str = "model-a", recipe_name: str = "private"):
    return UserConfig.model_validate(
        {
            "version": "v0.5",
            "providers": {
                "defaults": {"model": "model-a"},
                "models": [
                    {
                        "name": "model-a",
                        "backend_refs": [
                            {"endpoint": "127.0.0.1:8000", "provider": "vllm"}
                        ],
                    }
                ],
            },
            "entrypoints": [
                {"model_names": ["amd/rocm-v1-private"], "recipe": recipe_name},
                {"model_names": ["vllm-sr/auto"], "recipe": "default", "default": True},
            ],
            "recipes": [
                {
                    "name": "private",
                    "decisions": [
                        {
                            "name": "private-route",
                            "description": "private",
                            "priority": 1,
                            "tier": 7,
                            "rules": {"operator": "AND", "conditions": []},
                            "modelRefs": [{"model": recipe_model}],
                        }
                    ],
                },
                {
                    "name": "default",
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
            ],
        }
    )


def test_recipe_model_references_are_validated():
    errors = validate_user_config(recipe_config(recipe_model="missing-model"))

    assert any("unknown model 'missing-model'" in error.message for error in errors)


@pytest.mark.parametrize(
    ("ref", "message"),
    [
        ({"model": "ghost"}, "Decision 'support' references unknown model 'ghost'"),
        (
            {"model": "general", "lora_name": "undeclared"},
            "Decision 'support' references unknown LoRA 'undeclared' for model 'general'",
        ),
        ({"model": "general", "lora_name": "declared"}, None),
        ({"model": "declared"}, None),
    ],
)
def test_candidate_iteration_models_are_checked_as_the_router_checks_them(ref, message):
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "providers": {
                "defaults": {"model": "general"},
                "models": [
                    {
                        "name": "general",
                        "backend_refs": [
                            {"endpoint": "127.0.0.1:8000", "provider": "vllm"}
                        ],
                        "loras": [{"name": "declared"}],
                    }
                ],
            },
            "recipes": [
                {
                    "name": "default",
                    "decisions": [
                        {
                            "name": "support",
                            "priority": 1,
                            "rules": {"operator": "AND", "conditions": []},
                            "modelRefs": [{"model": "general"}],
                            "candidateIterations": [
                                {
                                    "variable": "candidate",
                                    "source": "models",
                                    "models": [ref],
                                    "outputs": [
                                        {"type": "model", "value": "candidate"}
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )

    errors = [
        (error.field, error.message)
        for error in validate_user_config(config, log_summary=False)
    ]

    field = "recipes.default.decisions.support.candidateIterations[0].models"
    assert errors == ([(field, message)] if message else [])


def test_catalog_model_cannot_override_reasoning_binding():
    config = recipe_config()
    config.providers.models[0].catalog = "vllm-sr/mom-v1-lite"
    config.providers.models[0].reasoning = Reasoning(family="qwen3")

    errors = validate_user_config(config)

    assert any(
        error.field == "providers.models.model-a.reasoning"
        and "inherits reasoning" in error.message
        for error in errors
    )


def _catalog_backed_qwen_config() -> UserConfig:
    config = recipe_config()
    config.providers.models[0].catalog = "qwen/qwen3.8-27b"
    return config


@pytest.mark.parametrize(
    ("overlay", "field", "added"),
    [
        (
            {
                "capabilities": [
                    "chat",
                    "reasoning",
                    "tools",
                    "structured_output",
                    "text",
                ]
            },
            "capabilities",
            "text",
        ),
        (
            {"modalities": {"input": ["text", "audio"], "output": ["text"]}},
            "modalities",
            "audio",
        ),
    ],
)
def test_catalog_card_widening_is_rejected_before_serve(overlay, field, added):
    config = _catalog_backed_qwen_config()
    card = config.providers.models[0]
    for name, value in overlay.items():
        setattr(card, name, value)

    errors = validate_user_config(config, log_summary=False)

    assert any(
        error.field == f"providers.models[{card.name}].{field}"
        and "widen a built-in claim" in error.message
        and added in error.message
        for error in errors
    )


def test_catalog_card_claims_may_be_narrowed():
    config = _catalog_backed_qwen_config()
    config.providers.models[0].capabilities = ["chat", "reasoning"]
    config.providers.models[0].modalities = {"input": ["text"], "output": ["text"]}

    errors = validate_user_config(config, log_summary=False)

    assert not any(
        error.field and error.field.startswith("providers.models[") for error in errors
    )


def test_custom_card_sharing_builtin_name_may_declare_its_own_claims():
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "providers": {
                "defaults": {"model": "qwen/qwen3.8-27b"},
                "models": [
                    {
                        "name": "qwen/qwen3.8-27b",
                        "backend_refs": [
                            {"endpoint": "127.0.0.1:8000", "provider": "vllm"}
                        ],
                        "capabilities": ["chat", "text"],
                    }
                ],
            },
            "recipes": [{"name": "default"}],
        }
    )

    errors = validate_user_config(config, log_summary=False)

    assert not any("widen a built-in claim" in error.message for error in errors)


def test_config_validate_reports_unverified_catalog_claim(tmp_path: Path):
    config = _catalog_backed_qwen_config()
    config.providers.models[0].capabilities = [
        "chat",
        "reasoning",
        "tools",
        "structured_output",
        "text",
    ]
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            config.model_dump(mode="json", by_alias=True, exclude_none=True)
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["config", "validate", "--config", str(path)])

    assert result.exit_code != 0
    assert "capabilities widen a built-in claim" in result.output
    assert "capabilities" in result.output


def test_inline_reasoning_allows_an_implicit_mode_contract():
    reasoning = Reasoning(
        type="chat_template_kwargs",
        parameter="enable_thinking",
    )

    assert reasoning.modes == []
    assert reasoning.default_mode is None


def test_inline_reasoning_keeps_disable_sentinel_separate_from_effort_levels():
    reasoning = Reasoning(
        type="reasoning_effort",
        parameter="reasoning_effort",
        levels=["low", "high"],
        default="high",
        disabled="none",
        modes=["enabled", "disabled"],
        default_mode="enabled",
    )

    assert reasoning.disabled == "none"
    assert reasoning.levels == ["low", "high"]


def test_inline_effort_flags_ignore_a_disabled_sentinel_level():
    reasoning = Reasoning(
        type="reasoning_effort",
        parameter="reasoning_effort",
        activation_parameter="enable_thinking",
        effort_flags={"low": "low_effort", "medium": "medium_effort"},
        levels=["none", "low", "medium", "high"],
        default="high",
        disabled="none",
        modes=["enabled", "disabled"],
        default_mode="enabled",
    )

    assert reasoning.effort_flags == {
        "low": "low_effort",
        "medium": "medium_effort",
    }


def test_lora_adapter_is_declared_on_its_provider_model():
    config = recipe_config()
    config.providers.models[0].loras = [LoRAAdapter(name="general-expert")]

    errors = validate_user_config(config)

    assert not any(
        error.field and error.field.startswith("providers.models") for error in errors
    )


def test_routing_only_model_card_can_supply_default_model():
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "providers": {
                "defaults": {"model": "private-model"},
                "models": [
                    {
                        "name": "private-model",
                        "description": "Metadata-only routing model",
                    }
                ],
            },
            "recipes": [{"name": "default"}],
        }
    )

    assert validate_user_config(config, log_summary=False) == []


def test_routing_only_decision_refs_remain_external_gateway_metadata():
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "providers": {"models": [{"name": "private-model"}]},
            "recipes": [
                {
                    "name": "default",
                    "decisions": [
                        {
                            "name": "external-route",
                            "priority": 100,
                            "modelRefs": [{"model": "private-model"}],
                        }
                    ],
                }
            ],
        }
    )

    assert validate_user_config(config, log_summary=False) == []


def test_blank_provider_model_identity_is_rejected():
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "providers": {"models": [{"name": " "}]},
            "recipes": [{"name": "default"}],
        }
    )

    errors = validate_user_config(config, log_summary=False)

    assert any(
        error.field == "providers.models[0].name"
        and error.message == "Provider model name cannot be empty"
        for error in errors
    )


def test_name_only_provider_model_describes_a_routing_target():
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "listeners": [],
            "providers": {"models": [{"name": "bare"}]},
            "recipes": [{"name": "default"}],
        }
    )

    errors = validate_user_config(config, log_summary=False)

    assert not any(error.field == "providers.models.bare" for error in errors)


def test_empty_pricing_block_is_a_name_only_provider_model():
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "listeners": [],
            "providers": {"models": [{"name": "bare", "pricing": {}}]},
            "recipes": [{"name": "default"}],
        }
    )

    errors = validate_user_config(config, log_summary=False)

    assert not any(error.field == "providers.models.bare" for error in errors)


def test_explicit_zero_pricing_counts_as_provider_model_metadata():
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "listeners": [],
            "providers": {
                "models": [
                    {
                        "name": "free-model",
                        "pricing": {"prompt_per_1m": 0, "completion_per_1m": 0},
                    }
                ]
            },
            "recipes": [{"name": "default"}],
        }
    )

    assert validate_user_config(config, log_summary=False) == []


def test_operator_evaluation_requires_versioned_identity_and_finite_metrics():
    with pytest.raises(PydanticValidationError, match="string_pattern_mismatch"):
        EvaluationRecord(model="private", benchmark="support", metrics={"score": 0.8})
    with pytest.raises(PydanticValidationError, match="named finite numbers"):
        EvaluationRecord(
            model="private",
            benchmark="acme/support@1",
            metrics={"score": float("nan")},
        )


def test_operator_evaluation_preserves_profile_and_reasoning_effort():
    evaluation = EvaluationRecord(
        model="private",
        benchmark="acme/support@1.0.0",
        benchmark_profile="published-standard",
        reasoning_effort="high",
        metrics={"score": 0.82},
    )

    assert evaluation.benchmark_profile == "published-standard"
    assert evaluation.reasoning_effort == "high"


def test_omitted_provider_defaults_do_not_invent_reasoning_effort():
    config = UserConfig.model_validate({"version": "v0.5"})

    assert config.providers.defaults.reasoning_effort is None
    assert config.providers.defaults.model_dump(exclude_none=True) == {}


def test_recipe_decision_tier_survives_schema_parse():
    config = recipe_config()

    assert config.recipes[0].decisions[0].tier == 7


def test_entrypoints_must_reference_known_recipe():
    errors = validate_user_config(recipe_config(recipe_name="missing-recipe"))

    assert any("unknown recipe 'missing-recipe'" in error.message for error in errors)
    assert any(
        error.hint and "name of a recipe" in error.hint
        for error in errors
        if "unknown recipe 'missing-recipe'" in error.message
    )


def test_entrypoint_names_cannot_collide_with_provider_models():
    config = recipe_config()
    config.entrypoints[0].model_names = ["model-a"]

    errors = validate_user_config(config)

    assert any(
        "conflicts with a configured backend model" in error.message for error in errors
    )
    assert any(
        error.hint and "distinct entrypoint model name" in error.hint
        for error in errors
        if "conflicts with a configured backend model" in error.message
    )


def test_a_single_recipe_may_be_named_default():
    config = recipe_config()
    config.recipes = [config.recipes[0]]
    config.recipes[0].name = "default"
    config.entrypoints = [config.entrypoints[0]]
    config.entrypoints[0].recipe = "default"

    errors = validate_user_config(config)

    assert not any(
        "Duplicate recipe name 'default'" in error.message for error in errors
    )
    assert not any("unknown recipe 'default'" in error.message for error in errors)


def test_two_recipes_named_default_are_rejected():
    config = recipe_config()
    config.default_recipe.decisions = []
    config.default_recipe.strategy = "priority"
    config.recipes[0].name = "default"
    config.entrypoints[0].recipe = "default"

    errors = validate_user_config(config)

    assert any("Duplicate recipe name 'default'" in error.message for error in errors)
    assert any(
        error.hint and "unique name" in error.hint
        for error in errors
        if "Duplicate recipe name 'default'" in error.message
    )


def test_decision_adaptation_mode_boundaries_apply_inside_recipes():
    with pytest.raises(PydanticValidationError, match="cannot be 'apply'"):
        DecisionAdaptationsConfig(
            mode="bypass",
            adaptation={"mode": "apply"},
        )


def test_recipe_remom_synthesis_model_must_be_in_model_refs():
    config = recipe_config()
    config.recipes[0].decisions[0].algorithm = AlgorithmConfig(
        type="remom",
        remom=ReMoMAlgorithmConfig(
            breadth_schedule=[2],
            synthesis_model="missing-model",
        ),
    )

    errors = validate_user_config(config)

    assert any("synthesis_model 'missing-model'" in error.message for error in errors)


def test_recipe_fusion_rejects_quorum_above_panel_size():
    config = recipe_config()
    config.recipes[0].decisions[0].algorithm = AlgorithmConfig(
        type="fusion",
        fusion=FusionAlgorithmConfig(min_successful_responses=2),
    )

    errors = validate_user_config(config)

    assert any("exceeds panel size 1" in error.message for error in errors)


def test_recipe_workflow_rejects_quorum_above_parallelism():
    config = recipe_config()
    config.recipes[0].decisions[0].algorithm = AlgorithmConfig(
        type="workflows",
        workflows=WorkflowsAlgorithmConfig(
            mode="dynamic",
            planner=WorkflowPlannerConfig(model="model-a"),
            max_parallel=1,
            min_successful_responses=2,
        ),
    )

    errors = validate_user_config(config)

    assert any("exceeds max_parallel=1" in error.message for error in errors)


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
    config.default_recipe.signals.keywords = [
        KeywordSignal(
            name="shared-keyword",
            operator="OR",
            keywords=["shared"],
            case_sensitive=False,
        )
    ]
    config.recipes[0].signals.keywords = [
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
    config.default_recipe.signals.keywords = [signal]
    config.recipes[0].signals.keywords = [signal]

    errors = validate_user_config(config)

    assert not any("shared-keyword" in error.message for error in errors)


def test_signal_references_cannot_cross_recipe_boundaries():
    config = recipe_config()
    config.recipes[0].signals.keywords = [
        KeywordSignal(
            name="private-only",
            operator="OR",
            keywords=["private"],
            case_sensitive=False,
        )
    ]
    config.default_recipe.decisions[0].rules.conditions = [
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
    config.default_recipe.signals.keywords = [
        KeywordSignal(
            name="shared-name",
            operator="OR",
            keywords=["shared"],
            case_sensitive=False,
        )
    ]
    config.default_recipe.decisions[0].rules.conditions = [
        Condition(type="authz", name="shared-name")
    ]

    errors = validate_user_config(config)

    assert any(
        "recipe 'default'" in error.message
        and "unknown signal 'shared-name'" in error.message
        for error in errors
    )


def authz_config(reference: str) -> UserConfig:
    return UserConfig.model_validate(
        {
            "version": "v0.5",
            "providers": {
                "defaults": {"model": "model-a"},
                "models": [
                    {
                        "name": "model-a",
                        "backend_refs": [
                            {"endpoint": "127.0.0.1:8000", "provider": "vllm"}
                        ],
                    }
                ],
            },
            "recipes": [
                {
                    "name": "default",
                    "signals": {
                        "role_bindings": [
                            {
                                "name": "admins",
                                "role": "admin",
                                "subjects": [
                                    {"kind": "Group", "name": "platform-admins"}
                                ],
                            }
                        ]
                    },
                    "decisions": [
                        {
                            "name": "admin-route",
                            "priority": 100,
                            "rules": {
                                "operator": "AND",
                                "conditions": [{"type": "authz", "name": reference}],
                            },
                            "modelRefs": [{"model": "model-a"}],
                        }
                    ],
                }
            ],
        }
    )


def test_authz_conditions_reference_the_role_a_binding_grants():
    assert validate_user_config(authz_config("admin"), log_summary=False) == []

    errors = validate_user_config(authz_config("admins"), log_summary=False)

    assert any(
        "recipe 'default'" in error.message
        and "unknown signal 'admins'" in error.message
        for error in errors
    )


def test_nested_signal_conditions_validate_only_leaf_references():
    config = recipe_config()
    config.default_recipe.signals.keywords = [
        KeywordSignal(
            name="nested-keyword",
            operator="OR",
            keywords=["nested"],
            case_sensitive=False,
        )
    ]
    config.default_recipe.decisions[0].rules.conditions = [
        Condition(
            operator="OR",
            conditions=[Condition(type="keyword", name="nested-keyword")],
        )
    ]

    errors = validate_user_config(config)

    assert not any(
        "unsupported signal type 'None'" in error.message for error in errors
    )
    assert not any("nested-keyword" in error.message for error in errors)


def test_decision_names_can_repeat_across_recipes():
    config = recipe_config()
    config.recipes[0].decisions[0].name = "default-route"

    errors = validate_user_config(config)

    assert not any("more than one routing profile" in error.message for error in errors)


def test_orchestration_names_are_ordinary_explicit_entrypoints():
    config = recipe_config()
    config.entrypoints[0].model_names = ["vllm-sr/remom"]
    assert not validate_user_config(config)


def test_nullable_global_sections_do_not_crash_validation():
    config = recipe_config()
    config.global_ = {"router": None, "integrations": None}

    validate_user_config(config)


def test_configured_model_names_keep_exact_collision_semantics():
    config = recipe_config()
    config.providers.models[0].name = " model-a "
    config.entrypoints[0].model_names = ["model-a"]

    errors = validate_user_config(config)

    assert not any(
        error.field == "entrypoints.0.model_names"
        and "configured model" in error.message
        for error in errors
    )


def test_projection_dependencies_cannot_cross_recipe_profiles():
    config = recipe_config()
    config.default_recipe.projections.scores = [
        ProjectionScore(name="base-score", method="weighted_sum", inputs=[])
    ]
    config.recipes[0].projections.scores = [
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
    config.default_recipe.projections.mappings = [
        ProjectionMapping(
            name="default-band",
            source="base-score",
            method="threshold_bands",
            outputs=[output],
        )
    ]
    config.recipes[0].projections.mappings = [
        ProjectionMapping(
            name="recipe-band",
            source="recipe-score",
            method="threshold_bands",
            outputs=[output],
        )
    ]

    errors = validate_user_config(config)

    assert not any("duplicate projection output" in error.message for error in errors)


def test_a_recipe_names_only_the_domains_it_declares():
    config = recipe_config()
    config.default_recipe.signals.domains = [
        Domain(name="top-level-domain", description="Top-level only", labels=["math"])
    ]
    config.recipes[0].decisions[0].rules.conditions = [
        Condition(type="domain", name="undeclared-domain")
    ]

    errors = validate_user_config(config)

    assert any("undeclared-domain" in error.message for error in errors)


def test_domain_references_cannot_cross_recipe_boundaries():
    config = recipe_config()
    config.default_recipe.signals.domains = [
        Domain(name="top-level-domain", description="Top-level only")
    ]
    config.recipes[0].signals.domains = [
        Domain(name="recipe-domain", description="Recipe-owned")
    ]
    config.default_recipe.decisions[0].rules.conditions = [
        Condition(type="domain", name="recipe-domain")
    ]

    errors = validate_user_config(config)

    assert any(
        "recipe 'default'" in error.message and "recipe-domain" in error.message
        for error in errors
    )


def test_domain_names_can_have_different_definitions_across_recipes():
    config = recipe_config()
    config.default_recipe.signals.domains = [
        Domain(name="shared-domain", description="Custom definition")
    ]
    config.recipes[0].decisions[0].rules.conditions = [
        Condition(type="domain", name="shared-domain")
    ]

    errors = validate_user_config(config)

    assert not any("conflicting definitions" in error.message for error in errors)


def test_default_entrypoint_override_and_reset():
    config = recipe_config()
    config.entrypoints = [
        entry for entry in config.entrypoints if entry.recipe != "default"
    ]
    assert effective_entrypoints(config)[0].model_names == ["vllm-sr/auto"]
    config.entrypoints.append(
        Entrypoint(recipe="default", model_names=["our/router", "MoM"], default=True)
    )
    assert [
        entry.model_names
        for entry in effective_entrypoints(config)
        if entry.recipe == "default"
    ] == [["our/router", "MoM"]]


def test_systemone_listener_scope_is_explicit_and_nonempty():
    listener = Listener(
        name="public", address="0.0.0.0", port=8000, models=["chat-only"]
    )
    assert listener.systemone is None
    for models in ([], [""], [" padded "], ["one", "one"]):
        with pytest.raises(PydanticValidationError):
            Listener(
                name="public",
                address="0.0.0.0",
                port=8000,
                systemone={"models": models},
            )
    listener = Listener(
        name="public",
        address="0.0.0.0",
        port=8000,
        models=["chat-only"],
        systemone={"models": ["decision-public"]},
    )
    assert listener.models == ["chat-only"]
    assert listener.systemone.models == ["decision-public"]
