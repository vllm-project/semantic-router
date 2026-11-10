"""Tests for model selection algorithm configuration.

Tests for CLI algorithm support as required by PR #1196 review.
Addresses @Xunzhuo's comment: "we should have a detailed test in the CLI PR"
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.algorithms import (  # noqa: E402
    EXECUTOR_ALGORITHM_TYPES,
    AlgorithmConfig,
    FusionAlgorithmConfig,
    MultiFactorObjectiveConfig,
    MultiFactorSelectionConfig,
    QualityEvidenceConfig,
    RatingsAlgorithmConfig,
    ReMoMAlgorithmConfig,
    WorkflowFinalConfig,
    WorkflowPlannerConfig,
    WorkflowRoleConfig,
    WorkflowsAlgorithmConfig,
)
from cli.config_algorithm_forms import removed_algorithm_forms  # noqa: E402

CHAT_ESCALATE = {"accept": {"confidence": {"method": "avg_logprob", "gte": 0.7}}}


class TestAlgorithmConfigTypes:
    """Test algorithm type validation."""

    def test_executor_types(self):
        """The executors call several candidates and return one response."""
        assert {
            "escalate",
            "fusion",
            "remom",
            "ratings",
            "workflows",
        } == EXECUTOR_ALGORITHM_TYPES
        config = AlgorithmConfig(type="escalate", escalate=CHAT_ESCALATE)
        assert config.escalate == CHAT_ESCALATE
        for algo_type in ["ratings", "remom", "fusion", "workflows"]:
            config = AlgorithmConfig(type=algo_type)
            assert config.type == algo_type

    def test_selector_types(self):
        """The selectors choose one candidate before calling it."""
        for algo_type in ["static", "multi_factor", "random"]:
            config = AlgorithmConfig(type=algo_type)
            assert config.type == algo_type

    @pytest.mark.parametrize(
        "algo_type",
        [
            "confidence",
            "cascade",
            "prompt",
            "router_dc",
            "automix",
            "hybrid",
            "latency_aware",
            "knn",
            "kmeans",
            "svm",
            "mlp",
        ],
    )
    def test_removed_types_point_to_migrate(self, algo_type):
        document = {
            "recipes": [
                {
                    "name": "default",
                    "decisions": [{"name": "route", "algorithm": {"type": algo_type}}],
                }
            ]
        }
        (line,) = removed_algorithm_forms(document)
        assert line.startswith(
            "recipes[default].decisions[route].algorithm: removed_algorithm_type: "
            f"algorithm type {algo_type} is removed; run `vllm-sr config migrate` ("
        )

    def test_removed_types_are_not_extension_algorithms(self):
        with pytest.raises(PydanticValidationError):
            AlgorithmConfig(type="confidence", confidence={"threshold": 0.5})

    def test_executor_settings_have_no_default(self):
        config = AlgorithmConfig(type="fusion")
        assert config.on_error is None
        assert config.budget is None

    def test_executor_reads_shared_settings(self):
        config = AlgorithmConfig(
            type="ratings",
            on_error="fail",
            budget={"deadline": "30s", "max_calls": 2},
        )
        assert config.on_error == "fail"
        assert config.budget == {"deadline": "30s", "max_calls": 2}

    @pytest.mark.parametrize(
        "settings",
        [{"on_error": "skip"}, {"budget": {"deadline": "30s", "max_calls": 2}}],
    )
    def test_selector_rejects_executor_settings(self, settings):
        with pytest.raises(PydanticValidationError, match="apply to executors"):
            AlgorithmConfig(type="static", **settings)

    def test_escalate_block_is_required(self):
        with pytest.raises(PydanticValidationError, match="requires escalate"):
            AlgorithmConfig(type="escalate")

    def test_escalate_block_is_checked_against_the_router_schema(self):
        AlgorithmConfig(
            type="escalate",
            budget={"deadline": "3s", "max_calls": 2},
            escalate={
                "stages": [
                    {
                        "model": "kai",
                        "accept": {
                            "native": [
                                {
                                    "question_type": "choice",
                                    "field": "top_probability",
                                    "gte": 0.6,
                                }
                            ]
                        },
                    },
                    {"model": "vega"},
                ]
            },
        )
        with pytest.raises(PydanticValidationError, match="EscalateAlgorithmConfig"):
            AlgorithmConfig(
                type="escalate",
                escalate={"escalation_order": "size", **CHAT_ESCALATE},
            )

    def test_minimum_candidates_must_be_positive(self):
        config = AlgorithmConfig(type="static", minimum_candidates=2)
        assert config.minimum_candidates == 2

        with pytest.raises(PydanticValidationError):
            AlgorithmConfig(type="static", minimum_candidates=0)

    def test_a_registered_algorithms_block_is_kept_for_the_router(self):
        config = AlgorithmConfig.model_validate(
            {"type": "custom_pick", "custom_pick": {"pick": "b"}}
        )
        assert config.type == "custom_pick"
        dumped = config.model_dump(exclude_none=True)
        assert dumped["custom_pick"] == {"pick": "b"}
        assert "extensions" not in dumped

    def test_an_unknown_type_without_its_block_is_rejected(self):
        with pytest.raises(PydanticValidationError):
            AlgorithmConfig(type="custom_pick")
        with pytest.raises(PydanticValidationError):
            AlgorithmConfig.model_validate({"type": "static", "custom_pick": {}})


class TestMultiFactorSelectionConfig:
    """Test multi-factor selection configuration."""

    def test_slo_and_weight_fields(self):
        config = MultiFactorSelectionConfig(
            weights={"quality": 0.4, "latency": 0.2, "cost": 0.2, "load": 0.2},
            slo={"max_tpot_ms": 200, "max_ttft_ms": 800, "max_cost_per_1m": 5.0},
            quality={
                "index": "vllm-sr/coding@1.0.0",
                "on_missing": "disable_quality",
            },
            latency_percentile=95,
            on_no_candidates="cheapest",
        )
        assert config.weights.quality == 0.4
        assert config.slo.max_ttft_ms == 800
        assert config.quality.index == "vllm-sr/coding@1.0.0"
        assert config.quality.on_missing == "disable_quality"
        assert config.latency_percentile == 95

    def test_quality_evidence_defaults_to_strict_missingness(self):
        quality = QualityEvidenceConfig(index="vllm-sr/intelligence@1.0.0")
        assert quality.on_missing == "exclude"

    @pytest.mark.parametrize(
        "quality",
        [
            {"index": " vllm-sr/coding@1.0.0"},
            {"index": "vllm-sr/coding@1.0.0", "on_missing": "impute"},
        ],
    )
    def test_quality_evidence_rejects_ambiguous_contracts(self, quality):
        with pytest.raises(PydanticValidationError):
            MultiFactorSelectionConfig(quality=quality)


class TestReMoMAlgorithmConfig:
    """Test ReMoM (Reasoning for Mixture of Models) configuration."""

    def test_required_breadth_schedule(self):
        """Test that breadth_schedule is required."""
        config = ReMoMAlgorithmConfig(breadth_schedule=[32, 4])
        assert config.breadth_schedule == [32, 4]

    def test_default_values(self):
        """Test ReMoM config default values."""
        config = ReMoMAlgorithmConfig(breadth_schedule=[32])
        assert config.model_distribution == "weighted"
        assert config.temperature == 1.0
        assert config.include_reasoning is False
        assert config.compaction_strategy == "full"

    def test_missing_breadth_schedule(self):
        """Test that breadth_schedule cannot be empty."""
        # breadth_schedule is required and must be provided
        with pytest.raises(PydanticValidationError):
            ReMoMAlgorithmConfig()  # Missing required field

    def test_model_distribution_validation(self):
        """Test ReMoM model distribution enum validation."""
        config = ReMoMAlgorithmConfig(
            breadth_schedule=[3, 2],
            model_distribution="round_robin",
        )
        assert config.model_distribution == "round_robin"

        with pytest.raises(PydanticValidationError):
            ReMoMAlgorithmConfig(
                breadth_schedule=[3, 2],
                model_distribution="uniform",
            )

    def test_quorum_and_timeout_controls(self):
        """Test ReMoM long-tail controls."""
        config = ReMoMAlgorithmConfig(
            breadth_schedule=[3],
            synthesis_model="model-b",
            max_completion_tokens=768,
            round_timeout_seconds=120,
            min_successful_responses=2,
        )
        assert config.synthesis_model == "model-b"
        assert config.max_completion_tokens == 768
        assert config.round_timeout_seconds == 120
        assert config.min_successful_responses == 2

        with pytest.raises(PydanticValidationError):
            ReMoMAlgorithmConfig(
                breadth_schedule=[3],
                max_completion_tokens=0,
            )

        with pytest.raises(PydanticValidationError):
            ReMoMAlgorithmConfig(
                breadth_schedule=[3],
                round_timeout_seconds=0,
            )


class TestFusionAlgorithmConfig:
    """Test Fusion multi-model deliberation configuration."""

    def test_default_values(self):
        config = FusionAlgorithmConfig()
        assert config.analysis_mode == "separate"
        assert config.include_analysis is True
        assert config.include_intermediate_responses is True
        assert config.judge_prompt_version == "fusion-v1"

    def test_custom_panel_and_judge(self):
        config = FusionAlgorithmConfig(
            model="judge-model",
            analysis_models=["panel-a", "panel-b"],
            analysis_overrides=[
                {
                    "model": "panel-a",
                    "temperature": 0.2,
                    "max_completion_tokens": 384,
                }
            ],
            max_concurrent=2,
            max_completion_tokens=512,
            temperature=0.2,
        )
        assert config.model == "judge-model"
        assert config.analysis_models == ["panel-a", "panel-b"]
        assert config.analysis_overrides is not None
        assert config.analysis_overrides[0].model == "panel-a"
        assert config.max_concurrent == 2
        assert config.max_completion_tokens == 512
        assert config.temperature == 0.2

    @pytest.mark.parametrize("analysis_mode", ["separate", "one_call", "none"])
    def test_analysis_mode_values(self, analysis_mode):
        config = FusionAlgorithmConfig(analysis_mode=analysis_mode)
        assert config.analysis_mode == analysis_mode

    def test_analysis_mode_rejects_unknown_value(self):
        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(analysis_mode="automatic")

    @pytest.mark.parametrize("analysis_mode", ["one_call", "none"])
    def test_analysis_mode_rejects_unused_analysis_template(self, analysis_mode):
        with pytest.raises(PydanticValidationError, match="analysis_template requires"):
            FusionAlgorithmConfig(
                analysis_mode=analysis_mode,
                analysis_template="compare {{responses}}",
            )

    def test_positive_limits(self):
        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(max_concurrent=0)

        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(max_completion_tokens=0)

        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(round_timeout_seconds=0)

        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(min_successful_responses=0)

        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(
                analysis_overrides=[{"model": "panel-a", "max_completion_tokens": 0}]
            )

    def test_analysis_override_models_are_unique_and_non_empty(self):
        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(analysis_overrides=[{"model": "  "}])

        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(
                analysis_overrides=[
                    {"model": "panel-a"},
                    {"model": " panel-a "},
                ]
            )

    def test_quorum_and_timeout_controls(self):
        config = FusionAlgorithmConfig(
            round_timeout_seconds=90,
            min_successful_responses=2,
        )
        assert config.round_timeout_seconds == 90
        assert config.min_successful_responses == 2

    def test_on_error_is_the_algorithms_setting(self):
        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(on_error="fail")

    def test_grounding_defaults(self):
        config = FusionAlgorithmConfig(grounding={"enabled": True})
        assert config.grounding.enabled is True
        assert config.grounding.reference == "hybrid"
        assert config.grounding.policy == "weight"
        assert config.grounding.min_score == 0.0
        assert config.grounding.min_keep == 1
        assert config.grounding.contradiction_penalty == 1.0
        assert config.grounding.on_error == "skip"

    def test_grounding_bounds_match_go_validator(self):
        # min_score in [0, 1]
        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(grounding={"min_score": 1.5})
        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(grounding={"min_score": -0.1})
        # min_keep >= 0
        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(grounding={"min_keep": -1})
        # penalty >= 0
        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(grounding={"contradiction_penalty": -1})
        # reference enum + on_error enum
        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(grounding={"reference": "elsewhere"})
        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(grounding={"on_error": "ignore"})
        # policy enum
        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(grounding={"policy": "drop"})


class TestAlgorithmConfigIntegration:
    """Integration tests for AlgorithmConfig with nested configs."""

    def test_ratings_algorithm_config(self):
        """Test AlgorithmConfig with ratings looper config."""
        config = AlgorithmConfig(
            type="ratings",
            ratings=RatingsAlgorithmConfig(max_concurrent=3),
        )
        assert config.type == "ratings"
        assert config.ratings.max_concurrent == 3

    def test_multi_factor_algorithm_config(self):
        """Test AlgorithmConfig with multi_factor selection config."""
        config = AlgorithmConfig(
            type="multi_factor",
            multi_factor=MultiFactorSelectionConfig(
                weights={"quality": 0.4, "latency": 0.2, "cost": 0.2, "load": 0.2},
                quality={"index": "vllm-sr/intelligence@1.0.0"},
            ),
        )
        assert config.type == "multi_factor"
        assert config.multi_factor.weights.quality == 0.4
        assert config.multi_factor.quality.on_missing == "exclude"

    def test_multi_factor_lexicographic_objective(self):
        config = AlgorithmConfig(
            type="multi_factor",
            multi_factor=MultiFactorSelectionConfig(
                objective=MultiFactorObjectiveConfig(
                    strategy="lexicographic",
                    priorities=[
                        {"factor": "quality", "tolerance": 0.02},
                        {"factor": "cost"},
                    ],
                ),
                quality={
                    "index": "vllm-sr/intelligence@1.0.0",
                    "min_coverage": 1.0,
                    "min_score": 40,
                },
            ),
        )

        assert config.multi_factor.objective.priorities[0].factor == "quality"
        assert config.multi_factor.quality.min_coverage == 1.0
        assert config.multi_factor.quality.min_score == 40

    @pytest.mark.parametrize(
        "multi_factor",
        [
            {
                "objective": {
                    "strategy": "lexicographic",
                    "priorities": [{"factor": "quality"}],
                },
                "weights": {"quality": 1},
            },
            {
                "objective": {
                    "strategy": "lexicographic",
                    "priorities": [
                        {"factor": "cost"},
                        {"factor": "cost"},
                    ],
                }
            },
            {
                "quality": {
                    "index": "vllm-sr/intelligence@1.0.0",
                    "on_missing": "disable_quality",
                    "min_score": 40,
                }
            },
        ],
    )
    def test_multi_factor_rejects_ambiguous_objectives(self, multi_factor):
        with pytest.raises(PydanticValidationError):
            MultiFactorSelectionConfig.model_validate(multi_factor)

    def test_workflows_algorithm_config(self):
        """Test AlgorithmConfig with workflows dynamic config."""
        config = AlgorithmConfig(
            type="workflows",
            workflows=WorkflowsAlgorithmConfig(
                mode="dynamic",
                planner=WorkflowPlannerConfig(
                    model="qwen-coordinator",
                    max_completion_tokens=1024,
                ),
                max_steps=6,
                max_parallel=3,
                round_timeout_seconds=90,
                min_successful_responses=2,
            ),
        )
        assert config.type == "workflows"
        assert config.workflows.mode == "dynamic"
        assert config.workflows.planner.model == "qwen-coordinator"
        assert config.workflows.planner.max_completion_tokens == 1024
        assert config.workflows.max_steps == 6
        assert config.workflows.round_timeout_seconds == 90
        assert config.workflows.min_successful_responses == 2

    def test_workflows_planner_positive_limits(self):
        with pytest.raises(PydanticValidationError):
            WorkflowPlannerConfig(model="qwen-coordinator", max_completion_tokens=0)

        with pytest.raises(PydanticValidationError):
            WorkflowsAlgorithmConfig(round_timeout_seconds=0)

        with pytest.raises(PydanticValidationError):
            WorkflowsAlgorithmConfig(min_successful_responses=0)

    def test_workflows_static_roles_config(self):
        """Test AlgorithmConfig with workflows static role config."""
        config = AlgorithmConfig(
            type="workflows",
            workflows=WorkflowsAlgorithmConfig(
                mode="static",
                roles=[
                    WorkflowRoleConfig(name="thinker", models=["worker-a"]),
                    WorkflowRoleConfig(name="verifier", models=["worker-b"]),
                ],
                final=WorkflowFinalConfig(model="worker-b"),
            ),
        )
        assert config.workflows.roles is not None
        assert config.workflows.roles[0].name == "thinker"
        assert config.workflows.roles[0].models == ["worker-a"]
        assert config.workflows.final is not None
        assert config.workflows.final.model == "worker-b"

    def test_learning_algorithm_type_specific_blocks_are_rejected(self):
        """Learning systems live under global.router.learning."""
        for removed_type, removed_block in {
            "session_aware": {"base_method": "static"},
            "elo": {"k_factor": 48.0},
            "rl_driven": {"storage_path": "/tmp/rl.json"},
            "gmtrouter": {"num_layers": 3},
            "bandit": {"strategy": "thompson"},
            "personalization": {"per_user": True},
        }.items():
            with pytest.raises(PydanticValidationError):
                AlgorithmConfig(
                    type=removed_type,
                    **{removed_type: removed_block},
                )

    def test_removed_algorithm_type_specific_blocks_are_rejected(self):
        """Thompson and Router-R1 are Router Learning or future execution modes."""
        with pytest.raises(PydanticValidationError):
            AlgorithmConfig(
                type="thompson",
                thompson={"per_user": True},
            )

        with pytest.raises(PydanticValidationError):
            AlgorithmConfig(
                type="router_r1",
                router_r1={"router_endpoint": "http://localhost:8080"},
            )

    def test_remom_algorithm_config(self):
        """Test AlgorithmConfig with ReMoM looper."""
        config = AlgorithmConfig(
            type="remom",
            remom=ReMoMAlgorithmConfig(
                breadth_schedule=[32, 8, 2],
                model_distribution="equal",
            ),
        )
        assert config.type == "remom"
        assert config.remom.breadth_schedule == [32, 8, 2]
        assert config.remom.model_distribution == "equal"

    def test_fusion_algorithm_config(self):
        """Test AlgorithmConfig with Fusion looper."""
        config = AlgorithmConfig(
            type="fusion",
            fusion=FusionAlgorithmConfig(
                model="judge-model",
                analysis_models=["panel-a", "panel-b"],
            ),
        )
        assert config.type == "fusion"
        assert config.fusion.model == "judge-model"
        assert config.fusion.analysis_models == ["panel-a", "panel-b"]
