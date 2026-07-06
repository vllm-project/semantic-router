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
    AlgorithmConfig,
    AutoMixSelectionConfig,
    FusionAlgorithmConfig,
    HybridSelectionConfig,
    MultiFactorSelectionConfig,
    RatingsAlgorithmConfig,
    ReMoMAlgorithmConfig,
    RouterDCSelectionConfig,
    WorkflowFinalConfig,
    WorkflowPlannerConfig,
    WorkflowRoleConfig,
    WorkflowsAlgorithmConfig,
)


class TestAlgorithmConfigTypes:
    """Test algorithm type validation."""

    def test_valid_looper_types(self):
        """Test that looper algorithm types are accepted."""
        looper_types = ["confidence", "ratings", "remom", "fusion", "workflows"]

        for algo_type in looper_types:
            config = AlgorithmConfig(type=algo_type)
            assert config.type == algo_type

    def test_valid_selection_types(self):
        """Test that selection algorithm types are accepted."""
        selection_types = [
            "static",
            "router_dc",
            "automix",
            "hybrid",
            "knn",
            "kmeans",
            "svm",
            "mlp",
            "multi_factor",
            "latency_aware",
        ]

        for algo_type in selection_types:
            config = AlgorithmConfig(type=algo_type)
            assert config.type == algo_type

    def test_on_error_default(self):
        """Test that on_error defaults to 'skip'."""
        config = AlgorithmConfig(type="static")
        assert config.on_error == "skip"

    def test_on_error_fail(self):
        """Test that on_error can be set to 'fail'."""
        config = AlgorithmConfig(type="static", on_error="fail")
        assert config.on_error == "fail"


class TestRouterDCSelectionConfig:
    """Test RouterDC selection configuration."""

    def test_default_values(self):
        """Test RouterDC config default values."""
        config = RouterDCSelectionConfig()
        assert config.temperature == 0.07
        assert config.dimension_size == 768
        assert config.min_similarity == 0.3
        assert config.use_query_contrastive is True
        assert config.use_model_contrastive is True
        assert config.use_capabilities is True

    def test_temperature_validation(self):
        """Test that temperature must be positive."""
        with pytest.raises(PydanticValidationError):
            RouterDCSelectionConfig(temperature=0.0)  # Must be > 0

    def test_similarity_threshold(self):
        """Test min_similarity range validation."""
        config = RouterDCSelectionConfig(min_similarity=0.5)
        assert config.min_similarity == 0.5

        with pytest.raises(PydanticValidationError):
            RouterDCSelectionConfig(min_similarity=1.5)  # Above 1.0


class TestAutoMixSelectionConfig:
    """Test AutoMix (POMDP) selection configuration."""

    def test_default_values(self):
        """Test AutoMix config default values."""
        config = AutoMixSelectionConfig()
        assert config.verification_threshold == 0.7
        assert config.max_escalations == 2
        assert config.cost_aware_routing is True
        assert config.cost_quality_tradeoff == 0.3
        assert config.discount_factor == 0.95

    def test_verification_threshold_range(self):
        """Test that verification threshold is within 0-1."""
        config = AutoMixSelectionConfig(verification_threshold=0.9)
        assert config.verification_threshold == 0.9

        with pytest.raises(PydanticValidationError):
            AutoMixSelectionConfig(verification_threshold=1.5)


class TestHybridSelectionConfig:
    """Test Hybrid selection configuration."""

    def test_default_weights(self):
        """Test Hybrid config default weights."""
        config = HybridSelectionConfig()
        assert config.experience_weight == 0.3
        assert config.router_dc_weight == 0.3
        assert config.automix_weight == 0.2
        assert config.cost_weight == 0.2

    def test_custom_weights(self):
        """Test Hybrid config with custom weights."""
        config = HybridSelectionConfig(
            experience_weight=0.5,
            router_dc_weight=0.3,
            automix_weight=0.1,
            cost_weight=0.1,
        )
        # Weights should sum to 1.0
        total = (
            config.experience_weight
            + config.router_dc_weight
            + config.automix_weight
            + config.cost_weight
        )
        assert abs(total - 1.0) < 0.01


class TestMultiFactorSelectionConfig:
    """Test multi-factor selection configuration."""

    def test_slo_and_weight_fields(self):
        config = MultiFactorSelectionConfig(
            weights={"quality": 0.4, "latency": 0.2, "cost": 0.2, "load": 0.2},
            slo={"max_tpot_ms": 200, "max_ttft_ms": 800, "max_cost_per_1m": 5.0},
            latency_percentile=95,
            on_no_candidates="cheapest",
        )
        assert config.weights.quality == 0.4
        assert config.slo.max_ttft_ms == 800
        assert config.latency_percentile == 95


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
        assert config.on_error == "skip"

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
            round_timeout_seconds=120,
            min_successful_responses=2,
        )
        assert config.synthesis_model == "model-b"
        assert config.round_timeout_seconds == 120
        assert config.min_successful_responses == 2

        with pytest.raises(PydanticValidationError):
            ReMoMAlgorithmConfig(
                breadth_schedule=[3],
                round_timeout_seconds=0,
            )


class TestFusionAlgorithmConfig:
    """Test Fusion multi-model deliberation configuration."""

    def test_default_values(self):
        config = FusionAlgorithmConfig()
        assert config.include_analysis is True
        assert config.include_intermediate_responses is True
        assert config.on_error == "skip"
        assert config.judge_prompt_version == "fusion-v1"

    def test_custom_panel_and_judge(self):
        config = FusionAlgorithmConfig(
            model="judge-model",
            analysis_models=["panel-a", "panel-b"],
            max_concurrent=2,
            max_completion_tokens=512,
            temperature=0.2,
        )
        assert config.model == "judge-model"
        assert config.analysis_models == ["panel-a", "panel-b"]
        assert config.max_concurrent == 2
        assert config.max_completion_tokens == 512
        assert config.temperature == 0.2

    def test_positive_limits(self):
        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(max_concurrent=0)

        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(max_completion_tokens=0)

        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(round_timeout_seconds=0)

        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(min_successful_responses=0)

    def test_quorum_and_timeout_controls(self):
        config = FusionAlgorithmConfig(
            round_timeout_seconds=90,
            min_successful_responses=2,
        )
        assert config.round_timeout_seconds == 90
        assert config.min_successful_responses == 2

    def test_on_error_validation(self):
        assert FusionAlgorithmConfig(on_error="fail").on_error == "fail"

        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(on_error="ignore")

    def test_grounding_defaults(self):
        config = FusionAlgorithmConfig(grounding={"enabled": True})
        assert config.grounding.enabled is True
        assert config.grounding.reference == "hybrid"
        assert config.grounding.policy == "weight"
        assert config.grounding.min_score == 0.0
        assert config.grounding.min_keep == 1
        assert config.grounding.nli_contradiction_penalty == 1.0
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
            FusionAlgorithmConfig(grounding={"nli_contradiction_penalty": -1})
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
            ratings=RatingsAlgorithmConfig(max_concurrent=3, on_error="skip"),
        )
        assert config.type == "ratings"
        assert config.ratings.max_concurrent == 3

    def test_multi_factor_algorithm_config(self):
        """Test AlgorithmConfig with multi_factor selection config."""
        config = AlgorithmConfig(
            type="multi_factor",
            multi_factor=MultiFactorSelectionConfig(
                weights={"quality": 0.4, "latency": 0.2, "cost": 0.2, "load": 0.2}
            ),
        )
        assert config.type == "multi_factor"
        assert config.multi_factor.weights.quality == 0.4

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

    def test_hybrid_algorithm_config(self):
        """Test AlgorithmConfig with Hybrid selection and all weights."""
        config = AlgorithmConfig(
            type="hybrid",
            hybrid=HybridSelectionConfig(
                experience_weight=0.4,
                router_dc_weight=0.3,
                automix_weight=0.2,
                cost_weight=0.1,
            ),
        )
        assert config.type == "hybrid"
        total = (
            config.hybrid.experience_weight
            + config.hybrid.router_dc_weight
            + config.hybrid.automix_weight
            + config.hybrid.cost_weight
        )
        assert abs(total - 1.0) < 0.01
