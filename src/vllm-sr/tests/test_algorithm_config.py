"""Tests for model selection algorithm configuration.

Tests for CLI algorithm support as required by PR #1196 review.
Addresses @Xunzhuo's comment: "we should have a detailed test in the CLI PR"
"""

import pytest
from cli.algorithms import (
    AlgorithmConfig,
    AutoMixSelectionConfig,
    EloSelectionConfig,
    FusionAlgorithmConfig,
    GMTRouterConfig,
    HybridSelectionConfig,
    MultiFactorSelectionConfig,
    RatingsAlgorithmConfig,
    ReMoMAlgorithmConfig,
    RLDrivenSelectionConfig,
    RouterDCSelectionConfig,
    SessionAwareSelectionConfig,
)
from pydantic import ValidationError as PydanticValidationError


class TestAlgorithmConfigTypes:
    """Test algorithm type validation."""

    def test_valid_looper_types(self):
        """Test that looper algorithm types are accepted."""
        looper_types = ["confidence", "ratings", "remom", "fusion"]

        for algo_type in looper_types:
            config = AlgorithmConfig(type=algo_type)
            assert config.type == algo_type

    def test_valid_selection_types(self):
        """Test that selection algorithm types are accepted."""
        selection_types = [
            "static",
            "elo",
            "router_dc",
            "automix",
            "hybrid",
            "knn",
            "kmeans",
            "svm",
            "mlp",
            "multi_factor",
            "latency_aware",
            "session_aware",
            "rl_driven",
            "gmtrouter",
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


class TestEloSelectionConfig:
    """Test Elo rating selection configuration."""

    def test_default_values(self):
        """Test Elo config default values."""
        config = EloSelectionConfig()
        assert config.initial_rating == 1500.0
        assert config.k_factor == 32.0
        assert config.category_weighted is True
        assert config.decay_factor == 0.0
        assert config.min_comparisons == 5

    def test_custom_k_factor(self):
        """Test Elo config with custom K-factor."""
        config = EloSelectionConfig(k_factor=64.0)
        assert config.k_factor == 64.0

    def test_k_factor_validation(self):
        """Test that K-factor must be within valid range."""
        with pytest.raises(PydanticValidationError):
            EloSelectionConfig(k_factor=0.5)  # Below minimum (1)

        with pytest.raises(PydanticValidationError):
            EloSelectionConfig(k_factor=150)  # Above maximum (100)

    def test_storage_path(self):
        """Test Elo config with storage path."""
        config = EloSelectionConfig(storage_path="/tmp/elo_ratings.json")
        assert config.storage_path == "/tmp/elo_ratings.json"


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
        assert config.elo_weight == 0.3
        assert config.router_dc_weight == 0.3
        assert config.automix_weight == 0.2
        assert config.cost_weight == 0.2

    def test_custom_weights(self):
        """Test Hybrid config with custom weights."""
        config = HybridSelectionConfig(
            elo_weight=0.5,
            router_dc_weight=0.3,
            automix_weight=0.1,
            cost_weight=0.1,
        )
        # Weights should sum to 1.0
        total = (
            config.elo_weight
            + config.router_dc_weight
            + config.automix_weight
            + config.cost_weight
        )
        assert abs(total - 1.0) < 0.01


class TestGMTRouterConfig:
    """Test GMTRouter (GNN-based) selection configuration."""

    def test_default_values(self):
        """Test GMTRouter config default values."""
        config = GMTRouterConfig()
        assert config.enable_personalization is True
        assert config.history_sample_size == 5
        assert config.embedding_dimension == 768
        assert config.num_gnn_layers == 2
        assert config.attention_heads == 8
        assert config.min_interactions_for_personalization == 3
        assert config.max_interactions_per_user == 100
        assert config.feedback_types == ["rating", "ranking"]
        assert config.num_layers == 2
        assert config.hidden_dim == 64
        assert config.num_heads == 4
        assert config.learn_preferences is True

    def test_num_layers_validation(self):
        """Test that num_layers is within valid range."""
        with pytest.raises(PydanticValidationError):
            GMTRouterConfig(num_layers=0)  # Must be >= 1

        with pytest.raises(PydanticValidationError):
            GMTRouterConfig(num_layers=10)  # Must be <= 5

    def test_model_path(self):
        """Test GMTRouter config with model path."""
        config = GMTRouterConfig(
            model_path="/models/gmtrouter.pt",
            storage_path="/var/lib/vsr/gmt_graph.json",
            feedback_types=["rating", "ranking", "response"],
        )
        assert config.model_path == "/models/gmtrouter.pt"
        assert config.storage_path == "/var/lib/vsr/gmt_graph.json"
        assert config.feedback_types == ["rating", "ranking", "response"]


class TestRLDrivenSelectionConfig:
    """Test canonical rl_driven selection configuration."""

    def test_default_values(self):
        """Test RL-driven config default values."""
        config = RLDrivenSelectionConfig()
        assert config.exploration_rate == 0.3
        assert config.exploration_decay == 0.99
        assert config.min_exploration == 0.05
        assert config.use_thompson_sampling is True
        assert config.enable_personalization is True
        assert config.personalization_blend == 0.7
        assert config.session_context_weight == 0.3
        assert config.implicit_feedback_weight == 0.5
        assert config.cost_awareness is True
        assert config.cost_weight == 0.2
        assert config.auto_save_interval == "30s"
        assert config.use_router_r1_rewards is True
        assert config.cost_reward_alpha == 0.3
        assert config.format_reward_penalty == -1.0

    def test_storage_and_router_r1_fields(self):
        """Test persistence and Router-R1 fields."""
        config = RLDrivenSelectionConfig(
            storage_path="/var/lib/vsr/rl_state.json",
            auto_save_interval="45s",
            enable_llm_routing=True,
            router_r1_server_url="http://router-r1:8080",
            llm_routing_fallback="error",
            enable_multi_round_aggregation=True,
            max_aggregation_rounds=4,
        )
        assert config.storage_path == "/var/lib/vsr/rl_state.json"
        assert config.auto_save_interval == "45s"
        assert config.enable_llm_routing is True
        assert config.router_r1_server_url == "http://router-r1:8080"
        assert config.llm_routing_fallback == "error"
        assert config.enable_multi_round_aggregation is True
        assert config.max_aggregation_rounds == 4


class TestSessionAwareSelectionConfig:
    """Test session-aware selection configuration."""

    def test_base_method_field(self):
        """Test that session-aware uses base_method rather than fallback_method."""
        config = SessionAwareSelectionConfig(base_method="static")
        assert config.base_method == "static"

        with pytest.raises(PydanticValidationError):
            SessionAwareSelectionConfig(fallback_method="static")

    def test_cache_cost_multiplier_is_not_inverted(self):
        """Expensive-model cache pressure must not become weaker than neutral."""
        config = SessionAwareSelectionConfig(max_cache_cost_multiplier=1.0)
        assert config.max_cache_cost_multiplier == 1.0

        with pytest.raises(PydanticValidationError):
            SessionAwareSelectionConfig(max_cache_cost_multiplier=0.5)

    def test_remaining_turn_prior_horizon_is_positive(self):
        """The remaining-turn prior horizon must be explicit positive depth."""
        config = SessionAwareSelectionConfig(remaining_turn_prior_horizon=1)
        assert config.remaining_turn_prior_horizon == 1

        with pytest.raises(PydanticValidationError):
            SessionAwareSelectionConfig(remaining_turn_prior_horizon=0)


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

    def test_on_error_validation(self):
        assert FusionAlgorithmConfig(on_error="fail").on_error == "fail"

        with pytest.raises(PydanticValidationError):
            FusionAlgorithmConfig(on_error="ignore")

    def test_grounding_defaults(self):
        config = FusionAlgorithmConfig(grounding={"enabled": True})
        assert config.grounding.enabled is True
        assert config.grounding.reference == "hybrid"
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


class TestAlgorithmConfigIntegration:
    """Integration tests for AlgorithmConfig with nested configs."""

    def test_elo_algorithm_config(self):
        """Test AlgorithmConfig with Elo selection."""
        config = AlgorithmConfig(
            type="elo",
            elo=EloSelectionConfig(k_factor=48.0, min_comparisons=10),
        )
        assert config.type == "elo"
        assert config.elo.k_factor == 48.0
        assert config.elo.min_comparisons == 10

    def test_rl_driven_algorithm_config(self):
        """Test AlgorithmConfig with canonical RL-driven config."""
        config = AlgorithmConfig(
            type="rl_driven",
            rl_driven=RLDrivenSelectionConfig(storage_path="/tmp/rl.json"),
        )
        assert config.type == "rl_driven"
        assert config.rl_driven.storage_path == "/tmp/rl.json"

    def test_ratings_algorithm_config(self):
        """Test AlgorithmConfig with ratings looper config."""
        config = AlgorithmConfig(
            type="ratings",
            ratings=RatingsAlgorithmConfig(max_concurrent=3, on_error="skip"),
        )
        assert config.type == "ratings"
        assert config.ratings.max_concurrent == 3

    def test_gmtrouter_algorithm_config(self):
        """Test AlgorithmConfig with GMTRouter."""
        config = AlgorithmConfig(
            type="gmtrouter",
            gmtrouter=GMTRouterConfig(num_layers=3, hidden_dim=128),
        )
        assert config.type == "gmtrouter"
        assert config.gmtrouter.num_layers == 3
        assert config.gmtrouter.hidden_dim == 128

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

    def test_removed_algorithm_type_specific_blocks_are_rejected(self):
        """Thompson and Router-R1 are rl_driven modes, not top-level algorithms."""
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
                elo_weight=0.4,
                router_dc_weight=0.3,
                automix_weight=0.2,
                cost_weight=0.1,
            ),
        )
        assert config.type == "hybrid"
        total = (
            config.hybrid.elo_weight
            + config.hybrid.router_dc_weight
            + config.hybrid.automix_weight
            + config.hybrid.cost_weight
        )
        assert abs(total - 1.0) < 0.01
