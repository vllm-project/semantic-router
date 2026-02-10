"""Tests for model selection algorithm configuration.

Tests for CLI algorithm support as required by PR #1196 review.
Addresses @Xunzhuo's comment: "we should have a detailed test in the CLI PR"
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from cli.models import (
    AlgorithmConfig,
    EloSelectionConfig,
    RouterDCSelectionConfig,
    AutoMixSelectionConfig,
    HybridSelectionConfig,
    ThompsonSamplingConfig,
    GMTRouterConfig,
    RouterR1Config,
    ReMoMAlgorithmConfig,
)


class TestAlgorithmConfigTypes:
    """Test algorithm type validation."""

    def test_valid_looper_types(self):
        """Test that looper algorithm types are accepted."""
        looper_types = ["confidence", "concurrent", "remom"]

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
            "thompson",
            "gmtrouter",
            "router_r1",
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


class TestThompsonSamplingConfig:
    """Test Thompson Sampling (RL-driven) selection configuration."""

    def test_default_values(self):
        """Test Thompson Sampling config default values."""
        config = ThompsonSamplingConfig()
        assert config.prior_alpha == 1.0
        assert config.prior_beta == 1.0
        assert config.per_user is False
        assert config.decay_factor == 0.0
        assert config.min_samples == 10

    def test_prior_validation(self):
        """Test that priors must be positive."""
        with pytest.raises(PydanticValidationError):
            ThompsonSamplingConfig(prior_alpha=0.0)  # Must be > 0

    def test_per_user_personalization(self):
        """Test per-user personalization flag."""
        config = ThompsonSamplingConfig(per_user=True)
        assert config.per_user is True


class TestGMTRouterConfig:
    """Test GMTRouter (GNN-based) selection configuration."""

    def test_default_values(self):
        """Test GMTRouter config default values."""
        config = GMTRouterConfig()
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
        config = GMTRouterConfig(model_path="/models/gmtrouter.pt")
        assert config.model_path == "/models/gmtrouter.pt"


class TestRouterR1Config:
    """Test Router-R1 (LLM-as-router) selection configuration."""

    def test_default_values(self):
        """Test Router-R1 config default values."""
        config = RouterR1Config()
        assert config.router_endpoint is None
        assert config.max_iterations == 3
        assert config.temperature == 0.7
        assert config.use_cot is True
        assert config.fallback_to_static is True

    def test_max_iterations_validation(self):
        """Test that max_iterations is within valid range."""
        with pytest.raises(PydanticValidationError):
            RouterR1Config(max_iterations=0)  # Must be >= 1

        with pytest.raises(PydanticValidationError):
            RouterR1Config(max_iterations=20)  # Must be <= 10

    def test_router_endpoint(self):
        """Test Router-R1 config with endpoint."""
        config = RouterR1Config(router_endpoint="http://localhost:8080")
        assert config.router_endpoint == "http://localhost:8080"


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
        """Test AlgorithmConfig with Thompson Sampling (RL-driven)."""
        config = AlgorithmConfig(
            type="thompson",
            thompson=ThompsonSamplingConfig(per_user=True, min_samples=20),
        )
        assert config.type == "thompson"
        assert config.thompson.per_user is True
        assert config.thompson.min_samples == 20

    def test_gmtrouter_algorithm_config(self):
        """Test AlgorithmConfig with GMTRouter."""
        config = AlgorithmConfig(
            type="gmtrouter",
            gmtrouter=GMTRouterConfig(num_layers=3, hidden_dim=128),
        )
        assert config.type == "gmtrouter"
        assert config.gmtrouter.num_layers == 3
        assert config.gmtrouter.hidden_dim == 128

    def test_router_r1_algorithm_config(self):
        """Test AlgorithmConfig with Router-R1."""
        config = AlgorithmConfig(
            type="router_r1",
            router_r1=RouterR1Config(
                router_endpoint="http://localhost:8080",
                max_iterations=5,
            ),
        )
        assert config.type == "router_r1"
        assert config.router_r1.router_endpoint == "http://localhost:8080"
        assert config.router_r1.max_iterations == 5

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
