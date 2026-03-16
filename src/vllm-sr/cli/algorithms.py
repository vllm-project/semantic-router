"""Algorithm configuration models for multi-model orchestration."""

from typing import Optional

from pydantic import BaseModel, Field


class ModelRef(BaseModel):
    """Model reference in decision."""

    model: str
    use_reasoning: Optional[bool] = False
    reasoning_description: Optional[str] = None
    reasoning_effort: Optional[str] = (
        None  # Model-specific reasoning effort level (low, medium, high)
    )
    lora_name: Optional[str] = None  # LoRA adapter name (if using LoRA)
    weight: Optional[float] = None


class HybridWeightsConfig(BaseModel):
    """Weights configuration for hybrid confidence method."""

    logprob_weight: Optional[float] = 0.5  # Weight for avg_logprob (default: 0.5)
    margin_weight: Optional[float] = 0.5  # Weight for margin (default: 0.5)


class ConfidenceAlgorithmConfig(BaseModel):
    """Configuration for confidence algorithm.

    This algorithm tries smaller models first and escalates to larger models if confidence is low.
    """

    # Confidence evaluation method
    # - "avg_logprob": Use average logprob across all tokens (default)
    # - "margin": Use average margin between top-1 and top-2 logprobs (more accurate)
    # - "hybrid": Use weighted combination of both methods
    confidence_method: Optional[str] = "avg_logprob"

    # Threshold for escalation (meaning depends on confidence_method)
    # For avg_logprob: negative, closer to 0 = more confident (default: -1.0)
    # For margin: positive, higher = more confident (default: 0.5)
    # For hybrid: 0-1 normalized score (default: 0.5)
    threshold: Optional[float] = None

    # Hybrid weights (only used when confidence_method="hybrid")
    hybrid_weights: Optional[HybridWeightsConfig] = None

    # Behavior on model call failure: "skip" or "fail"
    on_error: Optional[str] = "skip"


class ConcurrentAlgorithmConfig(BaseModel):
    """Configuration for concurrent algorithm.

    This algorithm executes all models concurrently and aggregates results (arena mode).
    """

    # Maximum number of concurrent model calls (default: no limit)
    max_concurrent: Optional[int] = None

    # Behavior on model call failure: "skip" or "fail"
    on_error: Optional[str] = "skip"


class ReMoMAlgorithmConfig(BaseModel):
    """Configuration for ReMoM (Reasoning for Mixture of Models) algorithm.

    This algorithm performs multi-round parallel reasoning with intelligent synthesis.
    Inspired by PaCoRe (arXiv:2601.05593) but extended to support mixture of models.
    """

    # Breadth schedule: array of parallel calls per round
    # e.g., [32, 4] means 32 calls in round 1, 4 in round 2, then 1 final
    breadth_schedule: list[int]

    # Model distribution strategy: "weighted", "equal", or "first_only"
    model_distribution: Optional[str] = "weighted"

    # Temperature for model calls (default: 1.0 for diverse exploration)
    temperature: Optional[float] = 1.0

    # Whether to include reasoning content in synthesis prompts
    include_reasoning: Optional[bool] = False

    # Compaction strategy: "full" or "last_n_tokens"
    compaction_strategy: Optional[str] = "full"

    # Number of tokens to keep when using last_n_tokens compaction
    compaction_tokens: Optional[int] = 1000

    # Custom synthesis template (uses default if not provided)
    synthesis_template: Optional[str] = None

    # Maximum concurrent model calls per round
    max_concurrent: Optional[int] = None

    # Behavior on model call failure: "skip" or "fail"
    on_error: Optional[str] = "skip"

    # Random seed for shuffling responses (for reproducibility)
    shuffle_seed: Optional[int] = 42

    # Whether to include intermediate responses in the response body
    include_intermediate_responses: Optional[bool] = True

    # Maximum number of responses to keep per round (for memory efficiency)
    max_responses_per_round: Optional[int] = None


class LatencyAwareAlgorithmConfig(BaseModel):
    """Configuration for latency_aware algorithm."""

    tpot_percentile: Optional[int] = None
    ttft_percentile: Optional[int] = None
    description: Optional[str] = None


# =============================================================================
# Model Selection Algorithm Configs
# Reference papers:
#   - Elo: RouteLLM (arXiv:2406.18665) - Bradley-Terry model
#   - RouterDC: Query-Based Router by Dual Contrastive Learning (arXiv:2409.19886)
#   - AutoMix: Automatically Mixing Language Models (arXiv:2310.12963)
#   - Hybrid: Cost-Efficient Quality-Aware Query Routing (arXiv:2404.14618)
# =============================================================================


class EloSelectionConfig(BaseModel):
    """Configuration for Elo rating-based model selection.

    Uses Bradley-Terry model for pairwise comparisons, learning from user feedback.
    """

    # Starting Elo rating for new models (default: 1500)
    initial_rating: Optional[float] = Field(default=1500.0, ge=0)

    # Controls rating volatility - higher = faster adaptation (default: 32)
    k_factor: Optional[float] = Field(default=32.0, ge=1, le=100)

    # Enable per-category Elo ratings (default: true)
    category_weighted: Optional[bool] = True

    # Time decay for old comparisons (0-1, 0 = no decay)
    decay_factor: Optional[float] = Field(default=0.0, ge=0, le=1)

    # Minimum comparisons before rating is stable
    min_comparisons: Optional[int] = Field(default=5, ge=0)

    # Cost consideration factor (0 = ignore cost)
    cost_scaling_factor: Optional[float] = Field(default=0.0, ge=0)

    # File path for persisting Elo ratings (optional)
    storage_path: Optional[str] = None

    # Auto-save interval (e.g., "5m", "30s")
    auto_save_interval: Optional[str] = "1m"


class RouterDCSelectionConfig(BaseModel):
    """Configuration for RouterDC (Dual-Contrastive) model selection.

    Matches queries to models using embedding similarity based on model descriptions.
    """

    # Temperature for softmax scaling (default: 0.07)
    temperature: Optional[float] = Field(default=0.07, gt=0)

    # Embedding dimension size (default: 768)
    dimension_size: Optional[int] = Field(default=768, gt=0)

    # Minimum similarity threshold for valid matches
    min_similarity: Optional[float] = Field(default=0.3, ge=0, le=1)

    # Enable query-side contrastive learning
    use_query_contrastive: Optional[bool] = False

    # Enable model-side contrastive learning
    use_model_contrastive: Optional[bool] = False

    # Require all models to have descriptions
    require_descriptions: Optional[bool] = False

    # Include capability tags in embeddings
    use_capabilities: Optional[bool] = False


class AutoMixSelectionConfig(BaseModel):
    """Configuration for AutoMix (POMDP-based) model selection.

    Optimizes cost-quality tradeoff using Partially Observable MDP.
    """

    # Self-verification confidence threshold (default: 0.7)
    verification_threshold: Optional[float] = Field(default=0.7, ge=0, le=1)

    # Maximum escalation attempts (default: 2)
    max_escalations: Optional[int] = Field(default=2, ge=0)

    # Enable cost-quality tradeoff optimization
    cost_aware_routing: Optional[bool] = True

    # Balance between cost (1.0) and quality (0.0) (default: 0.3)
    cost_quality_tradeoff: Optional[float] = Field(default=0.3, ge=0, le=1)

    # POMDP discount factor (default: 0.95)
    discount_factor: Optional[float] = Field(default=0.95, ge=0, le=1)

    # Use logprobs for confidence verification
    use_logprob_verification: Optional[bool] = True


class HybridSelectionConfig(BaseModel):
    """Configuration for Hybrid model selection.

    Combines multiple selection methods with configurable weights.
    """

    # Weight for Elo rating contribution (0-1)
    elo_weight: Optional[float] = Field(default=0.3, ge=0, le=1)

    # Weight for RouterDC embedding similarity (0-1)
    router_dc_weight: Optional[float] = Field(default=0.3, ge=0, le=1)

    # Weight for AutoMix POMDP value (0-1)
    automix_weight: Optional[float] = Field(default=0.2, ge=0, le=1)

    # Weight for cost consideration (0-1)
    cost_weight: Optional[float] = Field(default=0.2, ge=0, le=1)

    # Quality gap threshold for escalation
    quality_gap_threshold: Optional[float] = Field(default=0.1, ge=0, le=1)

    # Normalize scores before combination
    normalize_scores: Optional[bool] = True


# =============================================================================
# RL-Driven Model Selection Algorithm Configs (from PR #1196 / Issue #994)
# Reference papers:
#   - Thompson Sampling: Exploration/exploitation via posterior sampling
#   - GMTRouter: Heterogeneous GNN for personalized routing (arXiv:2511.08590)
#   - Router-R1: LLM-as-router with think/route actions (arXiv:2506.09033)
# =============================================================================


class ThompsonSamplingConfig(BaseModel):
    """Configuration for Thompson Sampling model selection.

    Uses Bayesian posterior sampling for exploration/exploitation balance.
    """

    # Prior alpha for Beta distribution (default: 1.0)
    prior_alpha: Optional[float] = Field(default=1.0, gt=0)

    # Prior beta for Beta distribution (default: 1.0)
    prior_beta: Optional[float] = Field(default=1.0, gt=0)

    # Enable per-user personalization
    per_user: Optional[bool] = False

    # Decay factor for old observations (0 = no decay)
    decay_factor: Optional[float] = Field(default=0.0, ge=0, le=1)

    # Minimum samples before exploitation (default: 10)
    min_samples: Optional[int] = Field(default=10, ge=0)


class GMTRouterConfig(BaseModel):
    """Configuration for GMTRouter (Graph-based) model selection.

    Uses heterogeneous GNN for personalized routing decisions.
    """

    # Number of GNN layers (default: 2)
    num_layers: Optional[int] = Field(default=2, ge=1, le=5)

    # Hidden dimension size (default: 64)
    hidden_dim: Optional[int] = Field(default=64, gt=0)

    # Attention heads (default: 4)
    num_heads: Optional[int] = Field(default=4, ge=1)

    # Enable user preference learning
    learn_preferences: Optional[bool] = True

    # Path to pre-trained model weights (optional)
    model_path: Optional[str] = None


class RouterR1Config(BaseModel):
    """Configuration for Router-R1 (LLM-as-router) model selection.

    Uses LLM with think/route actions for routing decisions.
    """

    # Router LLM endpoint (required for full functionality)
    router_endpoint: Optional[str] = None

    # Maximum think iterations (default: 3)
    max_iterations: Optional[int] = Field(default=3, ge=1, le=10)

    # Temperature for router LLM (default: 0.7)
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)

    # Enable chain-of-thought reasoning
    use_cot: Optional[bool] = True

    # Fallback to static if router unavailable
    fallback_to_static: Optional[bool] = True


class AlgorithmConfig(BaseModel):
    """Algorithm configuration for multi-model decisions.

    Specifies how multiple models in a decision should be orchestrated.

    Supports three categories of algorithms:

    1. Looper algorithms (multi-model execution):
       - "confidence": Try smaller models first, escalate if confidence is low
       - "concurrent": Execute all models concurrently (arena mode)
       - "remom": Multi-round parallel reasoning with intelligent synthesis

    2. Selection algorithms (single model selection from candidates):
       - "static": Use first model (default)
       - "elo": Use Elo rating system with Bradley-Terry model
       - "router_dc": Use embedding similarity for query-model matching
       - "automix": Use POMDP-based cost-quality optimization
       - "hybrid": Combine multiple selection methods

    3. RL-driven selection algorithms (from issue #994):
       - "thompson": Thompson Sampling with exploration/exploitation
       - "gmtrouter": Graph neural network for personalized routing
       - "router_r1": LLM-as-router with think/route actions
    """

    # Algorithm type: looper ("confidence", "concurrent", "remom", "latency_aware") or
    # selection ("static", "elo", "router_dc", "automix", "hybrid",
    #            "thompson", "gmtrouter", "router_r1")
    type: str

    # Looper algorithm configurations
    confidence: Optional[ConfidenceAlgorithmConfig] = None
    concurrent: Optional[ConcurrentAlgorithmConfig] = None
    remom: Optional[ReMoMAlgorithmConfig] = None
    latency_aware: Optional[LatencyAwareAlgorithmConfig] = None

    # Selection algorithm configurations (from PR #1089, #1104)
    elo: Optional[EloSelectionConfig] = None
    router_dc: Optional[RouterDCSelectionConfig] = None
    automix: Optional[AutoMixSelectionConfig] = None
    hybrid: Optional[HybridSelectionConfig] = None

    # RL-driven selection algorithms (from PR #1196, issue #994)
    thompson: Optional[ThompsonSamplingConfig] = None
    gmtrouter: Optional[GMTRouterConfig] = None
    router_r1: Optional[RouterR1Config] = None

    # Behavior on algorithm failure: "skip" or "fail"
    on_error: Optional[str] = "skip"
