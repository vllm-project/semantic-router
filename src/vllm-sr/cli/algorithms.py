"""Algorithm configuration models for multi-model orchestration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelRef(BaseModel):
    """Model reference in decision."""

    model: str
    use_reasoning: bool | None = False
    reasoning_description: str | None = None
    reasoning_effort: str | None = (
        None  # Model-specific reasoning effort level (low, medium, high)
    )
    lora_name: str | None = None  # LoRA adapter name (if using LoRA)
    weight: float | None = None


class HybridWeightsConfig(BaseModel):
    """Weights configuration for hybrid confidence method."""

    logprob_weight: float | None = 0.5  # Weight for avg_logprob (default: 0.5)
    margin_weight: float | None = 0.5  # Weight for margin (default: 0.5)


class ConfidenceAlgorithmConfig(BaseModel):
    """Configuration for confidence algorithm.

    This algorithm tries smaller models first and escalates to larger models if confidence is low.
    """

    # Confidence evaluation method
    # - "avg_logprob": Use average logprob across all tokens (default)
    # - "margin": Use average margin between top-1 and top-2 logprobs (more accurate)
    # - "hybrid": Use weighted combination of both methods
    confidence_method: str | None = "avg_logprob"

    # Threshold for escalation (meaning depends on confidence_method)
    # For avg_logprob: negative, closer to 0 = more confident (default: -1.0)
    # For margin: positive, higher = more confident (default: 0.5)
    # For hybrid: 0-1 normalized score (default: 0.5)
    threshold: float | None = None

    # Hybrid weights (only used when confidence_method="hybrid")
    hybrid_weights: HybridWeightsConfig | None = None

    # Behavior on model call failure: "skip" or "fail"
    on_error: str | None = "skip"


class RatingsAlgorithmConfig(BaseModel):
    """Configuration for the ratings looper algorithm."""

    model_config = ConfigDict(extra="forbid")

    max_concurrent: int | None = Field(default=None, ge=1)
    on_error: str | None = "skip"


class ReMoMAlgorithmConfig(BaseModel):
    """Configuration for ReMoM (Reasoning for Mixture of Models) algorithm.

    This algorithm performs multi-round parallel reasoning with intelligent synthesis.
    Inspired by PaCoRe (arXiv:2601.05593) but extended to support mixture of models.
    """

    # Breadth schedule: array of parallel calls per round
    # e.g., [32, 4] means 32 calls in round 1, 4 in round 2, then 1 final
    breadth_schedule: list[int]

    # Model distribution strategy: "weighted", "equal", "round_robin", or "first_only"
    model_distribution: (
        Literal["weighted", "equal", "round_robin", "first_only"] | None
    ) = "weighted"

    # Temperature for model calls (default: 1.0 for diverse exploration)
    temperature: float | None = 1.0

    # Whether to include reasoning content in synthesis prompts
    include_reasoning: bool | None = False

    # Compaction strategy: "full" or "last_n_tokens"
    compaction_strategy: str | None = "full"

    # Number of tokens to keep when using last_n_tokens compaction
    compaction_tokens: int | None = 1000

    # Custom synthesis template (uses default if not provided)
    synthesis_template: str | None = None

    # Explicit final synthesis model. Must be one of the decision modelRefs.
    synthesis_model: str | None = None

    # Maximum concurrent model calls per round
    max_concurrent: int | None = None

    # Maximum completion tokens applied to every ReMoM model call
    max_completion_tokens: int | None = Field(default=None, ge=1)

    # Maximum wall-clock time to wait for a ReMoM round before using partial
    # responses when on_error="skip".
    round_timeout_seconds: int | None = Field(default=None, ge=1)

    # Return from a parallel round as soon as this many responses succeed.
    min_successful_responses: int | None = Field(default=None, ge=1)

    # Behavior on model call failure: "skip" or "fail"
    on_error: str | None = "skip"

    # Random seed for shuffling responses (for reproducibility)
    shuffle_seed: int | None = 42

    # Whether to include intermediate responses in the response body
    include_intermediate_responses: bool | None = True

    # Maximum number of responses to keep per round (for memory efficiency)
    max_responses_per_round: int | None = None


class FusionGroundingConfig(BaseModel):
    """Configuration for grounding-aware fusion.

    Scores each panel response for faithfulness before the judge synthesizes,
    then ranks/filters the panel. Uses local encoder models (hallucination
    detector + NLI) and makes no extra LLM calls. Bounds here MUST match the Go
    validator in pkg/config/fusion_config.go (ValidateFusionGroundingConfig).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = False
    reference: Literal["hybrid", "context", "panel"] | None = "hybrid"
    # How the groundedness scores are used. weight/annotate keep every response and
    # let the judge soft-weight; filter hard-drops below min_score. Default weight:
    # hard-dropping the least consistent response regresses quality on contested
    # factual items (see bench/grounded_fusion/FINDINGS.md).
    policy: Literal["weight", "annotate", "filter"] | None = "weight"
    min_score: float | None = Field(default=0.0, ge=0, le=1)
    min_keep: int | None = Field(default=1, ge=0)
    nli_contradiction_penalty: float | None = Field(default=1.0, ge=0)
    on_error: Literal["skip", "fail"] | None = "skip"


class FusionModelOverrideConfig(BaseModel):
    """Per-analysis-model sampling controls for Fusion."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0)
    max_completion_tokens: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_model_name(self):
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        return self


class FusionAlgorithmConfig(BaseModel):
    """Configuration for Fusion multi-model deliberation.

    The ``model`` field is the judge/calling model. ``analysis_models`` can
    override decision.modelRefs when a route wants a dedicated panel list.
    """

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    analysis_models: list[str] | None = None
    analysis_overrides: list[FusionModelOverrideConfig] | None = None
    max_concurrent: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    round_timeout_seconds: int | None = Field(default=None, ge=1)
    min_successful_responses: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    include_analysis: bool | None = True
    include_intermediate_responses: bool | None = True
    on_error: Literal["skip", "fail"] | None = "skip"
    analysis_template: str | None = None
    synthesis_template: str | None = None
    judge_prompt_version: str | None = "fusion-v1"
    grounding: FusionGroundingConfig | None = None

    @model_validator(mode="after")
    def validate_analysis_override_models(self):
        seen: set[str] = set()
        for override in self.analysis_overrides or []:
            model = override.model.strip()
            if model in seen:
                raise ValueError(f"analysis override model {model!r} is duplicated")
            seen.add(model)
        return self


class WorkflowPlannerConfig(BaseModel):
    """Control-plane model used by dynamic Router Flow planning."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    max_completion_tokens: int | None = Field(default=None, ge=1)


class WorkflowRoleConfig(BaseModel):
    """Static Router Flow role mapped to decision modelRefs."""

    model_config = ConfigDict(extra="forbid")

    name: str
    # Physical Models are supplied by Entrypoint assignments. Only the internal
    # compiled decision form may materialize this field.
    models: list[str] = Field(default_factory=list)
    prompt: str | None = None
    access_list: list[str] | None = None


class WorkflowFinalConfig(BaseModel):
    """Static Router Flow final synthesis override."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    prompt: str | None = None


class WorkflowsAlgorithmConfig(BaseModel):
    """Configuration for Router Flow workflow orchestration.

    decision.modelRefs is the worker boundary. ``planner.model`` is the
    control-plane model that generates dynamic workflow plans.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["static", "dynamic"] | None = "static"
    template: str | None = "micro_agent"
    roles: list[WorkflowRoleConfig] | None = None
    final: WorkflowFinalConfig | None = None
    planner: WorkflowPlannerConfig | None = None
    max_steps: int | None = Field(default=3, ge=1)
    max_parallel: int | None = Field(default=2, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    round_timeout_seconds: int | None = Field(default=None, ge=1)
    min_successful_responses: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    include_intermediate_responses: bool | None = True
    on_error: Literal["skip", "fail"] | None = "fail"


class LatencyAwareAlgorithmConfig(BaseModel):
    """Configuration for latency_aware algorithm."""

    tpot_percentile: int | None = None
    ttft_percentile: int | None = None
    description: str | None = None


# =============================================================================
# Model Selection Algorithm Configs
# Reference papers:
#   - RouterDC: Query-Based Router by Dual Contrastive Learning (arXiv:2409.19886)
#   - AutoMix: Automatically Mixing Language Models (arXiv:2310.12963)
#   - Hybrid: Cost-Efficient Quality-Aware Query Routing (arXiv:2404.14618)
# =============================================================================


class RouterDCSelectionConfig(BaseModel):
    """Configuration for RouterDC (Dual-Contrastive) model selection.

    Matches queries to models using embedding similarity based on model descriptions.
    """

    # Temperature for softmax scaling (default: 0.07)
    temperature: float | None = Field(default=0.07, gt=0)

    # Embedding dimension size (default: 768)
    dimension_size: int | None = Field(default=768, gt=0)

    # Minimum similarity threshold for valid matches
    min_similarity: float | None = Field(default=0.3, ge=0, le=1)

    # Enable query-side contrastive learning
    use_query_contrastive: bool | None = True

    # Enable model-side contrastive learning
    use_model_contrastive: bool | None = True

    # Require all models to have descriptions
    require_descriptions: bool | None = False

    # Include capability tags in embeddings
    use_capabilities: bool | None = True


class AutoMixSelectionConfig(BaseModel):
    """Configuration for AutoMix (POMDP-based) model selection.

    Optimizes cost-quality tradeoff using Partially Observable MDP.
    """

    # Self-verification confidence threshold (default: 0.7)
    verification_threshold: float | None = Field(default=0.7, ge=0, le=1)

    # Maximum escalation attempts (default: 2)
    max_escalations: int | None = Field(default=2, ge=0)

    # Enable cost-quality tradeoff optimization
    cost_aware_routing: bool | None = True

    # Balance between cost (1.0) and quality (0.0) (default: 0.3)
    cost_quality_tradeoff: float | None = Field(default=0.3, ge=0, le=1)

    # POMDP discount factor (default: 0.95)
    discount_factor: float | None = Field(default=0.95, ge=0, le=1)

    # Use logprobs for confidence verification
    use_logprob_verification: bool | None = True


class HybridSelectionConfig(BaseModel):
    """Configuration for Hybrid model selection.

    Combines multiple selection methods with configurable weights.
    """

    # Weight for feedback-derived experience evidence (0-1).
    experience_weight: float | None = Field(default=0.3, ge=0, le=1)

    # Weight for RouterDC embedding similarity (0-1)
    router_dc_weight: float | None = Field(default=0.3, ge=0, le=1)

    # Weight for AutoMix POMDP value (0-1)
    automix_weight: float | None = Field(default=0.2, ge=0, le=1)

    # Weight for cost consideration (0-1)
    cost_weight: float | None = Field(default=0.2, ge=0, le=1)

    # Quality gap threshold for escalation
    quality_gap_threshold: float | None = Field(default=0.1, ge=0, le=1)

    # Normalize scores before combination
    normalize_scores: bool | None = True


class MultiFactorWeightsConfig(BaseModel):
    """Weights for the multi_factor selector."""

    model_config = ConfigDict(extra="forbid")

    quality: float | None = Field(default=None, ge=0)
    latency: float | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0)
    load: float | None = Field(default=None, ge=0)


class MultiFactorSLOConfig(BaseModel):
    """SLO ceilings for the multi_factor selector."""

    model_config = ConfigDict(extra="forbid")

    max_tpot_ms: float | None = Field(default=None, ge=0)
    max_ttft_ms: float | None = Field(default=None, ge=0)
    max_cost_per_1m: float | None = Field(default=None, ge=0)
    max_inflight: int | None = Field(default=None, ge=0)


class MultiFactorSelectionConfig(BaseModel):
    """Configuration for the canonical multi_factor selector."""

    model_config = ConfigDict(extra="forbid")

    weights: MultiFactorWeightsConfig | None = None
    slo: MultiFactorSLOConfig | None = None
    latency_percentile: int | None = Field(default=95, ge=1, le=100)
    on_no_candidates: str | None = "cheapest"


class PromptSelectionConfig(BaseModel):
    """Configuration for prompt-driven candidate selection."""

    model_config = ConfigDict(extra="forbid")

    # The selector uses the Entrypoint-assigned candidate set in v0.4.
    model: str | None = None
    instructions: str
    timeout_seconds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_required_text(self):
        if self.model is not None and not self.model.strip():
            raise ValueError("model cannot be blank")
        if not self.instructions.strip():
            raise ValueError("instructions are required")
        return self


class MLKNNConfig(BaseModel):
    """Decision-local KNN selector settings."""

    model_config = ConfigDict(extra="forbid")

    k: int | None = Field(default=None, gt=0)
    pretrained_path: str | None = None


class MLKMeansConfig(BaseModel):
    """Decision-local K-Means selector settings."""

    model_config = ConfigDict(extra="forbid")

    num_clusters: int | None = Field(default=None, gt=0)
    efficiency_weight: float | None = Field(default=None, ge=0, le=1)
    pretrained_path: str | None = None


class MLSVMConfig(BaseModel):
    """Decision-local SVM selector settings."""

    model_config = ConfigDict(extra="forbid")

    kernel: Literal["linear", "rbf", "gaussian"] | None = None
    gamma: float | None = Field(default=None, ge=0)
    pretrained_path: str | None = None


class MLMLPConfig(BaseModel):
    """Decision-local MLP selector settings."""

    model_config = ConfigDict(extra="forbid")

    device: Literal["cpu", "cuda", "metal"] | None = None
    pretrained_path: str | None = None


class MLSelectionConfig(BaseModel):
    """Artifact and family settings owned by one Recipe Decision."""

    model_config = ConfigDict(extra="forbid")

    models_path: str | None = None
    embedding_dim: int | None = Field(default=None, gt=0)
    knn: MLKNNConfig | None = None
    kmeans: MLKMeansConfig | None = None
    svm: MLSVMConfig | None = None
    mlp: MLMLPConfig | None = None


class AlgorithmConfig(BaseModel):
    """Algorithm configuration for multi-model decisions.

    Specifies how multiple models in a decision should be orchestrated.

    Supports three categories of algorithms:

    1. Looper algorithms (multi-model execution):
       - "confidence": Try smaller models first, escalate if confidence is low
       - "ratings": Coordinate bounded candidate execution
       - "remom": Multi-round parallel reasoning with intelligent synthesis
       - "fusion": Parallel panel deliberation with judge analysis and final synthesis
       - "workflows": Router Flow dynamic/static micro-agent workflows

    2. Selection algorithms (single model selection from candidates):
       - "static": Use first model (default)
       - "router_dc": Use embedding similarity for query-model matching
       - "automix": Use POMDP-based cost-quality optimization
       - "hybrid": Combine multiple selection methods
       - "knn", "kmeans", "svm", "mlp": Shared ML model-selection selectors
       - "multi_factor": Combine quality, latency, cost, and load

    Cross-request learning systems live under global.router.learning.adaptation
    and global.router.learning.protection.
    """

    model_config = ConfigDict(extra="forbid")

    # Algorithm type: looper ("confidence", "ratings", "remom", "fusion",
    # "workflows") or
    # selection ("static", "router_dc", "automix", "hybrid", "knn",
    #            "kmeans", "svm", "mlp", "multi_factor", "latency_aware")
    type: Literal[
        "confidence",
        "ratings",
        "remom",
        "fusion",
        "workflows",
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
        "prompt",
    ]

    # Looper algorithm configurations
    confidence: ConfidenceAlgorithmConfig | None = None
    ratings: RatingsAlgorithmConfig | None = None
    remom: ReMoMAlgorithmConfig | None = None
    fusion: FusionAlgorithmConfig | None = None
    workflows: WorkflowsAlgorithmConfig | None = None
    latency_aware: LatencyAwareAlgorithmConfig | None = None

    # Selection algorithm configurations
    router_dc: RouterDCSelectionConfig | None = None
    automix: AutoMixSelectionConfig | None = None
    hybrid: HybridSelectionConfig | None = None
    multi_factor: MultiFactorSelectionConfig | None = None
    prompt: PromptSelectionConfig | None = None
    ml: MLSelectionConfig | None = None
    # Behavior on algorithm failure: "skip" or "fail"
    on_error: str | None = "skip"

    @model_validator(mode="after")
    def validate_scoped_algorithm_config(self):
        if self.type == "prompt":
            if self.prompt is None:
                raise ValueError("algorithm.type=prompt requires prompt configuration")
            if self.on_error == "skip":
                self.on_error = "fallback"
            if self.on_error not in (None, "", "fallback"):
                raise ValueError("prompt on_error must be fallback")
        elif self.prompt is not None:
            raise ValueError("prompt configuration requires algorithm.type=prompt")

        ml_types = {"knn", "kmeans", "svm", "mlp"}
        if self.type in ml_types:
            if self.ml is None:
                raise ValueError(
                    f"algorithm.type={self.type} requires algorithm.ml configuration"
                )
            configured_families = [
                family
                for family in ("knn", "kmeans", "svm", "mlp")
                if getattr(self.ml, family) is not None
            ]
            if configured_families != [self.type]:
                if not configured_families:
                    raise ValueError(
                        f"algorithm.type={self.type} requires algorithm.ml.{self.type} configuration"
                    )
                raise ValueError(
                    f"algorithm.type={self.type} requires only algorithm.ml.{self.type} "
                    f"configuration; found {', '.join(configured_families)}"
                )
        elif self.ml is not None:
            raise ValueError("algorithm.ml requires an ML algorithm.type")
        return self
