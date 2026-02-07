"""Pydantic models for vLLM Semantic Router configuration."""

from typing import List, Dict, Any, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field


class Listener(BaseModel):
    """Network listener configuration."""

    name: str
    address: str
    port: int
    timeout: Optional[str] = "300s"


class KeywordSignal(BaseModel):
    """Keyword-based signal configuration."""

    name: str
    operator: str
    keywords: List[str]
    case_sensitive: bool = False


class EmbeddingSignal(BaseModel):
    """Embedding-based signal configuration."""

    name: str
    threshold: float
    candidates: List[str]
    aggregation_method: str = "max"


class Domain(BaseModel):
    """Domain category configuration."""

    name: str
    description: str
    mmlu_categories: Optional[List[str]] = None


class FactCheck(BaseModel):
    """Fact-checking signal configuration."""

    name: str
    description: str


class UserFeedback(BaseModel):
    """User feedback signal configuration."""

    name: str
    description: str


class Preference(BaseModel):
    """Route preference signal configuration."""

    name: str
    description: str


class Language(BaseModel):
    """Language detection signal configuration."""

    name: str
    description: str


class Latency(BaseModel):
    """Latency signal configuration."""

    name: str
    tpot_percentile: Optional[int] = None
    ttft_percentile: Optional[int] = None
    description: str


class ContextRule(BaseModel):
    """Context-based (token count) signal configuration."""

    name: str
    min_tokens: str  # Supports suffixes: "1K", "1.5M", etc.
    max_tokens: str
    description: Optional[str] = None


class ComplexityCandidates(BaseModel):
    """Complexity candidates configuration."""

    candidates: List[str]


class ComplexityRule(BaseModel):
    """Complexity-based signal configuration using embedding similarity.

    The composer field allows filtering based on other signals (e.g., only apply
    code_complexity when domain is "computer_science"). This is evaluated after
    all signals are computed in parallel, enabling signal dependencies.
    """

    name: str
    threshold: float = 0.1
    hard: ComplexityCandidates
    easy: ComplexityCandidates
    description: Optional[str] = None
    composer: Optional["Rules"] = None  # Forward reference, defined below


class Signals(BaseModel):
    """All signal configurations."""

    keywords: Optional[List[KeywordSignal]] = []
    embeddings: Optional[List[EmbeddingSignal]] = []
    domains: Optional[List[Domain]] = []
    fact_check: Optional[List[FactCheck]] = []
    user_feedbacks: Optional[List[UserFeedback]] = []
    preferences: Optional[List[Preference]] = []
    language: Optional[List[Language]] = []
    latency: Optional[List[Latency]] = []
    context: Optional[List[ContextRule]] = []
    complexity: Optional[List[ComplexityRule]] = []


class Condition(BaseModel):
    """Routing condition."""

    type: str
    name: str


class Rules(BaseModel):
    """Routing rules."""

    operator: str
    conditions: List[Condition]


class ModelRef(BaseModel):
    """Model reference in decision."""

    model: str
    use_reasoning: Optional[bool] = False
    reasoning_effort: Optional[str] = (
        None  # Model-specific reasoning effort level (low, medium, high)
    )
    lora_name: Optional[str] = None  # LoRA adapter name (if using LoRA)


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

    # Breadth schedule: array of parallel calls per round (e.g., [32, 4] means 32 calls in round 1, 4 in round 2, then 1 final)
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

    # Whether to include intermediate responses in the response body for visualization
    include_intermediate_responses: Optional[bool] = True

    # Maximum number of responses to keep per round (for memory efficiency)
    max_responses_per_round: Optional[int] = None


class AlgorithmConfig(BaseModel):
    """Algorithm configuration for multi-model decisions.

    Specifies how multiple models in a decision should be orchestrated.
    """

    # Algorithm type: "sequential", "confidence", "concurrent", "remom"
    type: str

    # Algorithm-specific configurations (only one should be set based on type)
    confidence: Optional[ConfidenceAlgorithmConfig] = None
    concurrent: Optional[ConcurrentAlgorithmConfig] = None
    remom: Optional[ReMoMAlgorithmConfig] = None


class PluginType(str, Enum):
    """Supported plugin types."""

    SEMANTIC_CACHE = "semantic-cache"
    JAILBREAK = "jailbreak"
    PII = "pii"
    SYSTEM_PROMPT = "system_prompt"
    HEADER_MUTATION = "header_mutation"
    HALLUCINATION = "hallucination"
    ROUTER_REPLAY = "router_replay"


class SemanticCachePluginConfig(BaseModel):
    """Configuration for semantic-cache plugin."""

    enabled: bool
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Similarity threshold (0.0-1.0, default: None)",
    )
    ttl_seconds: Optional[int] = Field(
        default=None, ge=0, description="TTL in seconds (must be >= 0, default: None)"
    )


class JailbreakPluginConfig(BaseModel):
    """Configuration for jailbreak plugin."""

    enabled: bool
    threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Threshold (0.0-1.0, default: None)"
    )


class PIIPluginConfig(BaseModel):
    """Configuration for pii plugin."""

    enabled: bool
    threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Threshold (0.0-1.0, default: None)"
    )
    pii_types_allowed: Optional[List[str]] = None


class SystemPromptPluginConfig(BaseModel):
    """Configuration for system_prompt plugin."""

    enabled: Optional[bool] = None
    system_prompt: Optional[str] = None
    mode: Optional[Literal["replace", "insert"]] = None


class HeaderPair(BaseModel):
    """Header name-value pair."""

    name: str
    value: str


class HeaderMutationPluginConfig(BaseModel):
    """Configuration for header_mutation plugin."""

    add: Optional[List[HeaderPair]] = None
    update: Optional[List[HeaderPair]] = None
    delete: Optional[List[str]] = None


class HallucinationPluginConfig(BaseModel):
    """Configuration for hallucination plugin."""

    enabled: bool
    use_nli: Optional[bool] = None
    hallucination_action: Optional[Literal["header", "body", "none"]] = None
    unverified_factual_action: Optional[Literal["header", "body", "none"]] = None
    include_hallucination_details: Optional[bool] = None


class RouterReplayPluginConfig(BaseModel):
    """Configuration for router_replay plugin.

    The router_replay plugin captures routing decisions and payload snippets
    for later debugging and replay. Records are stored in memory and accessible
    via the /v1/router_replay API endpoint.
    """

    enabled: bool = True
    max_records: int = Field(
        default=200,
        gt=0,
        description="Maximum records in memory (must be > 0, default: 200)",
    )
    capture_request_body: bool = False  # Capture request payloads
    capture_response_body: bool = False  # Capture response payloads
    max_body_bytes: int = Field(
        default=4096,
        gt=0,
        description="Max bytes to capture per body (must be > 0, default: 4096)",
    )


class PluginConfig(BaseModel):
    """Plugin configuration with type validation.

    Configuration schema validation is performed in the validator module
    to ensure proper plugin-specific validation.
    """

    type: PluginType
    configuration: Dict[str, Any]

    def model_dump(self, **kwargs):
        """Override model_dump to serialize PluginType enum as string value."""
        # Use mode='python' to get Python native types, then convert enum
        # Pop mode from kwargs to avoid duplicate argument if caller passes it
        mode = kwargs.pop("mode", "python")
        data = super().model_dump(mode=mode, **kwargs)
        # Convert PluginType enum to its string value for YAML serialization
        if isinstance(data.get("type"), PluginType):
            data["type"] = data["type"].value
        elif hasattr(data.get("type"), "value"):
            data["type"] = data["type"].value
        return data


class Decision(BaseModel):
    """Routing decision configuration."""

    name: str
    description: str
    priority: int
    rules: Rules
    modelRefs: List[ModelRef] = Field(alias="modelRefs")
    algorithm: Optional[AlgorithmConfig] = None  # Multi-model orchestration algorithm
    plugins: Optional[List[PluginConfig]] = []

    class Config:
        populate_by_name = True


class Endpoint(BaseModel):
    """Backend endpoint configuration."""

    name: str
    weight: int
    endpoint: str
    protocol: str = "http"


class ModelPricing(BaseModel):
    """Model pricing configuration."""

    currency: Optional[str] = "USD"
    prompt_per_1m: Optional[float] = 0.0
    completion_per_1m: Optional[float] = 0.0


class Model(BaseModel):
    """Model configuration."""

    name: str
    endpoints: List[Endpoint]
    access_key: Optional[str] = None
    reasoning_family: Optional[str] = None
    pricing: Optional[ModelPricing] = None
    # Model parameter size (e.g., "1b", "7b", "70b", "100m")
    # Used by confidence algorithm to determine model order (smallest first)
    param_size: Optional[str] = None
    # API format: "openai" (default) or "anthropic"
    # When set to "anthropic", the router translates requests to Anthropic Messages API
    api_format: Optional[str] = None


class ReasoningFamily(BaseModel):
    """Reasoning family configuration."""

    type: str
    parameter: str


class ExternalModel(BaseModel):
    """External model configuration."""

    role: str  # "preference", "guardrail", etc.
    provider: str  # "vllm"
    endpoint: str  # "host:port"
    model_name: str
    timeout_seconds: Optional[int] = 30
    parser_type: Optional[str] = "json"
    access_key: Optional[str] = None  # Optional access key for Authorization header


class Providers(BaseModel):
    """Provider configuration."""

    models: List[Model]
    default_model: Optional[str] = None
    reasoning_families: Optional[Dict[str, ReasoningFamily]] = {}
    default_reasoning_effort: Optional[str] = "high"
    external_models: Optional[List[ExternalModel]] = []


class UserConfig(BaseModel):
    """Complete user configuration."""

    version: str
    listeners: List[Listener]
    signals: Optional[Signals] = None
    decisions: List[Decision]
    providers: Providers

    class Config:
        populate_by_name = True
