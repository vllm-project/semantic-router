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


class JailbreakRule(BaseModel):
    """Jailbreak detection signal configuration."""

    name: str
    threshold: float
    include_history: bool = False
    description: Optional[str] = None


class PIIRule(BaseModel):
    """PII detection signal configuration."""

    name: str
    threshold: float
    pii_types_allowed: Optional[List[str]] = None
    include_history: bool = False
    description: Optional[str] = None


class Signals(BaseModel):
    """All signal configurations."""

    keywords: Optional[List[KeywordSignal]] = []
    embeddings: Optional[List[EmbeddingSignal]] = []
    domains: Optional[List[Domain]] = []
    fact_check: Optional[List[FactCheck]] = []
    user_feedbacks: Optional[List[UserFeedback]] = []
    preferences: Optional[List[Preference]] = []
    language: Optional[List[Language]] = []
    context: Optional[List[ContextRule]] = []
    complexity: Optional[List[ComplexityRule]] = []
    jailbreak: Optional[List[JailbreakRule]] = []
    pii: Optional[List[PIIRule]] = []


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


class PluginType(str, Enum):
    """Supported plugin types."""

    SEMANTIC_CACHE = "semantic-cache"
    JAILBREAK = "jailbreak"
    PII = "pii"
    SYSTEM_PROMPT = "system_prompt"
    HEADER_MUTATION = "header_mutation"
    HALLUCINATION = "hallucination"
    ROUTER_REPLAY = "router_replay"
    MEMORY = "memory"
    RAG = "rag"


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


class MemoryPluginConfig(BaseModel):
    """Configuration for memory plugin (per-decision memory settings)."""

    enabled: bool = True
    retrieval_limit: Optional[int] = Field(
        default=None,
        gt=0,
        description="Max memories to retrieve (default: use global config)",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Min similarity score (0.0-1.0, default: use global config)",
    )
    auto_store: Optional[bool] = Field(
        default=None,
        description="Auto-extract memories from conversation (default: use request config)",
    )


class RAGPluginConfig(BaseModel):
    """Configuration for RAG (Retrieval-Augmented Generation) plugin.

    The RAG plugin retrieves relevant context from external knowledge bases
    and injects it into the LLM request.

    Supported backends:
    - milvus: Milvus vector database (reuses semantic cache connection)
    - external_api: External REST API (OpenAI, Pinecone, Weaviate, Elasticsearch)
    - mcp: MCP tool-based retrieval
    - openai: OpenAI file_search with vector stores
    - hybrid: Multi-backend with fallback strategy
    """

    # Required: Enable RAG retrieval
    enabled: bool = Field(..., description="Enable RAG retrieval for this decision")

    # Required: Backend type (milvus, external_api, mcp, openai, hybrid)
    backend: str = Field(
        ...,
        description="Retrieval backend: milvus, external_api, mcp, openai, hybrid",
    )

    # Optional: Similarity threshold (0.0-1.0)
    # Only documents with similarity >= threshold will be retrieved
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for retrieval (0.0-1.0)",
    )

    # Optional: Number of top-k documents to retrieve
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of top-k documents to retrieve",
    )

    # Optional: Maximum context length (in characters)
    max_context_length: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum context length to inject (characters)",
    )

    # Optional: Context injection mode
    # - "tool_role": Inject as tool role messages (compatible with hallucination detection)
    # - "system_prompt": Prepend to system prompt
    injection_mode: Optional[str] = Field(
        default=None,
        description="Injection mode: tool_role (default) or system_prompt",
    )

    # Optional: Backend-specific configuration
    # Structure depends on backend type (see Go: rag_plugin.go lines 64-174)
    backend_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Backend-specific configuration",
    )

    # Optional: Fallback behavior on retrieval failure
    # - "skip": Continue without context (default)
    # - "block": Return error response
    # - "warn": Continue with warning header
    on_failure: Optional[str] = Field(
        default=None,
        description="On failure: skip (default), block, or warn",
    )

    # Optional: Cache retrieved results
    cache_results: Optional[bool] = Field(
        default=None,
        description="Cache retrieved results",
    )

    # Optional: Cache TTL (seconds)
    cache_ttl_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Cache TTL in seconds",
    )

    # Optional: Minimum confidence for triggering retrieval
    min_confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for triggering retrieval",
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

    # Model selection fields (for RouterDC and quality-based selection)
    # Description of model capabilities for RouterDC embedding matching
    description: Optional[str] = None
    # Structured capability tags (e.g., ["coding", "math", "reasoning"])
    capabilities: Optional[List[str]] = None
    # Quality score for AutoMix selection (0.0-1.0, default: 0.8)
    quality_score: Optional[float] = Field(default=None, ge=0, le=1)


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


class MemoryMilvusConfig(BaseModel):
    """Milvus configuration for memory storage."""

    address: str
    collection: str = "agentic_memory"
    dimension: int = 384


class MemoryConfig(BaseModel):
    """Agentic Memory configuration for cross-session memory.

    Query rewriting and fact extraction are enabled by adding external_models
    with role="memory_rewrite" or role="memory_extraction".
    See external_models configuration in providers section for details.

    The embedding_model is auto-detected from embedding_models if not specified.
    Priority: bert > mmbert > qwen3 > gemma
    """

    enabled: bool = True
    auto_store: bool = False  # Auto-store extracted facts after each response
    milvus: Optional[MemoryMilvusConfig] = None
    # Embedding model to use for memory vectors
    # Options: "bert", "mmbert", "qwen3", "gemma"
    # If not set, auto-detected from embedding_models section (bert preferred)
    embedding_model: Optional[str] = None
    default_retrieval_limit: int = 5
    default_similarity_threshold: float = 0.70
    extraction_batch_size: int = 10  # Extract every N turns


class EmbeddingModelsConfig(BaseModel):
    """Embedding models configuration for memory and semantic features."""

    qwen3_model_path: Optional[str] = Field(
        None, description="Path to Qwen3-Embedding model"
    )
    gemma_model_path: Optional[str] = Field(
        None, description="Path to EmbeddingGemma model"
    )
    mmbert_model_path: Optional[str] = Field(
        None, description="Path to mmBERT 2D Matryoshka model"
    )
    bert_model_path: Optional[str] = Field(
        None,
        description="Path to BERT/MiniLM model (recommended for memory retrieval)",
    )
    use_cpu: bool = Field(True, description="Use CPU for inference")


class UserConfig(BaseModel):
    """Complete user configuration."""

    version: str
    listeners: List[Listener]
    signals: Optional[Signals] = None
    decisions: List[Decision]
    providers: Providers
    memory: Optional[MemoryConfig] = None  # Agentic Memory config
    embedding_models: Optional[EmbeddingModelsConfig] = (
        None  # Embedding models for memory
    )

    class Config:
        populate_by_name = True
