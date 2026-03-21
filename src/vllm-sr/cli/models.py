"""Pydantic models for vLLM Semantic Router configuration."""

from typing import List, Dict, Any, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field, model_validator

from .algorithms import AlgorithmConfig, ModelRef


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
    threshold: Optional[float] = None
    examples: Optional[List[str]] = None


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
    code_complexity when domain is "computer science"). This is evaluated after
    all signals are computed in parallel, enabling signal dependencies.
    """

    name: str
    threshold: float = 0.1
    hard: ComplexityCandidates
    easy: ComplexityCandidates
    description: Optional[str] = None
    composer: Optional["Rules"] = None  # Forward reference, defined below


class JailbreakRule(BaseModel):
    """Jailbreak detection signal configuration.

    Supports two methods:
    - "classifier" (default): BERT/LoRA-based jailbreak classifier
    - "contrastive": Embedding-based contrastive scoring against jailbreak/benign KBs
    """

    name: str
    threshold: float
    method: Optional[str] = None  # "classifier" (default) or "contrastive"
    include_history: bool = False
    jailbreak_patterns: Optional[list[str]] = (
        None  # Known jailbreak prompts (contrastive KB)
    )
    benign_patterns: Optional[list[str]] = None  # Known benign prompts (contrastive KB)
    description: Optional[str] = None


class PIIRule(BaseModel):
    """PII detection signal configuration."""

    name: str
    threshold: float
    pii_types_allowed: Optional[List[str]] = None
    include_history: bool = False
    description: Optional[str] = None


class ModalityRule(BaseModel):
    """Modality detection signal configuration.

    Classifies whether a prompt requires text (AR), image (DIFFUSION), or both (BOTH).
    Detection configuration is read from modality_detector (InlineModels).
    """

    name: str
    description: Optional[str] = None


class Subject(BaseModel):
    """RBAC subject (user or group) for role binding."""

    kind: str  # "User" or "Group"
    name: str


class RoleBindingRule(BaseModel):
    """RBAC role binding signal configuration.

    Maps subjects (users/groups) to a named role following the Kubernetes RBAC pattern.
    The role name is emitted as a signal of type "authz" in the decision engine.
    User identity is read from x-authz-user-id and x-authz-user-groups headers.
    """

    name: str
    role: str
    subjects: List[Subject]
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
    modality: Optional[List[ModalityRule]] = []
    role_bindings: Optional[List[RoleBindingRule]] = []
    jailbreak: Optional[List[JailbreakRule]] = []
    pii: Optional[List[PIIRule]] = []


class Condition(BaseModel):
    """Routing condition node (leaf or composite boolean expression)."""

    type: Optional[str] = None
    name: Optional[str] = None
    operator: Optional[str] = None
    conditions: Optional[List["Condition"]] = None

    @model_validator(mode="after")
    def validate_node_shape(self):
        has_leaf_fields = self.type is not None or self.name is not None
        has_operator = self.operator is not None

        if has_leaf_fields and has_operator:
            raise ValueError(
                "condition node must be either leaf (type/name) or composite (operator/conditions), not both"
            )

        if has_operator:
            if not self.conditions:
                raise ValueError(
                    "composite condition node requires non-empty conditions"
                )
            op = self.operator.strip().upper()
            if op not in {"AND", "OR", "NOT"}:
                raise ValueError("operator must be one of: AND, OR, NOT")
            if op == "NOT" and len(self.conditions) != 1:
                raise ValueError("NOT operator must have exactly one child condition")
            return self

        # Leaf node validation
        if self.type is None or self.name is None:
            raise ValueError("leaf condition node requires both type and name")
        if self.conditions:
            raise ValueError("leaf condition node cannot define child conditions")
        return self


class Rules(BaseModel):
    """Routing rules.

    Accepts three formats:
    1. Composite: {operator: "AND", conditions: [...]}
    2. Match-all: {operator: "AND"} or {} (no WHEN clause)
    3. Leaf node: {type: "keyword", name: "x"} (single signal ref)

    Formats 2 and 3 are auto-normalised to composite form.
    """

    operator: str = "AND"
    conditions: List[Condition] = []

    @model_validator(mode="before")
    @classmethod
    def normalise_leaf_or_empty(cls, data):
        """Wrap a bare leaf node into AND([leaf]) and fill missing fields."""
        if not isinstance(data, dict):
            return data
        # Leaf node: has type/name but no operator → wrap in AND
        if "type" in data and "operator" not in data:
            leaf = {"type": data["type"], "name": data.get("name", "")}
            return {"operator": "AND", "conditions": [leaf]}
        return data


class PluginType(str, Enum):
    """Supported plugin types."""

    SEMANTIC_CACHE = "semantic-cache"
    SYSTEM_PROMPT = "system_prompt"
    HEADER_MUTATION = "header_mutation"
    HALLUCINATION = "hallucination"
    ROUTER_REPLAY = "router_replay"
    MEMORY = "memory"
    RAG = "rag"
    FAST_RESPONSE = "fast_response"


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


class FastResponsePluginConfig(BaseModel):
    """Configuration for fast_response plugin."""

    message: str


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


class ModelPricing(BaseModel):
    """Model pricing configuration."""

    currency: Optional[str] = "USD"
    prompt_per_1m: Optional[float] = 0.0
    completion_per_1m: Optional[float] = 0.0


class Model(BaseModel):
    """Provider model binding for canonical providers.models entries."""

    name: str
    reasoning_family: Optional[str] = None
    provider_model_id: Optional[str] = None
    backend_refs: List["BackendRef"] = Field(default_factory=list)
    pricing: Optional[ModelPricing] = None
    api_format: Optional[str] = None
    external_model_ids: Optional[Dict[str, str]] = None


class LoRAAdapter(BaseModel):
    """LoRA adapter metadata exposed under routing.modelCards[].loras."""

    name: str
    description: Optional[str] = None


class RoutingModel(BaseModel):
    """Semantic model catalog entry exposed to routing/DSL."""

    name: str
    param_size: Optional[str] = None
    context_window_size: Optional[int] = Field(default=None, ge=1)
    description: Optional[str] = None
    capabilities: Optional[List[str]] = None
    loras: Optional[List[LoRAAdapter]] = None
    tags: Optional[List[str]] = None
    quality_score: Optional[float] = Field(default=None, ge=0, le=1)
    modality: Optional[str] = None


class ReasoningFamily(BaseModel):
    """Reasoning family configuration."""

    type: str
    parameter: str


class BackendRef(BaseModel):
    """Inline backend access details carried under providers.models[].backend_refs."""

    name: Optional[str] = None
    endpoint: Optional[str] = None
    protocol: str = "http"
    weight: int = 1
    type: Optional[str] = None
    base_url: Optional[str] = None
    provider: Optional[str] = None
    auth_header: Optional[str] = None
    auth_prefix: Optional[str] = None
    extra_headers: Optional[Dict[str, str]] = None
    api_version: Optional[str] = None
    chat_path: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None

    def resolve_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            import os

            return os.getenv(self.api_key_env)
        return None


class ProviderDefaults(BaseModel):
    """Provider-wide defaults that should not be mixed into per-model access bindings."""

    default_model: Optional[str] = None
    reasoning_families: Optional[Dict[str, "ReasoningFamily"]] = Field(
        default_factory=dict
    )
    default_reasoning_effort: Optional[str] = "high"


class Providers(BaseModel):
    """Provider configuration."""

    defaults: ProviderDefaults = Field(default_factory=ProviderDefaults)
    models: List[Model] = Field(default_factory=list)

    @property
    def default_model(self) -> Optional[str]:
        return self.defaults.default_model

    @property
    def reasoning_families(self) -> Dict[str, "ReasoningFamily"]:
        return self.defaults.reasoning_families or {}

    @property
    def default_reasoning_effort(self) -> Optional[str]:
        return self.defaults.default_reasoning_effort


class Routing(BaseModel):
    """Canonical routing block."""

    model_cards: List[RoutingModel] = Field(default_factory=list, alias="modelCards")
    signals: Signals = Field(default_factory=Signals)
    decisions: List[Decision] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class MemoryMilvusConfig(BaseModel):
    """Milvus configuration for memory storage."""

    address: str
    collection: str = "agentic_memory"
    dimension: int = 384


class MemoryConfig(BaseModel):
    """Agentic Memory configuration for cross-session memory.

    Query rewriting and fact extraction are enabled by adding external_models
    with role="memory_rewrite" or role="memory_extraction".
    See global.model_catalog.external configuration for details.

    The embedding_model is auto-detected from global.model_catalog.embeddings.semantic if not specified.
    Priority: mmbert > bert > multimodal > qwen3 > gemma
    """

    enabled: bool = True
    auto_store: bool = False  # Auto-store extracted facts after each response
    milvus: Optional[MemoryMilvusConfig] = None
    # Embedding model to use for memory vectors
    # Options: "bert", "mmbert", "multimodal", "qwen3", "gemma"
    # If not set, auto-detected from global.model_catalog.embeddings.semantic (mmbert preferred)
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
    multimodal_model_path: Optional[str] = Field(
        None,
        description="Path to multi-modal embedding model (text/image/audio)",
    )
    bert_model_path: Optional[str] = Field(
        None,
        description="Path to BERT/MiniLM model (recommended for memory retrieval)",
    )
    embedding_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Embedding classifier tuning (model_type/target_dimension/top_k/etc.)",
    )
    use_cpu: bool = Field(True, description="Use CPU for inference")

    class Config:
        # Preserve advanced nested fields when users pass through custom config blocks.
        extra = "allow"


class UserConfig(BaseModel):
    """Canonical v0.3 user configuration."""

    version: str
    listeners: List[Listener] = Field(default_factory=list)
    providers: Providers = Field(default_factory=Providers)
    routing: Routing = Field(default_factory=Routing)
    global_: Optional[Dict[str, Any]] = Field(default=None, alias="global")
    setup: Optional[Dict[str, Any]] = None

    @property
    def signals(self) -> Signals:
        return self.routing.signals

    @property
    def decisions(self) -> List[Decision]:
        return self.routing.decisions

    class Config:
        populate_by_name = True
        extra = "forbid"


# Resolve forward references for recursive condition trees.
Condition.model_rebuild()
Model.model_rebuild()
