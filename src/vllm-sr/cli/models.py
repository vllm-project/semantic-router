"""Pydantic models for vLLM Semantic Router configuration."""

import json
import math
import warnings
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

from .algorithms import AlgorithmConfig, ModelRef
from .config_contract import (
    CLASSIFIER_TYPE_LLM,
    CLASSIFIER_TYPE_LOCAL,
    ClassifierSignalType,
    UnknownPolicy,
)

RoutingStrategy = Literal["priority", "confidence"]
LOCAL_CLASSIFIER_LABEL_COUNT = 2
SEQUENCE_CLASSIFIER_MIN_LABEL_COUNT = 2
PROMPT_MIN_CANDIDATES = 2
MAX_DECISION_ANNOTATIONS = 32
MAX_DECISION_ANNOTATION_BYTES = 4096


class Listener(BaseModel):
    """Network listener configuration."""

    name: str
    address: str
    port: int
    timeout: Optional[str] = "300s"
    api_keys: Optional[List[str]] = Field(
        default=None,
        description="Bearer tokens required to call this listener. "
        "If set, requests without 'Authorization: Bearer <key>' matching one of these "
        "values are rejected with HTTP 401.",
    )


class KeywordSignal(BaseModel):
    """Keyword-based signal configuration."""

    name: str
    operator: str
    keywords: List[str]
    case_sensitive: bool = False


class EmbeddingSignal(BaseModel):
    """Embedding-based signal configuration.

    The ``query_modality`` field declares which modality of incoming request
    payload the embedding rule's query is computed from. It defaults to
    ``"text"`` when omitted, preserving existing behavior. ``"image"`` and
    ``"audio"`` require ``global.model_catalog.embeddings.semantic.embedding_config.model_type=multimodal``
    in the router config so the query and candidate embeddings land in the same
    shared space.
    """

    name: str
    threshold: float
    candidates: List[str]
    aggregation_method: str = "max"
    query_modality: Optional[Literal["text", "image", "audio"]] = None


class ProjectionPartition(BaseModel):
    """Partition metadata coordinating mutually exclusive routing signals."""

    name: str
    semantics: str
    members: List[str]
    temperature: Optional[float] = Field(default=None, gt=0)
    default: str


class ProjectionScoreInput(BaseModel):
    """One weighted signal contribution to a derived projection score.

    Supported value_source modes:
      - "binary" (default): contributes match/miss fixed values.
      - "confidence": contributes the signal confidence when matched.
      - "raw": contributes the raw numeric value from SignalValues.
      - "score": contributes the numeric score exposed by a knowledge-base metric.
    """

    type: str
    name: Optional[str] = None
    kb: Optional[str] = None
    metric: Optional[str] = None
    weight: float
    value_source: Optional[Literal["binary", "confidence", "raw", "score"]] = None
    match: Optional[float] = None
    miss: Optional[float] = None


class ProjectionScore(BaseModel):
    """Weighted derived score over existing routing signals."""

    name: str
    method: str
    inputs: List[ProjectionScoreInput]


class ProjectionMappingCalibration(BaseModel):
    """Confidence calibration for a projection mapping output band."""

    method: str
    slope: Optional[float] = None


class ProjectionMappingOutput(BaseModel):
    """One named threshold band emitted by a projection mapping."""

    name: str
    lt: Optional[float] = None
    lte: Optional[float] = None
    gt: Optional[float] = None
    gte: Optional[float] = None


class ProjectionMapping(BaseModel):
    """Maps a derived score into named routing outputs."""

    name: str
    source: str
    method: str
    calibration: Optional[ProjectionMappingCalibration] = None
    outputs: List[ProjectionMappingOutput]


class Projections(BaseModel):
    """Derived routing surfaces that sit alongside base signals."""

    partitions: Optional[List[ProjectionPartition]] = []
    scores: Optional[List[ProjectionScore]] = []
    mappings: Optional[List[ProjectionMapping]] = []


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


class StructureSource(BaseModel):
    """Source selector for structure-based signals."""

    model_config = ConfigDict(extra="forbid")

    type: str
    pattern: Optional[str] = None
    keywords: Optional[List[str]] = None
    case_sensitive: bool = False
    sequences: Optional[List[List[str]]] = None


class StructureFeature(BaseModel):
    """Typed request-shape feature extractor."""

    model_config = ConfigDict(extra="forbid")

    type: str
    source: StructureSource


class NumericPredicate(BaseModel):
    """Numeric threshold predicate for structure signals."""

    model_config = ConfigDict(extra="forbid")

    gt: Optional[float] = None
    gte: Optional[float] = None
    lt: Optional[float] = None
    lte: Optional[float] = None

    @model_validator(mode="after")
    def validate_contract(self):
        for value in (self.gt, self.gte, self.lt, self.lte):
            if value is not None and not math.isfinite(value):
                raise ValueError("numeric predicate values must be finite")
        if all(value is None for value in (self.gt, self.gte, self.lt, self.lte)):
            raise ValueError("numeric predicate requires at least one comparator")
        if self.gt is not None and self.gte is not None:
            raise ValueError("numeric predicate cannot set both gt and gte")
        if self.lt is not None and self.lte is not None:
            raise ValueError("numeric predicate cannot set both lt and lte")
        lower = self.gt if self.gt is not None else self.gte
        upper = self.lt if self.lt is not None else self.lte
        if lower is not None and upper is not None:
            strict = self.gt is not None or self.lt is not None
            if lower > upper or (lower == upper and strict):
                raise ValueError("numeric predicate defines an empty range")
        return self


class StructureRule(BaseModel):
    """Request-shape routing signal configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    feature: StructureFeature
    predicate: Optional[NumericPredicate] = None


class ConversationSource(BaseModel):
    """Source selector for conversation-shape signals."""

    model_config = ConfigDict(extra="forbid")

    type: str
    role: Optional[str] = None


class ConversationFeature(BaseModel):
    """Typed conversation-shape feature extractor."""

    model_config = ConfigDict(extra="forbid")

    type: str
    source: ConversationSource


class ConversationRule(BaseModel):
    """Conversation-shape routing signal configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    feature: ConversationFeature
    predicate: Optional[NumericPredicate] = None


class EventRule(BaseModel):
    """Structured event metadata routing signal configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    event_types: Optional[List[str]] = None
    severities: Optional[List[str]] = None
    action_codes: Optional[List[str]] = None
    temporal: bool = False
    description: Optional[str] = None


class ComplexityCandidates(BaseModel):
    """Complexity candidates configuration."""

    candidates: List[str]
    image_candidates: Optional[List[str]] = None


class PrototypeScoringConfig(BaseModel):
    """Prototype-bank construction and scoring controls for embedding-backed signals."""

    enabled: Optional[bool] = None
    cluster_similarity_threshold: Optional[float] = None
    max_prototypes: Optional[int] = None
    best_weight: Optional[float] = None
    top_m: Optional[int] = None
    margin_threshold: Optional[float] = None


class EmbeddingClassifierConfig(BaseModel):
    """Embedding classifier tuning, including prototype-aware label scoring controls."""

    model_config = ConfigDict(extra="allow")

    backend: Optional[str] = None
    model_type: Optional[str] = None
    preload_embeddings: Optional[bool] = None
    target_dimension: Optional[int] = None
    target_layer: Optional[int] = None
    enable_soft_matching: Optional[bool] = None
    top_k: Optional[int] = None
    min_score_threshold: Optional[float] = None
    prototype_scoring: Optional[PrototypeScoringConfig] = None


class EmbeddingEndpointConfig(BaseModel):
    """External embedding provider endpoint configuration."""

    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout_seconds: Optional[int] = Field(default=None, ge=0)
    max_retries: Optional[int] = Field(default=None, ge=0)
    max_response_bytes: Optional[int] = Field(default=None, ge=0)
    dimensions: Optional[int] = Field(default=None, ge=1)


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
    # "request" (default) scores the prompt; "response" scores the model's
    # output once it has answered, for the selected decision's
    # response_jailbreak plugin to enforce on.
    direction: Optional[Literal["request", "response"]] = None
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


class KBSignalTarget(BaseModel):
    """Binding target for a named knowledge base."""

    kind: Literal["label", "group"]
    value: str


class KBSignal(BaseModel):
    """Knowledge-base signal bound to a named global KB instance."""

    name: str
    kb: str
    target: KBSignalTarget
    match: Optional[Literal["best", "threshold"]] = None


class Reask(BaseModel):
    """History-aware repeated-question dissatisfaction signal."""

    name: str
    description: Optional[str] = None
    threshold: Optional[float] = None
    lookback_turns: Optional[int] = None


class MetadataPredicate(BaseModel):
    """Comparator for untrusted request metadata."""

    model_config = ConfigDict(extra="forbid")

    equals: Optional[str] = None
    in_: Optional[List[str]] = Field(default=None, alias="in")
    exists: Optional[bool] = None

    @model_validator(mode="after")
    def validate_comparator(self):
        configured = sum(
            (
                self.equals is not None,
                bool(self.in_),
                self.exists is not None,
            )
        )
        if configured != 1:
            raise ValueError("exactly one of equals, in, or exists is required")
        return self


class MetadataRule(BaseModel):
    """Matches caller-provided request metadata without granting authorization."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    key: str
    predicate: MetadataPredicate

    @model_validator(mode="after")
    def validate_canonical_names(self):
        if not self.name.strip() or self.name != self.name.strip():
            raise ValueError("metadata signal name must be nonempty and trimmed")
        if not self.key.strip() or self.key != self.key.strip():
            raise ValueError("metadata key must be nonempty and trimmed")
        return self


class InputModalityRule(BaseModel):
    """Deterministic structural input-modality presence signal."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    modality: Literal["text", "image", "audio", "video"]

    @model_validator(mode="after")
    def validate_canonical_names(self):
        if not self.name.strip() or self.name != self.name.strip():
            raise ValueError("input_modality signal name must be nonempty and trimmed")
        return self


class ClassifierSignal(BaseModel):
    """Generic label-score classifier signal."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    type: ClassifierSignalType
    model: Optional[str] = None
    model_path: Optional[str] = None
    labels: List[str]
    instructions: Optional[str] = None
    use_cpu: bool = False

    @model_validator(mode="after")
    def validate_classifier(self):
        if not self.name.strip() or self.name != self.name.strip():
            raise ValueError("classifier signal name must be nonempty and trimmed")
        if ":" in self.name:
            raise ValueError("classifier signal name cannot contain ':'")
        if not self.labels or any(not label.strip() for label in self.labels):
            raise ValueError("labels cannot be empty")
        if any(label != label.strip() or ":" in label for label in self.labels):
            raise ValueError("labels must be trimmed and cannot contain ':'")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels cannot contain duplicates")

        if self.type == CLASSIFIER_TYPE_LOCAL:
            self._validate_local()
        elif self.type == CLASSIFIER_TYPE_LLM:
            self._validate_llm()
        else:
            self._validate_sequence()
        return self

    def _validate_local(self):
        if not self.model_path:
            raise ValueError("local classifiers require model_path")
        if len(self.labels) != LOCAL_CLASSIFIER_LABEL_COUNT:
            raise ValueError("local classifiers require exactly two labels")
        if self.model or self.instructions:
            raise ValueError("local classifiers do not accept model or instructions")

    def _validate_llm(self):
        if not self.model:
            raise ValueError("llm classifiers require model")
        if not self.instructions:
            raise ValueError("llm classifiers require instructions")
        if self.model_path or self.use_cpu:
            raise ValueError("llm classifiers do not accept model_path or use_cpu")

    def _validate_sequence(self):
        if not self.model:
            raise ValueError("sequence_classifier classifiers require model")
        if len(self.labels) < SEQUENCE_CLASSIFIER_MIN_LABEL_COUNT:
            raise ValueError(
                "sequence_classifier classifiers require at least two labels"
            )
        if self.model_path or self.use_cpu or self.instructions:
            raise ValueError(
                "sequence_classifier classifiers do not accept model_path, use_cpu or instructions"
            )


class Signals(BaseModel):
    """All signal configurations."""

    keywords: Optional[List[KeywordSignal]] = []
    embeddings: Optional[List[EmbeddingSignal]] = []
    domains: Optional[List[Domain]] = []
    fact_check: Optional[List[FactCheck]] = []
    user_feedbacks: Optional[List[UserFeedback]] = []
    reasks: Optional[List[Reask]] = []
    preferences: Optional[List[Preference]] = []
    language: Optional[List[Language]] = []
    context: Optional[List[ContextRule]] = []
    structure: Optional[List[StructureRule]] = []
    complexity: Optional[List[ComplexityRule]] = []
    modality: Optional[List[ModalityRule]] = []
    role_bindings: Optional[List[RoleBindingRule]] = []
    jailbreak: Optional[List[JailbreakRule]] = []
    pii: Optional[List[PIIRule]] = []
    kb: Optional[List[KBSignal]] = []
    conversation: Optional[List[ConversationRule]] = []
    events: Optional[List[EventRule]] = []
    metadata: Optional[List[MetadataRule]] = []
    classifiers: Optional[List[ClassifierSignal]] = []
    input_modality: Optional[List[InputModalityRule]] = []

    @model_validator(mode="after")
    def validate_rule_names(self):
        for family in type(self).model_fields:
            seen: set[str] = set()
            for signal in getattr(self, family) or []:
                name = (
                    signal.name.lower()
                    if family in {"metadata", "classifiers", "input_modality"}
                    else signal.name
                )
                if name in seen:
                    raise ValueError(
                        f"{family} signal names must be unique within a recipe"
                    )
                seen.add(name)
        return self


class Condition(BaseModel):
    """Routing condition node (leaf or composite boolean expression)."""

    type: Optional[str] = None
    name: Optional[str] = None
    label: Optional[str] = None
    predicate: Optional[NumericPredicate] = None
    on_error: Optional[Literal["no_match", "match"]] = None
    on_unknown: Optional[UnknownPolicy] = None
    operator: Optional[str] = None
    conditions: Optional[List["Condition"]] = None

    @model_validator(mode="after")
    def validate_node_shape(self):
        has_leaf_fields = any(
            (
                self.type is not None,
                self.name is not None,
                self.label is not None,
                self.predicate is not None,
                self.on_error is not None,
            )
        )
        has_operator = self.operator is not None

        if has_leaf_fields and has_operator:
            raise ValueError(
                "condition node must be either leaf (type/name) or composite (operator/conditions), not both"
            )

        if has_operator:
            return self._validate_composite_node()
        return self._validate_leaf_node()

    def _validate_composite_node(self):
        if not self.conditions:
            raise ValueError("composite condition node requires non-empty conditions")
        op = self.operator.strip().upper()
        if op not in {"AND", "OR", "NOT"}:
            raise ValueError("operator must be one of: AND, OR, NOT")
        if op == "NOT" and len(self.conditions) != 1:
            raise ValueError("NOT operator must have exactly one child condition")
        if self.on_unknown is not None:
            raise ValueError("on_unknown is only valid on the root rules node")
        return self

    def _validate_leaf_node(self):
        if self.type is None or self.name is None:
            raise ValueError("leaf condition node requires both type and name")
        if self.conditions:
            raise ValueError("leaf condition node cannot define child conditions")
        if self.label is not None and self.type != "classifier":
            raise ValueError("label is only valid for classifier conditions")
        if self.type == "classifier" and (self.label is None or self.predicate is None):
            raise ValueError("classifier conditions require label and predicate")
        if self.on_error is not None and self.type != "classifier":
            raise ValueError("on_error is only valid for classifier conditions")
        if self.on_unknown is not None:
            raise ValueError("on_unknown is only valid on the root rules node")
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
    conditions: List[Condition] = Field(default_factory=list)
    on_unknown: Optional[UnknownPolicy] = None

    @model_validator(mode="before")
    @classmethod
    def normalise_leaf_or_empty(cls, data):
        """Wrap a bare leaf node into AND([leaf]) and fill missing fields."""
        if not isinstance(data, dict):
            return data
        # Leaf node: has type/name but no operator → wrap in AND
        if "type" in data and "operator" not in data:
            leaf = {
                key: data[key]
                for key in ("type", "name", "label", "predicate", "on_error")
                if key in data
            }
            leaf.setdefault("name", "")
            rules = {"operator": "AND", "conditions": [leaf]}
            if "on_unknown" in data:
                rules["on_unknown"] = data["on_unknown"]
            return rules
        return data


class PluginType(str, Enum):
    """Supported plugin types."""

    RESPONSE_CACHE = "response_cache"
    SYSTEM_PROMPT = "system_prompt"
    HEADER_MUTATION = "header_mutation"
    HALLUCINATION = "hallucination"
    ROUTER_REPLAY = "router_replay"
    MEMORY = "memory"
    RAG = "rag"
    FAST_RESPONSE = "fast_response"
    REQUEST_PARAMS = "request_params"
    RESPONSE_JAILBREAK = "response_jailbreak"
    TOOLS = "tools"
    TOOL_SELECTION = "tool_selection"
    CONTEXT_COMPRESSION = "context_compression"


class ResponseCacheSemanticConfig(BaseModel):
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ResponseCacheRequestControlsConfig(BaseModel):
    enabled: bool = False
    header: Optional[str] = None
    allowed: List[Literal["no-cache", "no-store", "bypass", "max-age", "ttl"]] = Field(
        default_factory=list
    )
    max_ttl_seconds: Optional[int] = Field(default=None, ge=0)


class ResponseCachePersonalizedConfig(BaseModel):
    mode: Literal["disabled", "exact"] = "disabled"


class ResponseCacheRevisionConfig(BaseModel):
    cache_epoch: Optional[str] = None
    model_revision: Optional[str] = None
    prompt_revision: Optional[str] = None
    policy_revision: Optional[str] = None


class ResponseCachePluginConfig(BaseModel):
    """Configuration for response_cache plugin."""

    enabled: bool
    mode: Literal["semantic", "exact", "exact_then_semantic"] = "semantic"
    scope: Literal["user", "team", "tenant", "global"] = "user"
    semantic: Optional[ResponseCacheSemanticConfig] = None
    request_controls: Optional[ResponseCacheRequestControlsConfig] = None
    personalized: Optional[ResponseCachePersonalizedConfig] = None
    revision: Optional[ResponseCacheRevisionConfig] = None
    # Deprecated flat compatibility fields.
    allow_request_controls: bool = False
    control_header: Optional[str] = None
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Similarity threshold (0.0-1.0, default: None)",
    )
    ttl_seconds: Optional[int] = Field(
        default=None, ge=0, description="TTL in seconds (must be >= 0, default: None)"
    )

    @model_validator(mode="after")
    def validate_compatibility_fields(self):
        if self.semantic is not None and self.similarity_threshold is not None:
            raise ValueError(
                "semantic.similarity_threshold conflicts with similarity_threshold"
            )
        if self.request_controls is not None and (
            self.allow_request_controls or self.control_header is not None
        ):
            raise ValueError(
                "request_controls conflicts with deprecated request-control fields"
            )
        return self


SemanticCachePluginConfig = ResponseCachePluginConfig


CompressionTokenLimit = Literal["auto"] | int


class ContextCompressionBudgetConfig(BaseModel):
    trigger_tokens: Optional[CompressionTokenLimit] = None
    target_tokens: Optional[CompressionTokenLimit] = None
    reserve_output_tokens: Optional[CompressionTokenLimit] = None

    @model_validator(mode="after")
    def validate_budget(self):
        values = (
            self.trigger_tokens,
            self.target_tokens,
            self.reserve_output_tokens,
        )
        if any(isinstance(value, int) and value < 0 for value in values):
            raise ValueError("compression token limits cannot be negative")
        if (
            isinstance(self.trigger_tokens, int)
            and isinstance(self.target_tokens, int)
            and self.trigger_tokens > 0
            and self.target_tokens >= self.trigger_tokens
        ):
            raise ValueError("budget.target_tokens must be less than trigger_tokens")
        return self


class ContextCompressionTargetConfig(BaseModel):
    mode: Literal["preserve", "extractive", "recoverable"] = "preserve"
    min_tokens: Optional[int] = Field(default=None, ge=0)
    target_tokens: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_target_budget(self):
        if (
            self.min_tokens
            and self.target_tokens
            and self.target_tokens >= self.min_tokens
        ):
            raise ValueError("target_tokens must be less than min_tokens")
        return self


class ContextCompressionTargetsConfig(BaseModel):
    tool_outputs: ContextCompressionTargetConfig = Field(
        default_factory=lambda: ContextCompressionTargetConfig(
            mode="extractive", min_tokens=2000, target_tokens=1000
        )
    )
    history: ContextCompressionTargetConfig = Field(
        default_factory=ContextCompressionTargetConfig
    )
    rag: ContextCompressionTargetConfig = Field(
        default_factory=ContextCompressionTargetConfig
    )
    memory: ContextCompressionTargetConfig = Field(
        default_factory=ContextCompressionTargetConfig
    )


class ContextCompressionScoringConfig(BaseModel):
    method: Literal["bm25", "embedding", "hybrid"] = "bm25"
    embedding_model_ref: Optional[str] = None

    @model_validator(mode="after")
    def validate_embedding_model(self):
        if self.method != "bm25" and not self.embedding_model_ref:
            raise ValueError("embedding_model_ref is required for embedding scoring")
        return self


class ContextCompressionRecoveryConfig(BaseModel):
    enabled: bool = False
    store: Optional[Literal["redis", "valkey", "response_cache"]] = None
    ttl_seconds: int = Field(default=900, ge=0)
    max_bytes_per_request: int = Field(default=10 * 1024 * 1024, ge=0)
    max_total_bytes: int = Field(default=256 * 1024 * 1024, ge=0)
    max_retrievals: int = Field(default=8, ge=0)

    @model_validator(mode="after")
    def validate_store(self):
        if self.enabled and self.store is None:
            raise ValueError("recovery.store is required when recovery is enabled")
        return self


class ContextCompressionRequestControlsConfig(BaseModel):
    enabled: bool = False
    header: Optional[str] = None
    allowed: List[Literal["bypass", "target"]] = Field(default_factory=list)
    max_target_tokens: int = Field(default=16000, ge=0)


class ContextCompressionPluginConfig(BaseModel):
    """Configuration for context_compression plugin."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    mode: Literal["auto", "always"] = "auto"
    budget: Optional[ContextCompressionBudgetConfig] = None
    targets: Optional[ContextCompressionTargetsConfig] = None
    scoring: Optional[ContextCompressionScoringConfig] = None
    recovery: Optional[ContextCompressionRecoveryConfig] = None
    request_controls: Optional[ContextCompressionRequestControlsConfig] = None
    failure_mode: Literal["fail_open", "fail_closed"] = "fail_open"


class FastResponsePluginConfig(BaseModel):
    """Configuration for fast_response plugin."""

    message: str


class RequestParamsPluginConfig(BaseModel):
    """Configuration for request_params plugin."""

    blocked_params: Optional[List[str]] = None
    max_tokens_limit: Optional[int] = Field(default=None, ge=1)
    max_n: Optional[int] = Field(default=None, ge=1)
    strip_unknown: Optional[bool] = None


class ResponseJailbreakPluginConfig(BaseModel):
    """Configuration for response_jailbreak plugin."""

    enabled: bool
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    action: Optional[Literal["block", "header", "none"]] = None


class ToolsDynamicRetrievalWeights(BaseModel):
    """Per-source weights for decision-scoped dynamic tool retrieval."""

    semantic: Optional[float] = None
    history: Optional[float] = None
    decision_prior: Optional[float] = None
    repetition_penalty: Optional[float] = None


class ToolsDynamicRetrievalConfig(BaseModel):
    """History-aware retrieval settings owned by the tools plugin."""

    enabled: bool
    strategy: str = "semantic_only"
    history_window: Optional[int] = None
    weights: Optional[ToolsDynamicRetrievalWeights] = None
    min_history_confidence: Optional[float] = None
    fallback_on_low_confidence: Optional[bool] = None

    @model_validator(mode="after")
    def validate_enabled_contract(self):
        if not self.enabled:
            return self
        if self.strategy not in ("", "semantic_only", "hybrid_history"):
            raise ValueError(
                "dynamic_retrieval.strategy must be semantic_only or hybrid_history"
            )
        if self.strategy == "hybrid_history" and (
            self.history_window is None or self.history_window < 1
        ):
            raise ValueError(
                "history_window must be at least 1 when dynamic_retrieval.strategy=hybrid_history"
            )
        if self.min_history_confidence is not None and not (
            0.0 <= self.min_history_confidence <= 1.0
        ):
            raise ValueError("min_history_confidence must be between 0.0 and 1.0")
        if self.weights is not None:
            for name, value in self.weights.model_dump(exclude_none=True).items():
                if value < 0.0:
                    raise ValueError(
                        f"dynamic_retrieval.weights.{name} must be non-negative"
                    )
        return self


class ToolsPluginConfig(BaseModel):
    """Configuration for tools plugin."""

    enabled: bool
    mode: str = "passthrough"
    semantic_selection: Optional[bool] = None
    allow_tools: Optional[List[str]] = None
    block_tools: Optional[List[str]] = None
    strip_tool_history: Optional[bool] = None
    strategy: Optional[str] = None
    dynamic_retrieval: Optional[ToolsDynamicRetrievalConfig] = None

    @model_validator(mode="after")
    def validate_mode_contract(self):
        if not self.enabled:
            return self
        if self.mode not in ("none", "passthrough", "filtered"):
            raise ValueError("mode must be none, passthrough, or filtered")
        has_filters = bool(self.allow_tools or self.block_tools)
        if self.mode == "filtered" and not has_filters:
            raise ValueError("mode=filtered requires allow_tools or block_tools")
        if self.mode != "filtered" and has_filters:
            raise ValueError("allow_tools and block_tools require mode=filtered")
        if self.strip_tool_history and self.mode != "none":
            raise ValueError("strip_tool_history requires mode=none")
        return self


class ToolFilteringWeights(BaseModel):
    """Weights for tool_selection advanced_filtering combined score.

    Mirrors the Go-side `config.ToolFilteringWeights`. All weights are optional
    pointers on the Go side; non-negative is the only structural constraint.
    Cross-field rules (e.g. weights-all-zero) are enforced by the Go validator.
    """

    embed: Optional[float] = Field(default=None, ge=0.0)
    lexical: Optional[float] = Field(default=None, ge=0.0)
    tag: Optional[float] = Field(default=None, ge=0.0)
    name: Optional[float] = Field(default=None, ge=0.0)
    category: Optional[float] = Field(default=None, ge=0.0)


class HybridHistoryConfig(BaseModel):
    """Sub-config for `retrieval_strategy: hybrid_history`.

    Mirrors the Go-side `config.HybridHistoryToolRetrievalConfig`. Cross-field
    semantic rules (e.g. min_history_steps > history_horizon) are enforced by
    the Go validator; this layer only mirrors the schema surface so Pydantic
    stops dropping the subtree.
    """

    history_horizon: Optional[int] = Field(default=None, ge=0)
    min_history_steps: Optional[int] = Field(default=None, ge=0)
    history_confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    weight_semantic: Optional[float] = Field(default=None, ge=0.0)
    weight_history_transition: Optional[float] = Field(default=None, ge=0.0)
    weight_decision_prior: Optional[float] = Field(default=None, ge=0.0)
    repetition_penalty_strength: Optional[float] = Field(default=None, ge=0.0)


class AdvancedToolFilteringConfig(BaseModel):
    """Advanced filtering layer applied on top of semantic retrieval.

    Mirrors the Go-side `config.AdvancedToolFilteringConfig`. The Go side
    treats `retrieval_strategy: hybrid_history` as opt-in; without this model
    the nested subtree was silently dropped by Pydantic and the Go router
    received an effectively empty advanced_filtering block.
    """

    enabled: Optional[bool] = None
    retrieval_strategy: Optional[Literal["weighted", "hybrid_history"]] = None
    candidate_pool_size: Optional[int] = Field(default=None, ge=0)
    min_lexical_overlap: Optional[int] = Field(default=None, ge=0)
    min_combined_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    weights: Optional[ToolFilteringWeights] = None
    use_category_filter: Optional[bool] = None
    category_confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    allow_tools: Optional[List[str]] = None
    block_tools: Optional[List[str]] = None
    hybrid_history: Optional[HybridHistoryConfig] = None


class ToolSelectionPluginConfig(BaseModel):
    """Configuration for tool_selection plugin (semantic add/filter on request tools)."""

    enabled: bool
    mode: Optional[Literal["add", "filter"]] = None
    tools_db_path: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=0)
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    strategy: Optional[str] = None
    fallback_to_empty: Optional[bool] = None
    relevance_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    preserve_count: Optional[int] = Field(default=None, ge=0)
    advanced_filtering: Optional[AdvancedToolFilteringConfig] = None


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
        default=10000,
        gt=0,
        description="Maximum records in memory (must be > 0, default: 10000)",
    )
    capture_request_body: bool = True  # Capture request payloads
    capture_response_body: bool = True  # Capture response payloads
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
    injection_mode: Optional[Literal["tool_role", "system_prompt"]] = Field(
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
    on_failure: Optional[Literal["skip", "block", "warn"]] = Field(
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

    @model_validator(mode="before")
    @classmethod
    def normalize_response_cache_aliases(cls, value):
        if isinstance(value, dict) and value.get("type") in {
            "semantic-cache",
            "semantic_cache",
            "response-cache",
        }:
            warnings.warn(
                f"plugin type {value.get('type')!r} is deprecated; use 'response_cache'",
                DeprecationWarning,
                stacklevel=2,
            )
            value = dict(value)
            value["type"] = PluginType.RESPONSE_CACHE.value
        return value

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


class DecisionLearningAdaptationConfig(BaseModel):
    """Decision-local control for Router Learning adaptation."""

    model_config = ConfigDict(extra="forbid")

    mode: Optional[Literal["apply", "observe", "bypass"]] = None
    candidate_set: Optional[Literal["decision", "tier", "global"]] = None


class DecisionLearningProtectionConfig(BaseModel):
    """Decision-local control for Router Learning protection."""

    model_config = ConfigDict(extra="forbid")

    mode: Optional[Literal["apply", "observe", "bypass"]] = None
    stability_weight: Optional[float] = Field(default=None, ge=0.0)
    switch_margin: Optional[float] = Field(default=None, ge=0.0)


class DecisionAdaptationsConfig(BaseModel):
    """Decision-local Router Learning controls."""

    model_config = ConfigDict(extra="forbid")

    mode: Optional[Literal["apply", "observe", "bypass"]] = None
    adaptation: Optional[DecisionLearningAdaptationConfig] = None
    protection: Optional[DecisionLearningProtectionConfig] = None

    @model_validator(mode="after")
    def validate_mode_boundaries(self):
        component_modes = [
            ("adaptation", self.adaptation.mode if self.adaptation else None),
            ("protection", self.protection.mode if self.protection else None),
        ]
        for component, component_mode in component_modes:
            if component_mode is None:
                continue
            if self.mode == "bypass" and component_mode != "bypass":
                raise ValueError(
                    f"{component}.mode cannot be {component_mode!r} when mode is 'bypass'"
                )
            if self.mode == "observe" and component_mode == "apply":
                raise ValueError(
                    f"{component}.mode cannot be 'apply' when mode is 'observe'"
                )
        return self


class RouterLearningAdaptationConfig(BaseModel):
    """Global Router Learning adaptation controls."""

    model_config = ConfigDict(extra="forbid")

    enabled: Optional[StrictBool] = None
    strategy: Optional[Literal["routing_sampling"]] = None
    candidate_set: Optional[Literal["decision", "tier", "global"]] = None


class RouterLearningIdentityHeadersConfig(BaseModel):
    """Header names used by Router Learning protection identity."""

    model_config = ConfigDict(extra="forbid")

    session: Optional[StrictStr] = None
    conversation: Optional[StrictStr] = None


class RouterLearningIdentityConfig(BaseModel):
    """Identity configuration used by Router Learning protection."""

    model_config = ConfigDict(extra="forbid")

    headers: Optional[RouterLearningIdentityHeadersConfig] = None


class RouterLearningProtectionTuningConfig(BaseModel):
    """Global Router Learning protection tuning."""

    model_config = ConfigDict(extra="forbid")

    idle_timeout_seconds: Optional[int] = Field(default=None, ge=0)
    min_turns_before_switch: Optional[int] = Field(default=None, ge=0)
    switch_margin: Optional[float] = Field(default=None, ge=0.0)
    stability_weight: Optional[float] = Field(default=None, ge=0.0)


class RouterLearningProtectionConfig(BaseModel):
    """Global Router Learning protection controls."""

    model_config = ConfigDict(extra="forbid")

    enabled: Optional[StrictBool] = None
    scope: Optional[Literal["conversation", "session"]] = None
    identity: Optional[RouterLearningIdentityConfig] = None
    tuning: Optional[RouterLearningProtectionTuningConfig] = None


class RouterLearningRedisStateStoreConfig(BaseModel):
    """Redis connectivity for shared Router Learning protection state."""

    model_config = ConfigDict(extra="forbid")

    address: str
    password: Optional[str] = None
    database: int = Field(default=0, ge=0)
    key_prefix: Optional[str] = None


class RouterLearningStateStoreConfig(BaseModel):
    """Optional shared Router Learning protection state."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["local", "redis"] = "local"
    ttl_seconds: int = Field(default=86400, ge=0)
    timeout_ms: int = Field(default=50, ge=0)
    redis: Optional[RouterLearningRedisStateStoreConfig] = None

    @model_validator(mode="after")
    def validate_redis(self):
        if self.backend == "redis" and self.redis is None:
            raise ValueError(
                "redis config is required when state_store.backend is redis"
            )
        return self


class RouterLearningConfig(BaseModel):
    """Global Router Learning controls."""

    model_config = ConfigDict(extra="forbid")

    enabled: Optional[StrictBool] = None
    adaptation: Optional[RouterLearningAdaptationConfig] = None
    protection: Optional[RouterLearningProtectionConfig] = None
    state_store: Optional[RouterLearningStateStoreConfig] = None


class OutputContractChoiceSetSpec(BaseModel):
    """Allowed values for choice-style output contracts."""

    model_config = ConfigDict(extra="forbid")

    values: List[str]


class OutputContractJSONSchemaSpec(BaseModel):
    """Structured JSON schema selector for router-enforced output contracts."""

    model_config = ConfigDict(extra="forbid")

    schema_ref: Literal["terminal_action_v1"]


class OutputContractReferenceSpec(BaseModel):
    """Reference selection source metadata."""

    model_config = ConfigDict(extra="forbid")

    source: Optional[Literal["candidate_responses"]] = None
    id_format: Optional[Literal["index", "reference_number"]] = None


class OutputContractRenderSpec(BaseModel):
    """How the router renders a normalized output value."""

    model_config = ConfigDict(extra="forbid")

    mode: Optional[Literal["value", "template"]] = None
    template: Optional[str] = None


class OutputContractExtractSpec(BaseModel):
    """Preferred response fields for extraction."""

    model_config = ConfigDict(extra="forbid")

    mode: Optional[Literal["exact", "json_object"]] = None
    sources: Optional[
        List[Literal["content", "reasoning_content", "candidate_responses"]]
    ] = None


class OutputContractNormalizeSpec(BaseModel):
    """Normalization policy for structured output contracts."""

    model_config = ConfigDict(extra="forbid")

    defaults: Optional[Dict[str, str]] = None


class OutputContractPostprocess(BaseModel):
    """Post-processing operation for output contract enforcement."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["dereference_selected_reference"]


class OutputContractSpec(BaseModel):
    """Router-executable typed output contract."""

    model_config = ConfigDict(extra="forbid")

    type: Optional[Literal["choice", "structured_json", "reference_selection"]] = None
    choice_set: Optional[OutputContractChoiceSetSpec] = None
    json_schema: Optional[OutputContractJSONSchemaSpec] = None
    reference: Optional[OutputContractReferenceSpec] = None
    render: Optional[OutputContractRenderSpec] = None
    extract: Optional[OutputContractExtractSpec] = None
    normalize: Optional[OutputContractNormalizeSpec] = None
    postprocess: Optional[List[OutputContractPostprocess]] = None


def _conditions_reference_signal(conditions, signal_type):
    for condition in conditions or []:
        if (condition.type or "").lower() == signal_type:
            return True
        if _conditions_reference_signal(condition.conditions, signal_type):
            return True
    return False


class DecisionAction(BaseModel):
    """Explicit action a matched decision applies instead of candidate ranking."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["route"]
    destination: str

    @model_validator(mode="after")
    def validate_destination(self):
        if not self.destination.strip():
            raise ValueError("action.destination is required")
        return self


class Decision(BaseModel):
    """Routing decision configuration."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: Optional[str] = None
    priority: int
    tier: int = Field(default=0, ge=0)
    # A decision without an explicit rule is the canonical match-all fallback.
    # This mirrors the Go runtime and the DSL `ROUTE` form without `WHEN`.
    rules: Rules = Field(default_factory=Rules)
    action: Optional[DecisionAction] = None
    output_contract: Optional[str] = None
    output_contract_spec: Optional[OutputContractSpec] = None
    modelRefs: List[ModelRef] = Field(alias="modelRefs")
    algorithm: Optional[AlgorithmConfig] = None  # Multi-model orchestration algorithm
    adaptations: Optional[DecisionAdaptationsConfig] = None
    plugins: Optional[List[PluginConfig]] = []
    annotations: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_action(self):
        if self.action is None:
            return self
        if not _conditions_reference_signal(self.rules.conditions, "jailbreak"):
            raise ValueError(
                "a route action requires an explicit jailbreak condition in rules"
            )
        return self

    @model_validator(mode="after")
    def validate_prompt_candidates(self):
        if self.algorithm and self.algorithm.minimum_candidates and self.modelRefs:
            effective_names = {
                (model_ref.model.strip(), (model_ref.lora_name or "").strip())
                for model_ref in self.modelRefs
                if model_ref.model.strip()
            }
            if len(effective_names) < self.algorithm.minimum_candidates:
                raise ValueError(
                    "algorithm.minimum_candidates="
                    f"{self.algorithm.minimum_candidates} requires at least that many "
                    f"unique modelRefs, got {len(effective_names)}"
                )
        if (
            self.algorithm
            and self.algorithm.type == "prompt"
            and len(self.modelRefs) < PROMPT_MIN_CANDIDATES
        ):
            raise ValueError("algorithm.type=prompt requires at least two modelRefs")
        if self.algorithm and self.algorithm.type == "prompt":
            model_names = [model_ref.model for model_ref in self.modelRefs]
            if len(model_names) != len(set(model_names)):
                raise ValueError("algorithm.type=prompt requires unique modelRefs")
            effective_names = [
                model_ref.lora_name or model_ref.model for model_ref in self.modelRefs
            ]
            if len(effective_names) != len(set(effective_names)):
                raise ValueError(
                    "algorithm.type=prompt requires unique effective model identities"
                )
        return self

    @model_validator(mode="after")
    def validate_annotation_bounds(self):
        if self.annotations is None:
            return self
        if len(self.annotations) > MAX_DECISION_ANNOTATIONS:
            raise ValueError(
                f"annotations cannot contain more than {MAX_DECISION_ANNOTATIONS} entries"
            )
        encoded = json.dumps(
            self.annotations,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > MAX_DECISION_ANNOTATION_BYTES:
            raise ValueError(
                f"annotations cannot exceed {MAX_DECISION_ANNOTATION_BYTES} encoded bytes"
            )
        return self


class ModelPricing(BaseModel):
    """Model pricing configuration."""

    model_config = ConfigDict(extra="forbid")

    currency: Optional[str] = Field(default="USD", pattern=r"^[A-Z]{3}$")
    prompt_per_1m: Optional[float] = Field(default=0.0, ge=0, allow_inf_nan=False)
    cached_input_per_1m: Optional[float] = Field(default=0.0, ge=0, allow_inf_nan=False)
    cache_write_per_1m: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    completion_per_1m: Optional[float] = Field(default=0.0, ge=0, allow_inf_nan=False)


class ProviderReliability(BaseModel):
    """Generated Envoy reliability policy for one provider model."""

    model_config = ConfigDict(extra="forbid")

    lb_policy: Literal["round_robin", "least_request"] = "round_robin"
    retry_count: int = Field(default=0, ge=0, le=5)
    retry_on: str = "connect-failure,refused-stream"
    consecutive_5xx: int = Field(default=0, ge=0)
    base_ejection_time: str = "30s"
    max_ejection_percent: int = Field(default=50, ge=0, le=100)
    health_check_path: Optional[str] = None
    health_check_interval: str = "10s"
    health_check_timeout: str = "2s"


class Model(BaseModel):
    """Provider model binding for canonical providers.models entries."""

    name: str
    reasoning_family: Optional[str] = None
    provider_model_id: Optional[str] = None
    backend_refs: List["BackendRef"] = Field(default_factory=list)
    pricing: Optional[ModelPricing] = None
    reliability: Optional[ProviderReliability] = None
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

    model_config = ConfigDict(populate_by_name=True)

    model_cards: List[RoutingModel] = Field(default_factory=list, alias="modelCards")
    signals: Signals = Field(default_factory=Signals)
    projections: Projections = Field(default_factory=Projections)
    decisions: List[Decision] = Field(default_factory=list)
    strategy: Optional[RoutingStrategy] = None


class Entrypoint(BaseModel):
    """Request-facing virtual model names mapped to one routing recipe."""

    model_config = ConfigDict(extra="forbid")

    model_names: List[str] = Field(min_length=1)
    recipe: str = Field(min_length=1)

    @field_validator("model_names", mode="before")
    @classmethod
    def normalize_model_names(cls, value):
        if not isinstance(value, list):
            return value
        if any(not isinstance(item, str) for item in value):
            raise ValueError("model_names entries must be strings")
        normalized = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        if not normalized:
            raise ValueError("model_names must contain at least one non-empty name")
        return normalized

    @field_validator("recipe")
    @classmethod
    def normalize_recipe(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("recipe must not be empty")
        return normalized


class RecipeRouting(BaseModel):
    """Recipe-owned routing profile; the shared model catalog stays top-level."""

    model_config = ConfigDict(extra="forbid")

    signals: Signals = Field(default_factory=Signals)
    projections: Projections = Field(default_factory=Projections)
    decisions: List[Decision] = Field(default_factory=list)
    strategy: Optional[RoutingStrategy] = None


class Recipe(BaseModel):
    """Named routing profile selected through an entrypoint."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: Optional[str] = None
    routing: RecipeRouting = Field(default_factory=RecipeRouting)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("name must not be empty")
        return normalized


class EmbeddingModelsConfig(BaseModel):
    """Embedding models configuration for memory and semantic features."""

    # Preserve advanced nested fields when users pass through custom config blocks.
    model_config = ConfigDict(extra="allow")

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
    embedding_config: Optional[EmbeddingClassifierConfig] = Field(
        default=None,
        description="Embedding classifier tuning (model_type/target_dimension/top_k/prototype_scoring/etc.)",
    )
    endpoint: Optional[EmbeddingEndpointConfig] = Field(
        default=None,
        description="External OpenAI-compatible embedding provider endpoint",
    )
    use_cpu: bool = Field(True, description="Use CPU for inference")


class UserConfig(BaseModel):
    """Canonical v0.3 user configuration."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    version: str
    listeners: List[Listener] = Field(default_factory=list)
    providers: Providers = Field(default_factory=Providers)
    routing: Routing = Field(default_factory=Routing)
    entrypoints: List[Entrypoint] = Field(default_factory=list)
    recipes: List[Recipe] = Field(default_factory=list)
    global_: Optional[Dict[str, Any]] = Field(default=None, alias="global")
    setup: Optional[Dict[str, Any]] = None

    @property
    def signals(self) -> Signals:
        return self.routing.signals

    @property
    def decisions(self) -> List[Decision]:
        return self.routing.decisions


# Resolve forward references for recursive condition trees.
Condition.model_rebuild()
Model.model_rebuild()
