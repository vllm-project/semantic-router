"""Pydantic models for vLLM Semantic Router configuration."""

import json
import math
import posixpath
import re
from datetime import date, datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional

from .models_systemone import SystemOneRule
from .models_native import CalibrationArtifact
from .models_predicates import NumericPredicate
from .models_application import (
    ComplexityRule,
    Domain,
    FactCheck,
    HallucinationRule,
    JailbreakRule,
    ModalityRule,
    PIIRule,
    RefusalRule,
    SafetyRule,
    TurnRelationRule,
    UserFeedback,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
    model_serializer,
    model_validator,
)

from .algorithms import AlgorithmConfig, ModelRef
from .config_contract import UnknownPolicy
from .config_schema import routing_surface_catalog, surface_types
from .context_bands import normalize_token_count, validate_context_band
from .durations import parse_duration

RoutingStrategy = Literal["priority", "confidence"]
MAX_DECISION_ANNOTATIONS = 32
MAX_DECISION_ANNOTATION_BYTES = 4096
# Condition types whose conditions may name one of the rule's options.
LABELED_CONDITION_TYPES = frozenset(
    surface["type"]
    for surface in routing_surface_catalog()["signals"]
    if surface.get("reference_qualifier") == "label"
)


class ListenerTLS(BaseModel):
    """A listener's server certificate for one-way TLS."""

    model_config = ConfigDict(extra="forbid")

    cert_file: str = Field(
        description="PEM certificate chain; a relative path is relative to "
        "the config file's directory."
    )
    key_file: str = Field(description="PEM private key of the certificate.")


class ListenerIdentity(BaseModel):
    """The identity sources a listener trusts."""

    model_config = ConfigDict(extra="forbid")

    trust_headers: bool = Field(
        default=False,
        description="Keep the client identity headers (x-authz-* and the "
        "global.services.authz.identity names) that an authenticating proxy "
        "in front of this listener sets. Off, the listener drops them: the "
        "Router in standalone mode, the generated Envoy listener with "
        "--gateway extproc.",
    )
    trusted_peers: Optional[List[str]] = Field(
        default=None,
        description="CIDRs whose connections may carry trusted identity "
        "headers; the TCP peer is checked, never X-Forwarded-For. Standalone "
        "mode checks it; --gateway extproc rejects it.",
    )


class ListenerSystemOne(BaseModel):
    """Explicit public models for native System One inference."""

    model_config = ConfigDict(extra="forbid")
    models: List[str] = Field(min_length=1)
    max_deadline_ms: Optional[int] = Field(
        default=None,
        gt=0,
        description="The longest this listener waits for a model it calls "
        "directly, however long a request's options.deadline_ms asks for. "
        "Omitted, 600000 (10 minutes).",
    )

    @field_validator("models")
    @classmethod
    def exact_model_names(cls, names):
        if any(not name or name != name.strip() for name in names):
            raise ValueError("systemone.models requires exact non-empty public names")
        if len(set(names)) != len(names):
            raise ValueError("systemone.models must not contain duplicate names")
        return names


class Listener(BaseModel):
    """Network listener configuration."""

    name: str
    address: str = "0.0.0.0"
    port: int
    timeout: Optional[str] = "300s"
    api_keys: Optional[List[str]] = Field(
        default=None,
        description="Client keys required to call this listener. "
        "If set, requests must send one of these values as "
        "'Authorization: Bearer <key>' or, for Azure OpenAI clients, "
        "'api-key: <key>'; other requests are rejected with HTTP 401.",
    )
    models: Optional[List[str]] = Field(
        default=None,
        description="The only request models this listener accepts, by exact "
        "'model' value; other models are rejected with HTTP 403 "
        "model_not_allowed and /v1/models lists only these. Empty accepts "
        "every model. Standalone mode enforces it; --gateway extproc rejects it.",
    )
    systemone: Optional[ListenerSystemOne] = Field(
        default=None,
        description="Publish native System One inference for these exact public "
        "model IDs. Omitted keeps the API private; Chat model access is unchanged.",
    )
    tls: Optional[ListenerTLS] = Field(
        default=None,
        description="Serve this listener over TLS. Standalone mode serves it; "
        "--gateway extproc does not.",
    )
    identity: Optional[ListenerIdentity] = Field(
        default=None,
        description="The identity sources this listener trusts; by default none.",
    )


class KeywordSignal(BaseModel):
    """Keyword-based signal configuration."""

    name: str
    description: Optional[str] = None
    operator: str
    keywords: List[str]
    case_sensitive: bool = False
    # Also match prior user messages; False keeps current-turn-only matching.
    include_history: bool = False
    # At most one: an OR rule matches on how often its keywords occur, as a
    # count or as occurrences per 1,000 characters.
    count: Optional[NumericPredicate] = None
    density: Optional[NumericPredicate] = None

    @model_validator(mode="after")
    def validate_measure(self):
        if self.count is not None and self.density is not None:
            raise ValueError("set count or density, not both")
        measure = "count" if self.count is not None else "density"
        if (self.count or self.density) and self.operator != "OR":
            raise ValueError(
                f"{measure} counts the keywords an OR rule finds; "
                f"operator {self.operator!r} matches on which keywords occur"
            )
        return self


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
      - "score": contributes the score of the projection score it names.
    """

    type: str
    name: Optional[str] = None
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


class Language(BaseModel):
    """Language detection signal configuration."""

    name: str
    description: str


class ContextRule(BaseModel):
    """Context-based (token count) signal configuration.

    A rule is an inclusive band: min_tokens <= token_count <= max_tokens.
    Both limits accept "1K" and "1.5M" suffixes. At least one limit must be
    set. Omitting min_tokens means 0; omitting max_tokens makes the band
    open-ended so every count at or above min_tokens matches. Equal limits
    are an exact-match band.
    """

    name: str
    min_tokens: Optional[str] = None
    max_tokens: Optional[str] = None
    description: Optional[str] = None

    @field_validator("min_tokens", "max_tokens", mode="before")
    @classmethod
    def coerce_token_count(cls, value, info):
        return normalize_token_count(value, info.field_name)

    @model_validator(mode="after")
    def validate_band(self):
        validate_context_band(self.min_tokens, self.max_tokens)
        return self


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


class Signals(BaseModel):
    """All signal configurations."""

    keywords: Optional[List[KeywordSignal]] = []
    domains: Optional[List[Domain]] = []
    fact_check: Optional[List[FactCheck]] = []
    user_feedbacks: Optional[List[UserFeedback]] = []
    language: Optional[List[Language]] = []
    context: Optional[List[ContextRule]] = []
    complexity: Optional[List[ComplexityRule]] = []
    modality: Optional[List[ModalityRule]] = []
    role_bindings: Optional[List[RoleBindingRule]] = []
    jailbreak: Optional[List[JailbreakRule]] = []
    safety: list[SafetyRule] = Field(default_factory=list)
    hallucination: Optional[List[HallucinationRule]] = []
    refusal: Optional[List[RefusalRule]] = []
    turn_relation: Optional[List[TurnRelationRule]] = []
    pii: Optional[List[PIIRule]] = []
    conversation: Optional[List[ConversationRule]] = []
    metadata: Optional[List[MetadataRule]] = []
    input_modality: Optional[List[InputModalityRule]] = []
    systemone: Optional[List[SystemOneRule]] = []

    @model_validator(mode="after")
    def validate_rule_names(self):
        for family in type(self).model_fields:
            seen: set[str] = set()
            for signal in getattr(self, family) or []:
                name = (
                    signal.name.lower()
                    if family in {"metadata", "input_modality"}
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

    model_config = ConfigDict(extra="forbid")

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
        if self.label is not None and self.type not in LABELED_CONDITION_TYPES:
            raise ValueError(
                "label is only valid for systemone and application signal conditions"
            )
        if self.on_error is not None and self.type != "systemone":
            raise ValueError("on_error is only valid for systemone conditions")
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

    model_config = ConfigDict(extra="forbid")

    operator: str = "AND"
    conditions: List[Condition] = Field(default_factory=list)
    on_unknown: Optional[UnknownPolicy] = None
    on_error: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalise_leaf_or_empty(cls, data):
        """Wrap a bare leaf node into AND([leaf]) and fill missing fields."""
        if not isinstance(data, dict):
            return data
        # Leaf node: has type/name but no operator → wrap in AND
        if "type" in data and "operator" not in data:
            leaf = {key: value for key, value in data.items() if key != "on_unknown"}
            leaf.setdefault("name", "")
            rules = {"operator": "AND", "conditions": [leaf]}
            if "on_unknown" in data:
                rules["on_unknown"] = data["on_unknown"]
            return rules
        return data

    @model_validator(mode="after")
    def validate_on_unknown_conflict(self):
        if self.on_unknown is not None and (
            self.on_error or _any_condition_sets_on_error(self.conditions)
        ):
            raise ValueError(
                "condition on_error has no effect when rules.on_unknown is set; "
                "remove one of them"
            )
        if self.on_error:
            raise ValueError("on_error only applies to leaf conditions")
        return self


def _any_condition_sets_on_error(conditions: Optional[List["Condition"]]) -> bool:
    for condition in conditions or []:
        if condition.on_error is not None:
            return True
        if _any_condition_sets_on_error(condition.conditions):
            return True
    return False


PluginType = Enum(
    "PluginType",
    {plugin_type.upper(): plugin_type for plugin_type in surface_types("plugins")},
    type=str,
)


class ResponseCacheSemanticConfig(BaseModel):
    """The semantic tier: its match block finds the cached requests a request
    may reuse.

    Mirrors the Go-side `config.ResponseCacheSemanticConfig`; the Go side is
    authoritative for the match block.
    """

    model_config = ConfigDict(extra="forbid")

    match: Optional[Dict[str, Any]] = None


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

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    mode: Literal["semantic", "exact", "exact_then_semantic"] = "semantic"
    scope: Literal["user", "team", "tenant", "global"] = "user"
    ttl_seconds: Optional[int] = Field(
        default=None, ge=0, description="TTL in seconds (must be >= 0, default: None)"
    )
    semantic: Optional[ResponseCacheSemanticConfig] = None
    request_controls: Optional[ResponseCacheRequestControlsConfig] = None
    personalized: Optional[ResponseCachePersonalizedConfig] = None
    revision: Optional[ResponseCacheRevisionConfig] = None


CompressionTokenLimit = Literal["auto"] | int


class ContextBudgetConfig(BaseModel):
    """The request the context plugin's compress block produces: max_tokens
    starts compression and is its target; auto is the model's window less
    the reserved output."""

    model_config = ConfigDict(extra="forbid")

    max_tokens: Optional[CompressionTokenLimit] = None
    reserve_output_tokens: Optional[CompressionTokenLimit] = None

    @model_validator(mode="after")
    def validate_budget(self):
        if isinstance(self.max_tokens, int) and self.max_tokens < 1:
            raise ValueError("budget.max_tokens must be at least 1, or auto")
        if (
            isinstance(self.reserve_output_tokens, int)
            and self.reserve_output_tokens < 0
        ):
            raise ValueError("budget.reserve_output_tokens cannot be negative")
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


class ContextCompressionCurrentUserConfig(BaseModel):
    mode: Literal["preserve", "truncate"] = "preserve"


class ContextCompressionTargetsConfig(BaseModel):
    current_user: ContextCompressionCurrentUserConfig = Field(
        default_factory=ContextCompressionCurrentUserConfig
    )
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


class ContextCompressConfig(BaseModel):
    """The context plugin's compress block: the request's own segments,
    judged by its match block (relevance by default), shrink within the
    budget."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["auto", "always"] = "auto"
    targets: Optional[ContextCompressionTargetsConfig] = None
    match: Optional[Dict[str, Any]] = None
    recovery: Optional[ContextCompressionRecoveryConfig] = None
    request_controls: Optional[ContextCompressionRequestControlsConfig] = None
    failure_mode: Literal["fail_open", "fail_closed"] = "fail_open"


class ContextMemoryStoreConfig(BaseModel):
    """Whether a memory source writes the turn after the response; unset
    follows global.stores.memory.auto_store."""

    model_config = ConfigDict(extra="forbid")

    auto: Optional[bool] = None


class ContextSource(BaseModel):
    """A source of retrieved context: a RAG backend or the user's memory,
    each kept by its match block."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["rag", "memory"]
    match: Optional[Dict[str, Any]] = None
    backend: Optional[
        Literal[
            "vectorstore", "milvus", "qdrant", "external_api", "mcp", "openai", "hybrid"
        ]
    ] = None
    backend_config: Optional[Dict[str, Any]] = None
    max_context_length: Optional[int] = Field(default=None, ge=0)
    on_failure: Optional[Literal["skip", "block", "warn"]] = None
    cache_results: Optional[bool] = None
    cache_ttl_seconds: Optional[int] = Field(default=None, ge=1)
    store: Optional[ContextMemoryStoreConfig] = None
    hybrid_search: Optional[bool] = None
    hybrid_mode: Optional[Literal["weighted", "rrf"]] = None
    reflection: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_type_fields(self):
        rag_fields = (
            "backend",
            "backend_config",
            "max_context_length",
            "on_failure",
            "cache_results",
            "cache_ttl_seconds",
        )
        memory_fields = ("store", "hybrid_search", "hybrid_mode", "reflection")
        other = memory_fields if self.type == "rag" else rag_fields
        for field in other:
            if getattr(self, field) is not None:
                raise ValueError(f"a {self.type} source takes no {field}")
        if self.type == "rag" and not self.backend:
            raise ValueError("a rag source needs a backend")
        return self


class ContextInjectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["tool_role", "system_prompt"] = "tool_role"


class ContextPluginConfig(BaseModel):
    """Configuration for the context plugin: sources (RAG, memory) kept by
    their match blocks, compress within the budget, and injection."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    sources: Optional[List[ContextSource]] = None
    compress: Optional[ContextCompressConfig] = None
    budget: Optional[ContextBudgetConfig] = None
    injection: Optional[ContextInjectionConfig] = None

    @model_validator(mode="after")
    def validate_context(self):
        if not self.enabled:
            return self
        if not self.sources and self.compress is None:
            raise ValueError(
                "an enabled context plugin retrieves sources, compresses the request or both"
            )
        types = [source.type for source in self.sources or []]
        for source_type in set(types):
            if types.count(source_type) > 1:
                raise ValueError(
                    f"a context plugin has at most one {source_type} source"
                )
        if self.budget is not None and self.compress is None:
            raise ValueError(
                "budget bounds the request compress produces; add compress"
            )
        if self.injection is not None and not self.sources:
            raise ValueError("injection places what the sources retrieve; add a source")
        return self


class PromptCachePluginConfig(BaseModel):
    """Route-local prompt-cache marker injection."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    ttl: Literal["5m", "1h"] = "5m"
    targets: List[Literal["instructions", "tools"]] = Field(
        default_factory=lambda: ["instructions", "tools"]
    )
    on_unsupported: Literal["skip", "reject"] = "skip"

    @model_validator(mode="after")
    def validate_targets(self) -> "PromptCachePluginConfig":
        if not self.targets:
            raise ValueError("targets must not be empty")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must not contain duplicates")
        return self


class FastResponsePluginConfig(BaseModel):
    """Configuration for fast_response plugin."""

    message: str


class RequestParamsPluginConfig(BaseModel):
    """Configuration for request_params plugin."""

    blocked_params: Optional[List[str]] = None
    default_max_tokens: Optional[
        Annotated[int, Field(ge=1, strict=True)] | Literal["auto"]
    ] = None
    max_tokens_limit: Optional[int] = Field(default=None, ge=1)
    max_n: Optional[int] = Field(default=None, ge=1)
    strip_unknown: Optional[bool] = None


ResponseGuardAction = Literal["block", "annotate", "none"]


class ResponseGuardRule(BaseModel):
    """One response_guard rule: what it does when its condition tree holds."""

    model_config = ConfigDict(extra="forbid")

    when: Condition
    action: ResponseGuardAction
    details: Optional[bool] = None

    @model_validator(mode="after")
    def details_annotate(self):
        if self.details and self.action != "annotate":
            raise ValueError("details applies only to action annotate")
        return self


class ResponseGuardPluginConfig(BaseModel):
    """Configuration for the response_guard plugin.

    Each rule reads named signals over the answer (and the request); a rule
    whose response-stage signal could not be read takes on_unknown.
    """

    model_config = ConfigDict(extra="forbid")

    rules: List[ResponseGuardRule] = Field(min_length=1)
    on_unknown: Optional[ResponseGuardAction] = None


class ToolsPolicyConfig(BaseModel):
    """The tools plugin's deterministic policy.

    Mirrors the Go-side `config.ToolsPolicyConfig`; the Go side is
    authoritative for trusted_facts.
    """

    mode: Literal["passthrough", "filtered", "none"] = "passthrough"
    allow_tools: Optional[List[str]] = None
    block_tools: Optional[List[str]] = None
    strip_tool_history: Optional[bool] = None
    trusted_facts: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_mode_contract(self):
        has_filters = bool(self.allow_tools or self.block_tools)
        if self.mode == "filtered" and not has_filters:
            raise ValueError("mode=filtered requires allow_tools or block_tools")
        if self.mode != "filtered" and has_filters:
            raise ValueError("allow_tools and block_tools require mode=filtered")
        if self.strip_tool_history and self.mode != "none":
            raise ValueError("strip_tool_history requires mode=none")
        return self


class StickyToolSelectionConfig(BaseModel):
    """Session-scoped sticky tool-set selection (issue #3347).

    Mirrors the Go-side `config.StickyToolSelectionConfig`. Opt-in and
    disabled by default. This layer only mirrors the schema surface so
    Pydantic stops dropping the subtree; the Go side is authoritative for
    bounds (max_tools 1..128, max_new_tools_per_turn 0..max_tools) and for
    refusing sticky.enabled in this release.
    """

    enabled: bool = False
    max_tools: Optional[int] = Field(default=None, ge=1, le=128)
    max_new_tools_per_turn: Optional[int] = Field(default=None, ge=0)
    pin_called_tools: Optional[bool] = None


class ToolsSelectInput(BaseModel):
    """What the decision model reads to choose the tools."""

    model_config = ConfigDict(extra="forbid")

    history: Optional[bool] = None


class ToolsSelectConfig(BaseModel):
    """The tools the decision model chooses, among the request's own tools
    (filter) or a catalog's (add).

    Mirrors the Go-side `config.ToolsSelectConfig`; the Go side is
    authoritative for the match block and the catalog.
    """

    mode: Literal["filter", "add"]
    tools_db_path: Optional[str] = None
    match: Optional[Dict[str, Any]] = None
    need_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    preserve_count: Optional[int] = Field(default=None, ge=0, le=255)
    on_error: Optional[Literal["keep", "strip", "skip", "fail"]] = None
    on_overflow: Optional[Literal["keep", "reject"]] = None
    input: Optional[ToolsSelectInput] = None
    timeout_ms: Optional[int] = Field(default=None, ge=0, le=60000)
    sticky: Optional[StickyToolSelectionConfig] = None

    @model_validator(mode="after")
    def validate_mode_fields(self):
        if self.mode == "add":
            if not self.tools_db_path:
                raise ValueError("mode=add requires tools_db_path")
            if self.preserve_count is not None or self.on_overflow is not None:
                raise ValueError("preserve_count and on_overflow apply to mode=filter")
            if self.on_error not in (None, "skip", "fail"):
                raise ValueError("mode=add takes on_error skip or fail")
        else:
            if self.tools_db_path:
                raise ValueError("tools_db_path applies to mode=add")
            if self.on_error not in (None, "keep", "strip", "fail"):
                raise ValueError("mode=filter takes on_error keep, strip or fail")
        return self


class ToolsPluginConfig(BaseModel):
    """Configuration for the tools plugin."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    policy: Optional[ToolsPolicyConfig] = None
    select: Optional[ToolsSelectConfig] = None

    @model_validator(mode="after")
    def validate_select_has_tools(self):
        if (
            self.select is not None
            and self.policy is not None
            and self.policy.mode == "none"
        ):
            raise ValueError(
                "select requires a policy that keeps tools; mode=none removes every tool"
            )
        return self


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


class RouterReplayPluginConfig(BaseModel):
    """Decision overrides for global.services.router_replay capture defaults.

    Omitted fields inherit the shared service policy. Explicit false and zero
    remain present so a decision can disable capture or remove a trace limit.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: Optional[StrictBool] = None
    capture_request_body: Optional[StrictBool] = None
    capture_response_body: Optional[StrictBool] = None
    capture_personal_data: Optional[StrictBool] = None
    capture_tool_results: Optional[StrictBool] = None
    max_records: Optional[int] = Field(default=None, ge=0)
    max_body_bytes: Optional[int] = Field(default=None, ge=0)
    max_tool_trace_bytes: Optional[int] = Field(default=None, ge=0)
    max_tool_trace_steps: Optional[int] = Field(default=None, ge=0)
    shadow: Optional["RouterReplayShadowConfig"] = None


# Headers that carry a credential on the primary path. A shadow copy never
# carries them and router_replay shadow.forward_headers may not list them.
# Keep in step with the Router's shadowCredentialHeaders in pkg/config.
SHADOW_CREDENTIAL_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "x-user-openai-key",
        "x-user-anthropic-key",
        "x-user-azure-openai-key",
        "x-user-bedrock-key",
        "x-user-gemini-key",
        "x-user-vertex-ai-key",
        "x-user-minimax-key",
    }
)


class RouterReplayShadowConfig(BaseModel):
    """router_replay's shadow block.

    Sends a bounded, sampled copy of the approved request to a secondary
    configured model after the primary dispatch is finalized, and records its
    outcome in the replay record. The primary response never waits on or
    changes because of the shadow call.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="Configured logical model receiving the shadow copy")
    sample_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_concurrency: int = Field(default=2, ge=0)
    max_queue_depth: int = Field(default=8, ge=0)
    timeout_seconds: int = Field(default=30, ge=0)
    max_response_bytes: int = Field(default=1048576, ge=0)
    max_retries: int = Field(default=0, ge=0, le=3)
    capture_response_body: bool = False
    max_capture_bytes: int = Field(default=4096, ge=0)
    tls_skip_verify: bool = False
    forward_headers: list[str] = Field(
        default_factory=list,
        description=(
            "Decision header_mutation names the shadow copy may carry; nothing "
            "else a decision sets for the primary backend is forwarded"
        ),
    )

    @field_validator("forward_headers")
    @classmethod
    def reject_credential_headers(cls, names: list[str]) -> list[str]:
        # Mirrors the Router's validateShadowForwardHeaders so both admission
        # paths refuse the same names.
        for name in names:
            stripped = name.strip()
            if not stripped:
                raise ValueError("shadow forward_headers entries cannot be empty")
            if stripped.startswith(":"):
                raise ValueError(
                    f"shadow forward_headers cannot include pseudo-header {stripped!r}"
                )
            if stripped.lower() in SHADOW_CREDENTIAL_HEADERS:
                raise ValueError(
                    "shadow forward_headers cannot include credential header "
                    f"{stripped!r}"
                )
        return names

    @field_validator("model")
    @classmethod
    def require_model(cls, model: str) -> str:
        if not model.strip():
            raise ValueError("shadow model is required")
        return model


RouterReplayPluginConfig.model_rebuild()


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
    agent: Optional[StrictStr] = None


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


class RouterLearningProtectionReleaseConfig(BaseModel):
    """Client-requested release of a protection hold."""

    model_config = ConfigDict(extra="forbid")

    enabled: Optional[StrictBool] = None
    header: Optional[StrictStr] = Field(default=None, min_length=1)
    min_user_turns_between: Optional[int] = Field(default=None, ge=0)


class RouterLearningProtectionConfig(BaseModel):
    """Global Router Learning protection controls."""

    model_config = ConfigDict(extra="forbid")

    enabled: Optional[StrictBool] = None
    scope: Optional[Literal["conversation", "session"]] = None
    identity: Optional[RouterLearningIdentityConfig] = None
    tuning: Optional[RouterLearningProtectionTuningConfig] = None
    release: Optional[RouterLearningProtectionReleaseConfig] = None


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


class DecisionRetention(BaseModel):
    """What the Router keeps of a decision's answers.

    Mirrors the Go-side `config.DecisionRetention`: drop suppresses response
    cache, memory and Responses-object writes; ttl_turns bounds cache entries;
    keep_current_model and prefer_prefix_retention are retention hints.
    """

    model_config = ConfigDict(extra="forbid")

    drop: Optional[StrictBool] = None
    ttl_turns: Optional[int] = Field(default=None, ge=0)
    keep_current_model: Optional[StrictBool] = None
    prefer_prefix_retention: Optional[StrictBool] = None

    @model_validator(mode="after")
    def reject_drop_with_ttl(self):
        if self.drop and self.ttl_turns:
            raise ValueError(
                f"retention: drop=true conflicts with ttl_turns={self.ttl_turns}"
            )
        return self


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


# Decision reliability fields Envoy cannot apply to one request.
DECISION_RELIABILITY_NATIVE_ONLY = (
    "idle_timeout",
    "first_byte_timeout",
    "retry_back_off_base",
    "retry_back_off_max",
    "retry_after_max",
)


class DecisionReliability(BaseModel):
    """A decision's override of its provider model's timeouts and retries.

    The fields mean what they mean in the provider reliability block. Timeouts
    and retry_count replace the provider model's, and 0s disables a timeout;
    retry_on and retriable_status_codes add to the provider model's, as Envoy's
    per-request headers do. Unset fields keep the provider model's.
    """

    model_config = ConfigDict(extra="forbid")

    total_timeout: Optional[str] = None
    per_try_timeout: Optional[str] = None
    idle_timeout: Optional[str] = None
    first_byte_timeout: Optional[str] = None
    retry_count: Optional[int] = Field(default=None, ge=0, le=5)
    retry_on: Optional[str] = None
    retriable_status_codes: Optional[List[Annotated[int, Field(ge=100, le=599)]]] = None
    retry_back_off_base: Optional[str] = None
    retry_back_off_max: Optional[str] = None
    retry_after_max: Optional[str] = None

    @field_validator(
        "total_timeout", "per_try_timeout", "idle_timeout", "first_byte_timeout"
    )
    @classmethod
    def _duration_or_zero(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            parse_duration(value)
        return value

    @field_validator("retry_back_off_base", "retry_back_off_max", "retry_after_max")
    @classmethod
    def _positive_duration(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and parse_duration(value) <= 0:
            raise ValueError("must be a positive duration such as 30s")
        return value

    @model_validator(mode="after")
    def _back_off_bounds(self) -> "DecisionReliability":
        if self.retry_back_off_max is None:
            return self
        base = parse_duration(self.retry_back_off_base or "25ms")
        if parse_duration(self.retry_back_off_max) < base:
            raise ValueError("retry_back_off_max must not be below retry_back_off_base")
        return self

    def native_only_fields(self) -> List[str]:
        """The fields set that only standalone mode honors."""
        return [
            field
            for field in DECISION_RELIABILITY_NATIVE_ONLY
            if getattr(self, field) is not None
        ]


def _fallback_duration_seconds(value: str | int):
    """A fallback duration as the Router reads it: a Go duration string such
    as 30s, or integer nanoseconds."""
    if isinstance(value, bool):
        raise ValueError("must be a duration such as 30s")
    if isinstance(value, int):
        return value / 1_000_000_000
    return parse_duration(value)


class FallbackOverride(BaseModel):
    """A decision's override of its recipe's cross-model fallback policy.

    Each field it sets replaces the recipe's (routing.fallback, then
    global.router.fallback); unset fields keep it. The candidates stay the
    decision's ranked models. circuit_breaker is per backend, so it stays on
    the recipe's or the global policy.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: Optional[StrictBool] = None
    max_attempts: Optional[int] = Field(default=None, ge=0)
    total_timeout: Optional[str | int] = None
    per_attempt_timeout: Optional[str | int] = None
    retryable_status_codes: Optional[List[Annotated[int, Field(ge=100, le=599)]]] = None

    @field_validator("total_timeout", "per_attempt_timeout")
    @classmethod
    def _non_negative_duration(cls, value: Optional[str | int]) -> Optional[str | int]:
        if value is not None and _fallback_duration_seconds(value) < 0:
            raise ValueError("must not be negative")
        return value

    @model_validator(mode="after")
    def _per_attempt_within_total(self) -> "FallbackOverride":
        if self.total_timeout is None or self.per_attempt_timeout is None:
            return self
        total = _fallback_duration_seconds(self.total_timeout)
        per_attempt = _fallback_duration_seconds(self.per_attempt_timeout)
        if total > 0 and per_attempt > total:
            raise ValueError("per_attempt_timeout cannot exceed total_timeout")
        return self


class FallbackCircuitBreaker(BaseModel):
    """Per-backend back-off of a recipe's or the global fallback policy."""

    model_config = ConfigDict(extra="forbid")

    consecutive_failures: Optional[int] = Field(default=None, ge=0)
    cooldown_period: Optional[str | int] = None
    half_open_probes: Optional[int] = Field(default=None, ge=0)


class FallbackPolicy(FallbackOverride):
    """A recipe's cross-model fallback policy (routing.fallback): the
    decision fields plus the version and the per-backend circuit breaker."""

    version: Optional[Literal[0, 1]] = None
    circuit_breaker: Optional[FallbackCircuitBreaker] = None


class Decision(BaseModel):
    """Routing decision configuration."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: Optional[str] = None
    priority: int = 0
    tier: int = Field(default=0, ge=0)
    # A decision without an explicit rule is the canonical match-all fallback.
    # This mirrors the Go runtime and the DSL `ROUTE` form without `WHEN`.
    rules: Rules = Field(default_factory=Rules)
    action: Optional[DecisionAction] = None
    output_contract: Optional[str] = None
    output_contract_spec: Optional[OutputContractSpec] = None
    modelRefs: List[ModelRef] = Field(default_factory=list, alias="modelRefs")
    algorithm: Optional[AlgorithmConfig] = None  # Multi-model orchestration algorithm
    adaptations: Optional[DecisionAdaptationsConfig] = None
    reliability: Optional[DecisionReliability] = None
    fallback: Optional[FallbackOverride] = None
    plugins: Optional[List[PluginConfig]] = []
    retention: Optional[DecisionRetention] = None
    annotations: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def reject_native_chat_directives(cls, data):
        if not isinstance(data, dict):
            return data
        algorithm = data.get("algorithm")
        # These Chat directives have no native owner; reject before Pydantic
        # can discard fields outside its typed view.
        escalate = algorithm.get("escalate") if isinstance(algorithm, dict) else None
        if (
            isinstance(escalate, dict)
            and algorithm.get("type") == "escalate"
            and escalate.get("stages")
            and (data.get("candidateIterations") or data.get("retention"))
        ):
            raise ValueError(
                "candidateIterations and retention are unsupported for native execution"
            )
        return data

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
    def validate_minimum_candidates(self):
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
    cache_write_1h_per_1m: Optional[float] = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    completion_per_1m: Optional[float] = Field(default=0.0, ge=0, allow_inf_nan=False)


class ProviderReliability(BaseModel):
    """Load balancing, timeouts, retries and endpoint health for one provider
    model, honored the same way by every data plane."""

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
    connect_timeout: Optional[str] = None
    total_timeout: Optional[str] = None
    idle_timeout: Optional[str] = None
    per_try_timeout: Optional[str] = None
    first_byte_timeout: Optional[str] = Field(
        default=None,
        description="Wait for the first response body byte, so a stalled stream "
        "can still be retried. Only standalone mode honors it.",
    )
    retriable_status_codes: List[Annotated[int, Field(ge=100, le=599)]] = Field(
        default_factory=list
    )
    retry_back_off_base: Optional[str] = None
    retry_back_off_max: Optional[str] = None
    retry_after_max: Optional[str] = None
    retry_budget_percent: Optional[float] = Field(default=None, gt=0, le=100)
    retry_budget_min_concurrency: Optional[int] = Field(default=None, ge=0)

    @field_validator(
        "base_ejection_time",
        "health_check_interval",
        "health_check_timeout",
        "connect_timeout",
        "per_try_timeout",
        "first_byte_timeout",
        "retry_back_off_base",
        "retry_back_off_max",
        "retry_after_max",
    )
    @classmethod
    def _positive_duration(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and parse_duration(value) <= 0:
            raise ValueError("must be a positive duration such as 30s")
        return value

    @field_validator("total_timeout", "idle_timeout")
    @classmethod
    def _duration_or_zero(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            parse_duration(value)
        return value

    @model_validator(mode="after")
    def _back_off_bounds(self) -> "ProviderReliability":
        if self.retry_back_off_max is None:
            return self
        base = parse_duration(self.retry_back_off_base or "25ms")
        if parse_duration(self.retry_back_off_max) < base:
            raise ValueError("retry_back_off_max must not be below retry_back_off_base")
        return self


class Reasoning(BaseModel):
    """Built-in family reference or inline behavior for a custom model."""

    model_config = ConfigDict(extra="forbid")

    family: Optional[str] = None
    type: Optional[
        Literal[
            "chat_template_kwargs",
            "reasoning_effort",
            "reasoning_mode",
            "top_level_reasoning_effort",
        ]
    ] = None
    parameter: Optional[str] = None
    activation_parameter: Optional[str] = None
    effort_flags: Dict[str, str] = Field(default_factory=dict)
    levels: List[str] = Field(default_factory=list)
    default: Optional[str] = None
    modes: List[Literal["enabled", "disabled", "adaptive"]] = Field(
        default_factory=list
    )
    default_mode: Optional[Literal["enabled", "disabled", "adaptive"]] = None
    disabled: Optional[str] = None

    def _has_inline_fields(self) -> bool:
        return bool(
            self.type
            or self.parameter
            or self.activation_parameter
            or self.effort_flags
            or self.levels
            or self.default
            or self.modes
            or self.default_mode
            or self.disabled
        )

    def _validate_reference_shape(self, inline: bool) -> None:
        if self.family and inline:
            raise ValueError(
                "family and inline reasoning fields are mutually exclusive"
            )
        if not self.family and not inline:
            raise ValueError(
                "reasoning must reference a family or define inline behavior"
            )
        if inline and (not self.type or not self.parameter):
            raise ValueError("inline reasoning requires type and parameter")

    def _validate_activation_parameter(self) -> None:
        if self.activation_parameter:
            if not self.activation_parameter.strip():
                raise ValueError(
                    "inline reasoning activation_parameter cannot be blank"
                )
            if self.activation_parameter == self.parameter:
                raise ValueError(
                    "inline reasoning activation_parameter must differ from parameter"
                )
            if self.type != "reasoning_effort":
                raise ValueError(
                    "inline reasoning activation_parameter requires reasoning_effort type"
                )

    def _validate_effort_flags(self) -> None:
        if self.effort_flags:
            if self.type != "reasoning_effort":
                raise ValueError(
                    "inline reasoning effort_flags requires reasoning_effort type"
                )
            if not self.activation_parameter:
                raise ValueError(
                    "inline reasoning effort_flags requires activation_parameter"
                )
            if any(effort not in self.levels for effort in self.effort_flags):
                raise ValueError(
                    "inline reasoning effort_flags keys must be listed in levels"
                )
            parameters = list(self.effort_flags.values())
            if any(not parameter.strip() for parameter in parameters):
                raise ValueError("inline reasoning effort_flags values cannot be blank")
            if len(parameters) != len(set(parameters)):
                raise ValueError("inline reasoning effort_flags values must be unique")
            if self.parameter in parameters or self.activation_parameter in parameters:
                raise ValueError(
                    "inline reasoning effort_flags values must differ from parameter and activation_parameter"
                )
            active_levels = [level for level in self.levels if level != self.disabled]
            if len(active_levels) - len(self.effort_flags) > 1:
                raise ValueError(
                    "inline reasoning effort_flags cannot leave multiple levels indistinguishable by omission"
                )

    def _validate_effort_levels(self) -> None:
        if self.levels and self.default is not None and self.default not in self.levels:
            raise ValueError("inline reasoning default must be listed in levels")
        if (
            self.type
            in {
                "reasoning_effort",
                "top_level_reasoning_effort",
            }
            and not self.levels
        ):
            raise ValueError("inline effort-based reasoning requires levels")
        # Operator-authored families may omit the mode contract entirely. This
        # keeps the user-facing custom-model form small for the common cases
        # where a boolean activation flag or an effort value already describes
        # the complete wire behavior. Built-in families are validated strictly
        # by the catalog generator.

    def _validate_modes(self) -> None:
        if not self.modes and self.default_mode is not None:
            raise ValueError(
                "inline reasoning modes are required when default_mode is set"
            )
        if self.modes and self.default_mode is None:
            raise ValueError(
                "inline reasoning default_mode is required when modes are set"
            )
        if self.default_mode is not None and self.default_mode not in self.modes:
            raise ValueError("inline reasoning default_mode must be listed in modes")
        if self.disabled and self.modes and "disabled" not in self.modes:
            raise ValueError("inline reasoning disabled value requires disabled mode")

    @model_validator(mode="after")
    def validate_shape(self):
        inline = self._has_inline_fields()
        self._validate_reference_shape(inline)
        if not inline:
            return self
        self._validate_activation_parameter()
        self._validate_effort_flags()
        self._validate_effort_levels()
        self._validate_modes()
        return self


class Model(BaseModel):
    """Provider model binding under the existing providers.models hierarchy."""

    model_config = ConfigDict(extra="forbid")

    name: str
    catalog: Optional[str] = None
    reasoning: Optional[Reasoning] = None
    provider_model_id: Optional[str] = None
    deployment: Optional[str] = None
    backend_refs: List["BackendRef"] = Field(default_factory=list)
    pricing: Optional[ModelPricing] = None
    reliability: Optional[ProviderReliability] = None
    api_format: Optional[str] = None
    external_model_ids: Optional[Dict[str, str]] = None


class LoRAAdapter(BaseModel):
    """LoRA adapter metadata exposed under routing.modelCards[].loras."""

    name: str
    description: Optional[str] = None


class EvaluationRecord(BaseModel):
    """Small operator-authored benchmark result linked to one model card."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    benchmark: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)+@[0-9]+(?:\.[0-9]+\.[0-9]+)?$",
    )
    benchmark_profile: Optional[str] = None
    reasoning_effort: Optional[str] = None
    metrics: Dict[str, float]
    source: Optional[str] = None
    measured_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_measurement(self):
        if not self.metrics:
            raise ValueError("metrics cannot be empty")
        if any(
            not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", key) or not math.isfinite(value)
            for key, value in self.metrics.items()
        ):
            raise ValueError("metrics must contain named finite numbers")
        if any(
            isinstance(value, (dict, list, tuple, set))
            or (isinstance(value, float) and not math.isfinite(value))
            for value in self.metadata.values()
        ):
            raise ValueError("metadata values must be scalar")
        if any(not key.strip() for key in self.metadata):
            raise ValueError("metadata keys cannot be empty")
        if self.measured_at:
            try:
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.measured_at):
                    raise ValueError
                date.fromisoformat(self.measured_at)
            except ValueError as error:
                raise ValueError("measured_at must use YYYY-MM-DD") from error
        return self


EVALUATION_RESOURCE_ID_PATTERN = (
    r"^[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)+" r"@[0-9]+(?:\.[0-9]+\.[0-9]+)?$"
)


class EvaluationBenchmarkProfile(BaseModel):
    """One reproducible execution profile for an operator benchmark."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class EvaluationNormalizationPoint(BaseModel):
    """One point in a piecewise-linear metric normalization."""

    model_config = ConfigDict(extra="forbid")

    input: float
    output: float = Field(ge=0, le=1)


class EvaluationNormalization(BaseModel):
    """Maps one benchmark metric onto the common [0, 1] utility scale."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "identity",
        "one_minus",
        "linear_clamp",
        "piecewise_linear",
        "logistic",
        "lookup",
    ] = "identity"
    min: Optional[float] = None
    max: Optional[float] = None
    k: Optional[float] = None
    x0: Optional[float] = None
    points: List[EvaluationNormalizationPoint] = Field(default_factory=list)
    values: Dict[str, float] = Field(default_factory=dict)


class EvaluationBenchmarkMetric(BaseModel):
    """Metric semantics owned by a versioned benchmark definition."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    unit: str = Field(min_length=1)
    direction: Literal["higher_is_better", "lower_is_better"]
    range: List[float] = Field(min_length=2, max_length=2)
    normalization: Optional[EvaluationNormalization] = None

    @field_validator("range")
    @classmethod
    def validate_range(cls, value: List[float]) -> List[float]:
        if any(not math.isfinite(item) for item in value) or value[0] >= value[1]:
            raise ValueError("range must contain two finite increasing values")
        return value


class EvaluationBenchmarkDefinition(BaseModel):
    """Operator-owned, namespaced benchmark contract."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=EVALUATION_RESOURCE_ID_PATTERN)
    display_name: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    tags: List[str] = Field(default_factory=list)
    source: Optional[str] = None
    default_profile: str = Field(min_length=1)
    profiles: List[EvaluationBenchmarkProfile] = Field(min_length=1)
    metrics: List[EvaluationBenchmarkMetric] = Field(min_length=1)


class EvaluationMissingPolicy(BaseModel):
    """Completeness policy for a custom capability or aggregate index."""

    model_config = ConfigDict(extra="forbid")

    policy: Literal["require_all", "reported_only", "require_coverage"]
    minimum: Optional[float] = Field(default=None, gt=0, le=1)


class EvaluationIndexComponent(BaseModel):
    """One benchmark metric or nested index in an index DAG."""

    model_config = ConfigDict(extra="forbid")

    benchmark: Optional[str] = None
    metric: Optional[str] = None
    benchmark_profile: Optional[str] = None
    benchmark_profiles: List[str] = Field(default_factory=list)
    index: Optional[str] = None
    weight: float = Field(gt=0)
    normalization: EvaluationNormalization = Field(
        default_factory=EvaluationNormalization
    )

    @model_validator(mode="after")
    def validate_reference(self):
        if (self.metric is None) == (self.index is None):
            raise ValueError("component must reference exactly one metric or index")
        return self


class EvaluationIndexDefinition(BaseModel):
    """Operator-owned, versioned capability or aggregate index."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=EVALUATION_RESOURCE_ID_PATTERN)
    display_name: str = Field(min_length=1)
    description: Optional[str] = None
    methodology: Optional[str] = None
    aggregation: Literal["weighted_mean"]
    scale: List[float] = Field(min_length=2, max_length=2)
    missing: EvaluationMissingPolicy
    domains: Dict[str, float] = Field(default_factory=dict)
    components: List[EvaluationIndexComponent] = Field(min_length=1)


class Evaluation(BaseModel):
    """Unified benchmark definitions, index DAGs, and model measurements."""

    model_config = ConfigDict(extra="forbid")

    calibrations: List[CalibrationArtifact] = Field(default_factory=list)
    benchmarks: List[EvaluationBenchmarkDefinition] = Field(default_factory=list)
    indices: List[EvaluationIndexDefinition] = Field(default_factory=list)
    records: List[EvaluationRecord] = Field(default_factory=list)


class RoutingModelPresentation(BaseModel):
    """Optional product presentation for a handwritten model card."""

    model_config = ConfigDict(extra="forbid")

    logo: str
    monogram: str
    monochrome: bool = False


class RoutingModelDistribution(BaseModel):
    """Distribution metadata for a handwritten physical or virtual model."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["open_weights", "proprietary_api", "router_recipe"]
    source: str
    license: Optional[str] = None


class RoutingModel(BaseModel):
    """Handwritten custom card or explicit overlay of a built-in card."""

    model_config = ConfigDict(extra="forbid")

    name: str
    display_name: Optional[str] = None
    publisher: Optional[str] = None
    presentation: Optional[RoutingModelPresentation] = None
    distribution: Optional[RoutingModelDistribution] = None
    family: Optional[str] = None
    revision: Optional[str] = None
    released_at: Optional[str] = None
    knowledge_cutoff: Optional[str] = None
    lifecycle: Optional[str] = None
    param_size: Optional[str] = None
    context_window_size: Optional[int] = Field(default=None, ge=1)
    max_output_tokens: Optional[int] = Field(default=None, ge=1)
    description: Optional[str] = None
    capabilities: Optional[List[str]] = None
    loras: Optional[List[LoRAAdapter]] = None
    tags: Optional[List[str]] = None
    modalities: Optional[Dict[str, List[str]]] = None
    modality: Optional[str] = None

    @field_validator("released_at", "knowledge_cutoff", mode="before")
    @classmethod
    def normalize_date_metadata(cls, value: Any) -> Any:
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return value


class ReasoningFamily(BaseModel):
    """Reasoning family configuration."""

    type: str
    parameter: str
    activation_parameter: Optional[str] = None
    effort_flags: Dict[str, str] = Field(default_factory=dict)
    levels: List[str] = Field(default_factory=list)
    default: Optional[str] = None
    modes: List[Literal["enabled", "disabled", "adaptive"]]
    default_mode: Literal["enabled", "disabled", "adaptive"]
    disabled: Optional[str] = None


# Backend ref settings the Router removed; mirrors removedBackendRefFields in
# src/semantic-router/pkg/config/removed_backend_ref_fields.go.
REMOVED_BACKEND_REF_FIELDS = ("compat_gateway", "openai_compat_gateway")


class BackendRef(BaseModel):
    """Inline backend access details carried under providers.models[].backend_refs."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    endpoint: Optional[str] = None
    protocol: str = "http"
    weight: int = Field(default=1, ge=0)
    base_url: Optional[str] = None
    provider: str = "vllm"
    auth_header: Optional[str] = None
    auth_prefix: Optional[str] = None
    extra_headers: Optional[Dict[str, str]] = None
    api_version: Optional[str] = None
    chat_path: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    # The backend receives the request's deployment-injected gateway credential
    # (global.services.authz.gateway_credential) instead of a static key.
    trusted_gateway: bool = False

    @model_validator(mode="before")
    @classmethod
    def reject_removed_fields(cls, data):
        # Every backend now accepts response fields its api_format does not
        # define, so the opt-ins are gone; name them instead of reporting a
        # generic extra field, as the Router does.
        if isinstance(data, dict):
            for field in REMOVED_BACKEND_REF_FIELDS:
                if field in data:
                    raise ValueError(
                        f"{field} was removed; gateway response fields are "
                        "accepted by default, so delete it"
                    )
        return data

    def resolve_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            import os

            return os.getenv(self.api_key_env)
        return None


class ProviderDefaults(BaseModel):
    """Provider-wide defaults that should not be mixed into per-model access bindings."""

    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = None
    reasoning_effort: Optional[str] = None


class Providers(BaseModel):
    """Provider configuration."""

    model_config = ConfigDict(extra="forbid")

    defaults: ProviderDefaults = Field(default_factory=ProviderDefaults)
    models: List[Model] = Field(default_factory=list)

    @property
    def default_model(self) -> Optional[str]:
        return self.defaults.model

    @property
    def default_reasoning_effort(self) -> Optional[str]:
        return self.defaults.reasoning_effort


class PairScorerSelection(BaseModel):
    """An immutable trained exit; zero resolves to actual full depth/width."""

    model_config = ConfigDict(extra="forbid")
    layer: int = Field(default=0, ge=0)
    dimension: int = Field(default=0, ge=0)


class OperatingPointReference(BaseModel):
    """Explicit immutable score policy; relative paths are inside the deployment."""

    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_policy_path(cls, value):
        normalized = posixpath.normpath(value)
        if value != value.strip() or (
            not posixpath.isabs(value)
            and (normalized == ".." or normalized.startswith("../"))
        ):
            raise ValueError(
                "operating point path must be trimmed and stay inside the artifact"
            )
        return value


class ModelBinding(BaseModel):
    """The deployment that answers one shared-answer signal, globally or in one recipe.

    A binding is written as the deployment's name (`pii: local-vela`); the
    object form also takes a head. Its contract, adapter, mapping_path,
    pair_scorer and operating_point are deprecated: the Router derives the
    contract from the signal and the adapter from the deployment's provider.
    A rule that asks its own question names its deployment on the rule.
    """

    model_config = ConfigDict(extra="forbid")

    deployment: str
    head: Optional[str] = None
    contract: Optional[str] = None
    adapter: Optional[str] = None
    mapping_path: Optional[str] = None
    pair_scorer: Optional[PairScorerSelection] = None
    operating_point: Optional[OperatingPointReference] = None

    @model_validator(mode="before")
    @classmethod
    def read_deployment_name(cls, value):
        if isinstance(value, str):
            return {"deployment": value.strip()}
        return value

    @model_serializer(mode="wrap")
    def write_deployment_name(self, handler):
        if all(
            getattr(self, name) is None
            for name in type(self).model_fields
            if name != "deployment"
        ):
            return self.deployment
        return handler(self)


class CandidateRequirements(BaseModel):
    """Recipe candidate constraints; input token accounting remains estimated."""

    model_config = ConfigDict(extra="forbid")

    capabilities: Optional[Literal["declared"]] = None
    context: Optional[Literal["known_limits"]] = None


class Routing(BaseModel):
    """Canonical routing block."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    model_cards: List[RoutingModel] = Field(default_factory=list, alias="modelCards")
    model_bindings: Dict[str, ModelBinding] = Field(default_factory=dict)
    candidate_requirements: Optional[CandidateRequirements] = None
    signals: Signals = Field(default_factory=Signals)
    projections: Projections = Field(default_factory=Projections)
    decisions: List[Decision] = Field(default_factory=list)
    strategy: Optional[RoutingStrategy] = None
    fallback: Optional[FallbackPolicy] = None


class Entrypoint(BaseModel):
    """Request-facing virtual model names mapped to one routing recipe."""

    model_config = ConfigDict(extra="forbid")

    api: Literal["chat", "systemone"] | None = None
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

    model_bindings: Dict[str, ModelBinding] = Field(default_factory=dict)
    candidate_requirements: Optional[CandidateRequirements] = None
    signals: Signals = Field(default_factory=Signals)
    projections: Projections = Field(default_factory=Projections)
    decisions: List[Decision] = Field(default_factory=list)
    strategy: Optional[RoutingStrategy] = None
    fallback: Optional[FallbackPolicy] = None


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


class UserConfig(BaseModel):
    """Canonical v0.3 user configuration."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    version: str
    listeners: List[Listener] = Field(default_factory=list)
    providers: Providers = Field(default_factory=Providers)
    evaluation: Optional[Evaluation] = None
    routing: Routing = Field(default_factory=Routing)
    entrypoints: List[Entrypoint] = Field(default_factory=list)
    recipes: List[Recipe] = Field(default_factory=list)
    global_: Optional[Dict[str, Any]] = Field(default=None, alias="global")
    setup: Optional[Dict[str, Any]] = None

    @field_validator("global_")
    @classmethod
    def validate_routing_mode(cls, value):
        router = (value or {}).get("router") or {}
        if not isinstance(router.get("enabled", True), bool):
            raise ValueError("global.router.enabled must be a boolean")
        return value

    @property
    def routing_enabled(self) -> bool:
        router = (self.global_ or {}).get("router") or {}
        return router.get("enabled", True)

    @property
    def signals(self) -> Signals:
        return self.routing.signals

    @property
    def decisions(self) -> List[Decision]:
        return self.routing.decisions


# Resolve forward references for recursive condition trees.
Condition.model_rebuild()
Model.model_rebuild()
