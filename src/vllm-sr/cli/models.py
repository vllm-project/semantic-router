"""Pydantic models for vLLM Semantic Router configuration."""

from typing import List, Dict, Any, Optional
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


class Signals(BaseModel):
    """All signal configurations."""

    keywords: Optional[List[KeywordSignal]] = []
    embeddings: Optional[List[EmbeddingSignal]] = []
    domains: Optional[List[Domain]] = []
    fact_check: Optional[List[FactCheck]] = []
    user_feedbacks: Optional[List[UserFeedback]] = []
    preferences: Optional[List[Preference]] = []


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


class AlgorithmConfig(BaseModel):
    """Algorithm configuration for multi-model decisions.

    Specifies how multiple models in a decision should be orchestrated.
    """

    # Algorithm type: "sequential", "confidence", "concurrent"
    type: str

    # Algorithm-specific configurations (only one should be set based on type)
    confidence: Optional[ConfidenceAlgorithmConfig] = None
    concurrent: Optional[ConcurrentAlgorithmConfig] = None


class PluginConfig(BaseModel):
    """Plugin configuration."""

    type: str
    configuration: Dict[str, Any]


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
