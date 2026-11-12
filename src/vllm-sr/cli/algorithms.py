"""Algorithm configuration models for multi-model orchestration."""

import math
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from cli.config_algorithm_forms import REMOVED_ALGORITHM_TYPES
from cli.config_schema import routing_surface_catalog, surface_types

from .config_contract import QuorumFailurePolicy
from .models_native import AlgorithmBudget, EscalateAlgorithm
from .models_systemone import SystemOneSelectionConfig

SUPPORTED_ALGORITHM_TYPES = frozenset(surface_types("algorithms"))

# Executors call several candidates and read the shared budget and on_error;
# selectors choose one candidate before calling it.
EXECUTOR_ALGORITHM_TYPES = frozenset(
    entry["type"]
    for entry in routing_surface_catalog().get("algorithms", [])
    if isinstance(entry, dict) and entry.get("kind") == "executor"
)

# Types the Router retired, whose blocks are refused rather than passed on as
# a Router build's own algorithm.
RETIRED_ALGORITHM_TYPES = frozenset(
    {
        "policy",
        "session_aware",
        "elo",
        "rl_driven",
        "gmtrouter",
        "bandit",
        "personalization",
        "thompson",
        "router_r1",
    }
)


class ModelRef(BaseModel):
    """Model reference in decision."""

    model: str
    # Omission is not an authored false: runtime serialization must preserve
    # the caller's choice and leave canonical defaulting to the Router.
    use_reasoning: bool | None = None
    reasoning_description: str | None = None
    reasoning_mode: Literal["enabled", "disabled", "adaptive"] | None = None
    reasoning_effort: str | None = None  # Model-specific reasoning effort level.
    lora_name: str | None = None  # LoRA adapter name (if using LoRA)
    weight: float | None = None


class RatingsAlgorithmConfig(BaseModel):
    """Configuration for the ratings executor."""

    model_config = ConfigDict(extra="forbid")

    max_concurrent: int | None = Field(default=None, ge=1)


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

    # Temperature for model calls; unset sends the request's own
    temperature: float | None = None

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
    # responses when algorithm.on_error="skip".
    round_timeout_seconds: int | None = Field(default=None, ge=1)

    # Return from a parallel round as soon as this many responses succeed.
    min_successful_responses: int | None = Field(default=None, ge=1)

    # Random seed for shuffling responses (for reproducibility); unset is 0
    shuffle_seed: int | None = None

    # Whether to include intermediate responses in the response body
    include_intermediate_responses: bool | None = False

    # Maximum number of responses to keep per round (for memory efficiency)
    max_responses_per_round: int | None = None


class FusionGroundingConfig(BaseModel):
    """Configuration for grounding-aware fusion.

    Scores each panel response for faithfulness before the judge synthesizes,
    then ranks/filters the panel. Uses the router's hallucination detector
    (against the request's context, else against each peer response) and makes
    no extra LLM calls. Bounds here MUST match the Go validator in
    pkg/config/fusion_config.go (ValidateFusionGroundingConfig).
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
    contradiction_penalty: float | None = Field(default=1.0, ge=0)
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
    analysis_mode: Literal["separate", "one_call", "none"] = "separate"
    analysis_overrides: list[FusionModelOverrideConfig] | None = None
    max_concurrent: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    round_timeout_seconds: int | None = Field(default=None, ge=1)
    min_successful_responses: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    include_analysis: bool | None = True
    include_intermediate_responses: bool | None = True
    # Panel-level policy for a usable-response count below
    # min_successful_responses. Independent of algorithm.on_error, which only
    # governs a single failed panel attempt. Bounds here MUST match the Go validator in
    # pkg/config/fusion_config.go (validateFusionQuorumFailure).
    quorum_failure_policy: QuorumFailurePolicy | None = None
    quorum_fallback_target: str | None = None
    analysis_template: str | None = None
    synthesis_template: str | None = None
    judge_prompt_version: str | None = "fusion-v1"
    grounding: FusionGroundingConfig | None = None

    @model_validator(mode="after")
    def validate_quorum_failure_policy(self):
        target = (self.quorum_fallback_target or "").strip()
        if self.quorum_failure_policy == "fallback" and not target:
            raise ValueError(
                "quorum_failure_policy 'fallback' requires quorum_fallback_target"
            )
        if self.quorum_failure_policy != "fallback" and target:
            raise ValueError(
                "quorum_fallback_target requires quorum_failure_policy 'fallback'"
            )
        return self

    @model_validator(mode="after")
    def validate_analysis_override_models(self):
        seen: set[str] = set()
        for override in self.analysis_overrides or []:
            model = override.model.strip()
            if model in seen:
                raise ValueError(f"analysis override model {model!r} is duplicated")
            seen.add(model)

        if self.analysis_mode != "separate" and (
            self.analysis_template and self.analysis_template.strip()
        ):
            raise ValueError("analysis_template requires analysis_mode='separate'")
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
    models: list[str] = Field(min_length=1)
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


class QualityEvidenceConfig(BaseModel):
    """Versioned catalog evidence used as the multi-factor quality signal."""

    model_config = ConfigDict(extra="forbid")

    index: str = Field(min_length=1)
    on_missing: Literal["exclude", "disable_quality"] = "exclude"
    min_coverage: float | None = Field(default=None, ge=0, le=1)
    min_score: float | None = None

    @model_validator(mode="after")
    def validate_index(self):
        if self.index != self.index.strip():
            raise ValueError("index cannot have surrounding whitespace")
        if self.min_score is not None and not math.isfinite(self.min_score):
            raise ValueError("min_score must be finite")
        if self.min_score is not None and self.on_missing == "disable_quality":
            raise ValueError("min_score requires on_missing=exclude")
        return self


class MultiFactorPriorityConfig(BaseModel):
    """One ordered factor in a lexicographic routing objective."""

    model_config = ConfigDict(extra="forbid")

    factor: Literal["quality", "latency", "cost", "load"]
    tolerance: float = Field(default=0, ge=0, le=1)


class MultiFactorObjectiveConfig(BaseModel):
    """Weighted balance or ordered factor comparison."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["weighted", "lexicographic"] = "weighted"
    priorities: list[MultiFactorPriorityConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_strategy(self):
        if self.strategy == "weighted" and self.priorities:
            raise ValueError("priorities require strategy=lexicographic")
        if self.strategy == "lexicographic" and not self.priorities:
            raise ValueError("lexicographic objective requires priorities")
        factors = [priority.factor for priority in self.priorities]
        if len(factors) != len(set(factors)):
            raise ValueError("objective priorities cannot repeat a factor")
        return self


class MultiFactorSelectionConfig(BaseModel):
    """Configuration for the canonical multi_factor selector."""

    model_config = ConfigDict(extra="forbid")

    objective: MultiFactorObjectiveConfig | None = None
    weights: MultiFactorWeightsConfig | None = None
    slo: MultiFactorSLOConfig | None = None
    quality: QualityEvidenceConfig | None = None
    latency_percentile: int | None = Field(default=95, ge=1, le=100)
    latency_metric: Literal["ttft", "tpot"] | None = None
    on_no_candidates: Literal["cheapest", "first", "fail"] | None = "cheapest"

    @model_validator(mode="after")
    def validate_objective(self):
        if (
            self.objective is not None
            and self.objective.strategy == "lexicographic"
            and self.weights is not None
        ):
            raise ValueError(
                "weights cannot be combined with a lexicographic objective"
            )
        return self


class AlgorithmConfig(BaseModel):
    """Algorithm configuration for multi-model decisions.

    A selector chooses one candidate before calling it; an executor calls
    several and returns one response. Each type's settings are the block
    named after it; budget and on_error are the executors' shared settings.

    The supported types come from the generated Router contract.
    Cross-request systems live under global.router.learning.
    """

    model_config = ConfigDict(extra="forbid")

    # Minimum distinct decision candidates required once a model-free Recipe
    # is materialized by an Entrypoint.
    minimum_candidates: int | None = Field(default=None, ge=1)

    type: str

    # The block of an algorithm type a Router build registers beyond the
    # generated contract, under the type's name. The Router validates it.
    extensions: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def collect_extension_block(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        algorithm_type = str(data.get("type", "")).strip().lower()
        if (
            algorithm_type in SUPPORTED_ALGORITHM_TYPES
            or algorithm_type in RETIRED_ALGORITHM_TYPES
            or algorithm_type in REMOVED_ALGORITHM_TYPES
            or algorithm_type not in data
        ):
            return data
        data = dict(data)
        data["extensions"] = {algorithm_type: data.pop(algorithm_type)}
        return data

    @field_validator("type")
    @classmethod
    def normalize_algorithm_type(cls, value: str) -> str:
        return value.strip().lower()

    @model_validator(mode="after")
    def validate_algorithm_type(self):
        if self.type in RETIRED_ALGORITHM_TYPES:
            raise ValueError(f"unsupported retired algorithm type {self.type!r}")
        if (
            self.type not in SUPPORTED_ALGORITHM_TYPES
            and self.type not in self.extensions
        ):
            supported = ", ".join(sorted(SUPPORTED_ALGORITHM_TYPES))
            raise ValueError(
                f"unsupported algorithm type {self.type!r}; choose one of: {supported}"
            )
        return self

    @model_serializer(mode="wrap")
    def inline_extension_block(self, handler):
        data = handler(self)
        if isinstance(data, dict):
            data.update(self.extensions)
        return data

    # The executors' shared settings: the calls an executor may make within
    # its deadline, and what it does when its calls fail.
    budget: AlgorithmBudget | None = None
    on_error: str | None = None

    # The executors' blocks. escalate is a structural projection of the
    # generated Go schema: its Chat accept check or its System One stages.
    escalate: EscalateAlgorithm | None = None
    ratings: RatingsAlgorithmConfig | None = None
    remom: ReMoMAlgorithmConfig | None = None
    fusion: FusionAlgorithmConfig | None = None
    workflows: WorkflowsAlgorithmConfig | None = None

    # The selectors' blocks.
    multi_factor: MultiFactorSelectionConfig | None = None
    systemone: SystemOneSelectionConfig | None = None

    @model_validator(mode="after")
    def executor_settings(self):
        if self.type in EXECUTOR_ALGORITHM_TYPES:
            return self
        if self.budget is not None or self.on_error:
            raise ValueError(
                f"algorithm.budget and algorithm.on_error apply to executors; {self.type} is a selector"
            )
        return self

    @model_validator(mode="after")
    def required_blocks(self):
        if self.type == "escalate" and self.escalate is None:
            raise ValueError("algorithm.type=escalate requires escalate configuration")
        if self.type == "escalate" and self.escalate.get("stages"):
            if self.budget is None:
                raise ValueError("escalate stages require algorithm.budget")
            if self.minimum_candidates is not None:
                raise ValueError(
                    "escalate stages name their models; minimum_candidates is a Chat setting"
                )
        if self.type == "systemone" and self.systemone is None:
            raise ValueError(
                "algorithm.type=systemone requires systemone configuration"
            )
        if self.type != "systemone" and self.systemone is not None:
            raise ValueError(
                "systemone configuration requires algorithm.type=systemone"
            )
        return self
