"""Shared evidence values emitted by workers and verified by the Dashboard."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.gate_contract import (
    GATE_CONTRACT_VERSION,
    ChangeProfile,
    GateDisposition,
)
from cli.evaluation.metric_analysis_catalog import (
    PROVENANCE_CONTRACT_VERSION as METRIC_ANALYSIS_CONTRACT_VERSION,
)
from cli.evaluation.metric_analysis_catalog import (
    CatalogMetricAnalysisSpecification as MetricAnalysisSpecification,
)
from cli.evaluation.metric_analysis_catalog import (
    resolve_metric_analysis,
)

TrackID = Literal[
    "routing",
    "model_pool",
    "joint",
    "agentic",
    "multimodal",
    "preference",
    "safety",
    "capacity",
]
EvidenceLevel = Literal["E0", "E1", "E2", "E3", "E4", "E5"]
GateVerdict = Literal["pass", "fail", "unavailable", "not_applicable"]
DecisionVerdict = Literal["pass", "fail", "unavailable"]

_MAX_METRIC_ANALYSIS_IDENTIFIER_LENGTH = 160


def metric_analysis_specification(metric_id: str) -> MetricAnalysisSpecification:
    """Resolve one exact catalog contract; unknown metrics fail closed."""

    return resolve_metric_analysis(metric_id).specification


class EvaluationCoverage(StrictModel):
    evaluated: int = Field(ge=0)
    total: int = Field(ge=0)
    fraction: float = Field(ge=0, le=1)
    unavailable: int | None = Field(default=None, ge=0)
    confidence_level: float | None = Field(default=None, gt=0, lt=1)
    confidence_interval: tuple[float, float] | None = None


class MetricAnalysisProvenance(StrictModel):
    """Auditable estimator contract required for every published metric."""

    contract_version: Literal[METRIC_ANALYSIS_CONTRACT_VERSION]
    estimator_id: str
    estimator_version: str
    analysis_unit: str
    cluster_unit: str
    weighting: str
    missingness: Literal["fail_closed"]
    exclusion_policy: Literal["exclude_unavailable_evidence"]
    observed_exclusions: int = Field(ge=0)

    @field_validator(
        "estimator_id",
        "estimator_version",
        "analysis_unit",
        "cluster_unit",
        "weighting",
    )
    @classmethod
    def validate_contract_identifier(cls, value: str) -> str:
        if (
            not value
            or value.strip() != value
            or len(value) > _MAX_METRIC_ANALYSIS_IDENTIFIER_LENGTH
        ):
            raise ValueError(
                "metric analysis provenance identifiers must be trimmed and non-blank"
            )
        return value


class EvaluationMetric(StrictModel):
    id: str
    name: str
    track_id: TrackID | None = None
    value: float | None
    unit: str
    direction: Literal["higher_is_better", "lower_is_better", "target"] | None = None
    baseline_value: float | None = None
    delta: float | None = None
    confidence_interval: tuple[float, float] | None = None
    sample_count: int | None = Field(default=None, ge=0)
    analysis_provenance: MetricAnalysisProvenance

    @model_validator(mode="after")
    def analysis_provenance_matches_registered_metric(self) -> EvaluationMetric:
        specification = metric_analysis_specification(self.id)
        provenance = self.analysis_provenance
        if (
            provenance.estimator_id != specification.estimator_id
            or provenance.estimator_version != specification.estimator_version
            or provenance.analysis_unit != specification.analysis_unit
            or provenance.cluster_unit != specification.cluster_unit
            or provenance.weighting != specification.weighting
            or provenance.missingness != specification.missingness
            or provenance.exclusion_policy != specification.exclusion_policy
        ):
            raise ValueError(
                "metric analysis provenance does not match the registered estimator"
            )
        return self


class GateThreshold(StrictModel):
    operator: str
    value: float
    unit: str | None = None


class EvaluationGate(StrictModel):
    id: str
    name: str
    description: str | None = None
    track_id: TrackID | None = None
    disposition: GateDisposition
    verdict: GateVerdict
    change_profile: ChangeProfile
    contract_version: Literal[GATE_CONTRACT_VERSION]
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    evidence_level: EvidenceLevel | None = None
    observed: float | None = None
    threshold: GateThreshold | None = None
    sample_count: int | None = Field(default=None, ge=0)
    coverage: EvaluationCoverage | None = None
    owner: str | None = Field(default=None, min_length=1, max_length=160)
    evaluated_at: datetime | None = None
    rationale: str | None = None

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(not item.strip() for item in value):
            raise ValueError("gate evidence refs must be unique and non-blank")
        return value

    @model_validator(mode="after")
    def validate_disposition_contract(self) -> EvaluationGate:
        not_applicable = self.disposition == "not_applicable"
        if not_applicable != (self.verdict == "not_applicable"):
            raise ValueError("not-applicable gate disposition and verdict must match")
        if not_applicable and (self.observed is not None or self.threshold is not None):
            raise ValueError(
                "not-applicable gate cannot publish an observation or threshold"
            )
        return self


class EvaluationArtifact(StrictModel):
    id: str
    name: str
    kind: str
    uri: str | None = None
    digest: str | None = None
    media_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)


class EvaluationProvenance(StrictModel):
    schema_version: Literal[SCHEMA_VERSION]
    generated_at: datetime
    code_revision: str | None = None
    benchmark_revisions: dict[str, str] | None = None
    workload_snapshot_digest: str | None = None
    policy_snapshot_digest: str | None = None
    binding_snapshot_digest: str | None = None
    pool_snapshot_digest: str | None = None
    environment_snapshot_digest: str | None = None
    target_id: str
    seed: int
    redaction_policy: str | None = None


class EvaluationCostAmount(StrictModel):
    amount: float | None = Field(default=None, ge=0)
    currency: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    gpu_seconds: float | None = Field(default=None, ge=0)
    energy_kwh: float | None = Field(default=None, ge=0)


class EvaluationCostLedgers(StrictModel):
    runtime: EvaluationCostAmount
    evaluation_overhead: EvaluationCostAmount
    capacity_tco: EvaluationCostAmount
