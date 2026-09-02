"""Public report contracts kept in lockstep with the Dashboard types."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.contracts import StrictModel
from cli.evaluation.gate_contract import GATE_CONTRACT_VERSION, ChangeProfile

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
GateVerdict = Literal["pass", "fail", "unavailable", "waived", "not_applicable"]


class EvaluationRunProgress(StrictModel):
    percent: float = Field(ge=0, le=100)
    completed: int = Field(ge=0)
    total: int = Field(ge=0)
    current_track_id: TrackID | None = None
    message: str | None = None


class EvaluationRun(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    name: str
    description: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    mode: Literal["replay", "live"]
    evidence_level: EvidenceLevel
    target_id: str
    change_profile: ChangeProfile
    suite_ids: tuple[str, ...]
    track_ids: tuple[TrackID, ...]
    sample_limit: int
    concurrency: int
    seed: int
    baseline_run_id: str | None = None
    progress: EvaluationRunProgress
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class EvaluationCoverage(StrictModel):
    evaluated: int = Field(ge=0)
    total: int = Field(ge=0)
    fraction: float = Field(ge=0, le=1)
    unavailable: int | None = Field(default=None, ge=0)
    confidence_level: float | None = Field(default=None, gt=0, lt=1)
    confidence_interval: tuple[float, float] | None = None


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


class GateThreshold(StrictModel):
    operator: str
    value: float
    unit: str | None = None


class EvaluationGate(StrictModel):
    id: str
    name: str
    description: str | None = None
    track_id: TrackID | None = None
    disposition: Literal["required", "advisory", "not_applicable", "waived"]
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


class EvaluationArtifact(StrictModel):
    id: str
    name: str
    kind: str
    uri: str | None = None
    digest: str | None = None
    media_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)


class EvaluationProvenance(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
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


class EvaluationTrackReport(StrictModel):
    track_id: TrackID
    status: Literal[
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
        "unavailable",
        "skipped",
    ]
    evidence_level: EvidenceLevel
    summary: str
    coverage: EvaluationCoverage
    metrics: tuple[EvaluationMetric, ...]
    gates: tuple[EvaluationGate, ...]
    artifacts: tuple[EvaluationArtifact, ...] = ()
    error: str | None = None


class EvaluationReportSummary(StrictModel):
    verdict: GateVerdict
    quality_score: float | None
    latency_p95_ms: float | None
    runtime_cost: float | None
    capacity_tco: float | None
    coverage: EvaluationCoverage
    passed_gates: int = Field(ge=0)
    failed_gates: int = Field(ge=0)
    unavailable_gates: int = Field(ge=0)


class EvaluationReport(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    run: EvaluationRun
    summary: EvaluationReportSummary
    tracks: tuple[EvaluationTrackReport, ...]
    metrics: tuple[EvaluationMetric, ...]
    gates: tuple[EvaluationGate, ...]
    costs: EvaluationCostLedgers
    recommendations: tuple[str, ...]
    provenance: EvaluationProvenance
    artifacts: tuple[EvaluationArtifact, ...]

    @model_validator(mode="after")
    def coherent_gate_contract(self) -> EvaluationReport:
        expected_ids = tuple(f"G{index}" for index in range(10))
        if tuple(gate.id for gate in self.gates) != expected_ids:
            raise ValueError("report must contain G0 through G9 exactly once in order")
        if any(gate.change_profile != self.run.change_profile for gate in self.gates):
            raise ValueError("report gates must match the run change profile")
        return self


class EvaluationComparison(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    baseline_run_id: str
    candidate_run_id: str
    verdict: GateVerdict
    summary: str
    metrics: tuple[EvaluationMetric, ...]
    gates: tuple[EvaluationGate, ...]
    recommendations: tuple[str, ...]
    created_at: datetime | None = None


class WorkerEvent(StrictModel):
    type: str
    message: str
    track_id: TrackID | None = None
    progress: EvaluationRunProgress | None = None
    payload: dict[str, object] | None = None
