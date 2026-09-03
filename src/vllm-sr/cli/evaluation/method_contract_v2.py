"""Clean-break contracts for executable, gradeable evaluation methods.

This module deliberately does not translate the pre-v2 method records.  A
method is either described and reduced under this contract or it is not a v2
method.  That makes a report's methodological claims independently auditable.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise
from math import isfinite
from typing import Literal

from pydantic import Field, StrictBool, field_validator, model_validator

from cli.evaluation.constants import TRACK_IDS
from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.contract_validation import validate_portable_id

EVALUATION_METHOD_CONTRACT_VERSION = "evaluation-method.v2"
COMPOUND_MODEL_BUDGET_METHOD_ID = "r2.compound-model-budget.v2"
_MIN_SHARED_BUDGET_POINTS = 2


class ActionRef(StrictModel):
    """A stable action identity; labels must never be used as identities."""

    schema_version: Literal[EVALUATION_METHOD_CONTRACT_VERSION]
    id: str

    _id = field_validator("id")(validate_portable_id)


class SliceRef(StrictModel):
    """A stable cohort label included in the analysis contract and report."""

    schema_version: Literal[EVALUATION_METHOD_CONTRACT_VERSION]
    id: str

    _id = field_validator("id")(validate_portable_id)


class AnalysisPlan(StrictModel):
    """The unit, dependence cluster, slices, curve domain and null policy."""

    schema_version: Literal[EVALUATION_METHOD_CONTRACT_VERSION]
    id: str
    analysis_unit: str
    cluster_unit: str
    slices: tuple[SliceRef, ...]
    curve_domain: Literal["shared_budget", "not_applicable"]
    missingness: Literal["fail_closed"]

    _id = field_validator("id")(validate_portable_id)

    @model_validator(mode="after")
    def unique_slices(self) -> AnalysisPlan:
        ids = tuple(slice_ref.id for slice_ref in self.slices)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("analysis plan slices must be non-empty and unique")
        return self

    @model_validator(mode="after")
    def non_blank_units(self) -> AnalysisPlan:
        if not self.analysis_unit.strip() or not self.cluster_unit.strip():
            raise ValueError("analysis plan units must be non-blank")
        return self


class EvaluationMethodPlugin(StrictModel):
    """Versioned declaration of what a method can honestly execute and grade."""

    schema_version: Literal[EVALUATION_METHOD_CONTRACT_VERSION]
    id: str
    version: Literal[EVALUATION_METHOD_CONTRACT_VERSION]
    status: Literal[
        "native-qualified", "exploratory-import", "data-required", "blocked"
    ]
    execution_owner: Literal["server", "worker", "provider", "benchmark_native"]
    input_schema: str
    export_schema: str
    live_input_complete: StrictBool
    live_grader: StrictBool
    applicable_tracks: tuple[str, ...]
    live_tracks: tuple[str, ...]
    produced_metric_ids: tuple[str, ...]
    evidence_ceiling: Literal["E0", "E1", "E2", "E3", "E4", "E5"]
    native_parity: Literal["native", "source_qualified", "none"]
    required_artifact_ids: tuple[str, ...]
    analysis_plan: AnalysisPlan

    _id = field_validator("id")(validate_portable_id)

    @field_validator("input_schema", "export_schema")
    @classmethod
    def portable_schema_id(cls, value: str) -> str:
        return validate_portable_id(value)

    @model_validator(mode="after")
    def validate_live_claim(self) -> EvaluationMethodPlugin:
        if self.version != EVALUATION_METHOD_CONTRACT_VERSION:
            raise ValueError("method plugin version must match the v2 contract")
        if (
            not self.applicable_tracks
            or len(self.applicable_tracks) != len(set(self.applicable_tracks))
            or len(self.live_tracks) != len(set(self.live_tracks))
            or not self.produced_metric_ids
            or len(self.produced_metric_ids) != len(set(self.produced_metric_ids))
        ):
            raise ValueError(
                "method metric ids must be non-empty and all identities must be unique"
            )
        if any(
            not metric_id or metric_id.strip() != metric_id
            for metric_id in self.produced_metric_ids
        ):
            raise ValueError("method metric ids must be trimmed")
        known_tracks = frozenset(TRACK_IDS)
        if any(track not in known_tracks for track in self.applicable_tracks):
            raise ValueError("method applicable tracks must be known")
        if any(track not in known_tracks for track in self.live_tracks):
            raise ValueError("method live tracks must be known")
        if not set(self.live_tracks).issubset(self.applicable_tracks):
            raise ValueError("method live tracks must be applicable")
        if not self.required_artifact_ids or len(self.required_artifact_ids) != len(
            set(self.required_artifact_ids)
        ):
            raise ValueError("method required artifacts must be non-empty and unique")
        for artifact_id in self.required_artifact_ids:
            validate_portable_id(artifact_id)
        if self.status == "native-qualified" and (
            not self.live_input_complete or not self.live_grader or not self.live_tracks
        ):
            raise ValueError(
                "native-qualified methods require complete live input, grading, and tracks"
            )
        if self.status != "native-qualified" and (
            self.live_input_complete or self.live_grader
        ):
            raise ValueError(
                "non-qualified methods cannot claim complete live execution"
            )
        if (
            self.native_parity == "native"
            and self.execution_owner != "benchmark_native"
        ):
            raise ValueError("native parity requires benchmark-native execution")
        return self


class CompoundModelBudgetOutcome(StrictModel):
    """One graded action outcome at one point on the shared budget domain."""

    case_id: str
    action: ActionRef
    budget: int = Field(gt=0)
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    slice_refs: tuple[SliceRef, ...]

    _case_id = field_validator("case_id")(validate_portable_id)

    @model_validator(mode="after")
    def has_unique_slices(self) -> CompoundModelBudgetOutcome:
        ids = tuple(slice_ref.id for slice_ref in self.slice_refs)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("compound outcomes require unique non-empty slices")
        return self


class SharedDomainCurvePoint(StrictModel):
    action: ActionRef
    budget: int = Field(gt=0)
    mean_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    case_count: int = Field(gt=0)


class CompoundModelBudgetReport(StrictModel):
    """Report payload for R2: identity, analysis plan, curve and missingness."""

    method: EvaluationMethodPlugin
    analysis_plan: AnalysisPlan
    action_refs: tuple[ActionRef, ...]
    slice_refs: tuple[SliceRef, ...]
    raw_shared_domain_curve: tuple[SharedDomainCurvePoint, ...]
    audc: float = Field(ge=0, allow_inf_nan=False)
    nauc: float = Field(ge=0, le=1, allow_inf_nan=False)
    peak: float = Field(ge=0, le=1, allow_inf_nan=False)
    qnc: float = Field(ge=0, le=1, allow_inf_nan=False)
    missing_case_action_budget_cells: int = Field(ge=0)


R2_COMPOUND_MODEL_BUDGET_PLUGIN = EvaluationMethodPlugin(
    schema_version=EVALUATION_METHOD_CONTRACT_VERSION,
    id=COMPOUND_MODEL_BUDGET_METHOD_ID,
    version=EVALUATION_METHOD_CONTRACT_VERSION,
    status="exploratory-import",
    execution_owner="server",
    input_schema="r2-compound-input",
    export_schema="r2-compound-report",
    live_input_complete=False,
    live_grader=False,
    applicable_tracks=("routing", "model_pool", "joint", "capacity"),
    live_tracks=(),
    produced_metric_ids=(
        "r2.compound_model_budget.audc",
        "r2.compound_model_budget.nauc",
        "r2.compound_model_budget.peak",
        "r2.compound_model_budget.qnc",
    ),
    evidence_ceiling="E0",
    native_parity="source_qualified",
    required_artifact_ids=("curves",),
    analysis_plan=AnalysisPlan(
        schema_version=EVALUATION_METHOD_CONTRACT_VERSION,
        id="r2-compound-case-action-budget",
        analysis_unit="case_action_budget",
        cluster_unit="case",
        slices=(
            SliceRef(
                schema_version=EVALUATION_METHOD_CONTRACT_VERSION,
                id="all",
            ),
        ),
        curve_domain="shared_budget",
        missingness="fail_closed",
    ),
)


def _compound_domain(
    rows: tuple[CompoundModelBudgetOutcome, ...],
    method: EvaluationMethodPlugin,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, ...],
    tuple[str, ...],
    dict[tuple[str, str, int], float],
]:
    cases = tuple(sorted({row.case_id for row in rows}))
    actions = tuple(sorted({row.action.id for row in rows}))
    budgets = tuple(sorted({row.budget for row in rows}))
    slices = tuple(
        sorted({slice_ref.id for row in rows for slice_ref in row.slice_refs})
    )
    seen: set[tuple[str, str, int]] = set()
    cells: dict[tuple[str, str, int], float] = {}
    expected_slice_ids = {slice_ref.id for slice_ref in method.analysis_plan.slices}
    for row in rows:
        key = (row.case_id, row.action.id, row.budget)
        if key in seen:
            raise ValueError(
                "duplicate case x action x budget outcome: " + " x ".join(map(str, key))
            )
        seen.add(key)
        cells[key] = row.score
        if {slice_ref.id for slice_ref in row.slice_refs} != expected_slice_ids:
            raise ValueError(
                "compound outcome slices must exactly match the analysis plan"
            )
    expected = len(cases) * len(actions) * len(budgets)
    if len(cells) != expected:
        raise ValueError(
            "compound model+budget outcomes must form an exact shared "
            "case x action x budget domain"
        )
    return cases, actions, budgets, slices, cells


def _compound_curve(
    cases: tuple[str, ...],
    actions: tuple[str, ...],
    budgets: tuple[int, ...],
    cells: dict[tuple[str, str, int], float],
) -> tuple[list[SharedDomainCurvePoint], dict[tuple[str, int], float]]:
    curve: list[SharedDomainCurvePoint] = []
    score_by_action_budget: dict[tuple[str, int], float] = {}
    for action_id in actions:
        for budget in budgets:
            values = [cells[(case_id, action_id, budget)] for case_id in cases]
            mean_score = sum(values) / len(values)
            score_by_action_budget[(action_id, budget)] = mean_score
            curve.append(
                SharedDomainCurvePoint(
                    action=ActionRef(
                        schema_version=EVALUATION_METHOD_CONTRACT_VERSION,
                        id=action_id,
                    ),
                    budget=budget,
                    mean_score=mean_score,
                    case_count=len(cases),
                )
            )
    return curve, score_by_action_budget


def _compound_summary(
    actions: tuple[str, ...],
    budgets: tuple[int, ...],
    score_by_action_budget: dict[tuple[str, int], float],
) -> tuple[float, float, float, float]:
    if len(budgets) < _MIN_SHARED_BUDGET_POINTS:
        raise ValueError("compound AUDC requires at least two shared budget points")
    audc = 0.0
    for action_id in actions:
        for lower, upper in pairwise(budgets):
            audc += (
                (upper - lower)
                * (
                    score_by_action_budget[(action_id, lower)]
                    + score_by_action_budget[(action_id, upper)]
                )
                / 2
            )
    budget_span = budgets[-1] - budgets[0]
    nauc = audc / (len(actions) * budget_span)
    peak = max(score_by_action_budget.values())
    qnc = sum(
        score_by_action_budget[(action_id, budgets[-1])] for action_id in actions
    ) / len(actions)
    if not all(isfinite(value) for value in (audc, nauc, peak, qnc)):
        raise ValueError("compound reduction produced a non-finite metric")
    return audc, nauc, peak, qnc


def reduce_compound_model_budget(
    outcomes: Iterable[CompoundModelBudgetOutcome],
    *,
    method: EvaluationMethodPlugin = R2_COMPOUND_MODEL_BUDGET_PLUGIN,
) -> CompoundModelBudgetReport:
    """Reduce R2 with exact rectangular cardinality and a shared raw domain.

    AUDC is the sum of trapezoids across each action's score/budget curve.
    nAUC divides AUDC by the common budget span and action count. QNC is the
    mean score at the largest common budget; it intentionally never imputes a
    missing arm, case, or budget cell.
    """

    if method.id != COMPOUND_MODEL_BUDGET_METHOD_ID:
        raise ValueError("compound reducer requires the R2 compound method plugin")
    rows = tuple(outcomes)
    if not rows:
        raise ValueError("compound reducer requires outcomes")
    cases, actions, budgets, slices, cells = _compound_domain(rows, method)
    curve, score_by_action_budget = _compound_curve(cases, actions, budgets, cells)
    audc, nauc, peak, qnc = _compound_summary(actions, budgets, score_by_action_budget)
    return CompoundModelBudgetReport(
        method=method,
        analysis_plan=method.analysis_plan,
        action_refs=tuple(
            ActionRef(
                schema_version=EVALUATION_METHOD_CONTRACT_VERSION,
                id=action_id,
            )
            for action_id in actions
        ),
        slice_refs=tuple(
            SliceRef(
                schema_version=EVALUATION_METHOD_CONTRACT_VERSION,
                id=slice_id,
            )
            for slice_id in slices
        ),
        raw_shared_domain_curve=tuple(curve),
        audc=audc,
        nauc=nauc,
        peak=peak,
        qnc=qnc,
        missing_case_action_budget_cells=0,
    )
