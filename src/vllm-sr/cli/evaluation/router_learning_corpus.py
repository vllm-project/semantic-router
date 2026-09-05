"""Versioned deterministic Router Learning replay corpus."""

from __future__ import annotations

from importlib.resources import files
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cli.evaluation.canonical import strict_json_loads
from cli.evaluation.contract_primitives import Message, StrictModel
from cli.evaluation.contract_validation import validate_portable_id
from cli.evaluation.contracts import (
    CaseGrading,
    CaseVisible,
    GradingCaseSet,
    VisibleCaseSet,
)
from cli.evaluation.evidence import (
    ArmEvidence,
    CapacityEvidence,
    FixtureCaseEvidence,
    MultimodalEvidence,
    PreferenceEvidence,
    ReplayFixture,
    RouteEvidence,
    SafetyEvidence,
    TrajectoryEvidence,
)

ROUTER_LEARNING_CASE_COUNT = 12
_CANDIDATE_ARM_COUNT = 2

_RESOURCE_PATH = ("resources", "router_learning_core.v1.json")
_SCHEMA_VERSION = "router-learning-corpus.v1"
_COLUMNS = (
    "id",
    "prompt",
    "domain",
    "eligible_arm_ids",
    "protected_arm_id",
    "feedback_delay_rounds",
    "feedback_observed",
    "fast_success",
    "fast_quality",
    "fast_latency_ms",
    "fast_cost_usd",
    "fast_call_count",
    "strong_success",
    "strong_quality",
    "strong_latency_ms",
    "strong_cost_usd",
    "strong_call_count",
)


class RouterLearningArm(StrictModel):
    id: str
    quality_seed: float = Field(ge=0, le=1)
    input_cost_per_million_tokens_usd: float = Field(ge=0)
    output_cost_per_million_tokens_usd: float = Field(ge=0)

    _id = field_validator("id")(validate_portable_id)


class RouterLearningOutcome(StrictModel):
    success: bool
    quality: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    cost_usd: float = Field(ge=0)
    call_count: int = Field(ge=1)


class RouterLearningCase(StrictModel):
    id: str
    prompt: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    eligible_arm_ids: tuple[str, ...] = Field(min_length=1)
    protected_arm_id: str | None
    feedback_delay_rounds: int = Field(ge=0, le=16)
    feedback_observed: bool
    outcomes: dict[str, RouterLearningOutcome]

    _id = field_validator("id")(validate_portable_id)


class RouterLearningCorpus(StrictModel):
    schema_version: Literal["router-learning-corpus.v1"]
    candidate_arms: tuple[RouterLearningArm, ...] = Field(min_length=2)
    base_arm_id: str
    trial_seeds: tuple[int, ...] = Field(min_length=2)
    propensity_status: Literal["unsupported"]
    cases: tuple[RouterLearningCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_relations(self) -> RouterLearningCorpus:
        arm_ids = tuple(arm.id for arm in self.candidate_arms)
        case_ids = tuple(case.id for case in self.cases)
        if (
            len(arm_ids) != _CANDIDATE_ARM_COUNT
            or len(arm_ids) != len(set(arm_ids))
            or self.base_arm_id not in arm_ids
        ):
            raise ValueError("Router Learning candidate arms are invalid")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Router Learning case IDs must be unique")
        if len(self.trial_seeds) != len(set(self.trial_seeds)) or any(
            seed < 0 or seed > 2**32 - 1 for seed in self.trial_seeds
        ):
            raise ValueError("Router Learning trial seeds must be unique uint32 values")
        for case in self.cases:
            if not set(case.eligible_arm_ids).issubset(arm_ids):
                raise ValueError(
                    "Router Learning eligibility references an unknown arm"
                )
            if set(case.outcomes) != set(arm_ids):
                raise ValueError(
                    "Router Learning outcomes must cover every candidate arm"
                )
            if case.protected_arm_id is not None and (
                case.protected_arm_id not in case.eligible_arm_ids
            ):
                raise ValueError("Router Learning protected arm must be eligible")
        return self


def _load_corpus() -> RouterLearningCorpus:
    raw = strict_json_loads(
        files("cli.evaluation").joinpath(*_RESOURCE_PATH).read_bytes()
    )
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "candidate_arms",
        "base_arm_id",
        "trial_seeds",
        "propensity_status",
        "columns",
        "cases",
    }:
        raise RuntimeError(
            "Router Learning corpus must use the versioned object schema"
        )
    if raw["schema_version"] != _SCHEMA_VERSION or tuple(raw["columns"]) != _COLUMNS:
        raise RuntimeError("Router Learning corpus schema is unsupported")
    arm_ids = tuple(arm["id"] for arm in raw["candidate_arms"])
    cases: list[dict[str, object]] = []
    for index, row in enumerate(raw["cases"]):
        if not isinstance(row, list) or len(row) != len(_COLUMNS):
            raise RuntimeError(f"Router Learning case row {index} is malformed")
        values = dict(zip(_COLUMNS, row, strict=True))
        outcomes = {
            arm_ids[0]: {
                "success": values.pop("fast_success"),
                "quality": values.pop("fast_quality"),
                "latency_ms": values.pop("fast_latency_ms"),
                "cost_usd": values.pop("fast_cost_usd"),
                "call_count": values.pop("fast_call_count"),
            },
            arm_ids[1]: {
                "success": values.pop("strong_success"),
                "quality": values.pop("strong_quality"),
                "latency_ms": values.pop("strong_latency_ms"),
                "cost_usd": values.pop("strong_cost_usd"),
                "call_count": values.pop("strong_call_count"),
            },
        }
        cases.append({**values, "outcomes": outcomes})
    corpus = RouterLearningCorpus.model_validate(
        {key: value for key, value in raw.items() if key not in {"columns", "cases"}}
        | {"cases": cases}
    )
    if len(corpus.cases) != ROUTER_LEARNING_CASE_COUNT:
        raise RuntimeError("Router Learning corpus case count drifted")
    return corpus


ROUTER_LEARNING_CORPUS = _load_corpus()


def router_learning_case_sets(
    cases: tuple[RouterLearningCase, ...] | None = None,
) -> tuple[VisibleCaseSet, GradingCaseSet, ReplayFixture]:
    corpus = ROUTER_LEARNING_CORPUS
    selected = corpus.cases if cases is None else cases
    visible = VisibleCaseSet(
        cases=tuple(
            CaseVisible(
                id=case.id,
                track_ids=("joint",),
                messages=(Message(role="user", content=case.prompt),),
                tags=(
                    "router-learning-core",
                    f"domain:{case.domain}",
                    f"feedback-delay:{case.feedback_delay_rounds}",
                    f"feedback-observed:{str(case.feedback_observed).lower()}",
                    *(f"eligible:{arm_id}" for arm_id in case.eligible_arm_ids),
                    *(
                        (f"protected:{case.protected_arm_id}",)
                        if case.protected_arm_id
                        else ()
                    ),
                ),
            )
            for case in selected
        )
    )
    grading = GradingCaseSet(
        cases=tuple(
            CaseGrading(
                case_id=case.id,
                expected_route=max(
                    case.eligible_arm_ids,
                    key=lambda arm_id: case.outcomes[arm_id].quality,
                ),
                preferred_arm_id=max(
                    case.eligible_arm_ids,
                    key=lambda arm_id: case.outcomes[arm_id].quality,
                ),
                should_block=case.protected_arm_id is not None,
            )
            for case in selected
        )
    )
    fixture = ReplayFixture(
        cases=tuple(
            FixtureCaseEvidence(
                case_id=case.id,
                route=RouteEvidence(
                    selected_model=corpus.base_arm_id,
                    selection_status="selected",
                    success=case.outcomes[corpus.base_arm_id].success,
                    latency_ms=case.outcomes[corpus.base_arm_id].latency_ms,
                ),
                arms=tuple(
                    ArmEvidence(
                        arm_id=arm.id,
                        success=case.outcomes[arm.id].success,
                        quality=case.outcomes[arm.id].quality,
                        latency_ms=case.outcomes[arm.id].latency_ms,
                        runtime_cost=case.outcomes[arm.id].cost_usd,
                    )
                    for arm in corpus.candidate_arms
                ),
                trajectory=TrajectoryEvidence(
                    success=True, task_score=1.0, steps=1, tool_calls=0
                ),
                multimodal=MultimodalEvidence(supported=True, quality=1.0),
                preference=PreferenceEvidence(
                    chosen_arm_id=corpus.base_arm_id,
                    preferred_arm_id=max(
                        case.eligible_arm_ids,
                        key=lambda arm_id: case.outcomes[arm_id].quality,
                    ),
                ),
                safety=SafetyEvidence(
                    violations=0,
                    should_block=case.protected_arm_id is not None,
                    blocked=case.protected_arm_id is not None,
                ),
                capacity=CapacityEvidence(
                    concurrency=1,
                    success=True,
                    latency_ms=case.outcomes[corpus.base_arm_id].latency_ms,
                    throughput_rps=1.0,
                ),
            )
            for case in selected
        )
    )
    return visible, grading, fixture
