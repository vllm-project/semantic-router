"""Built-in and installed evaluation-suite catalog contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import (
    SerializerFunctionWrapHandler,
    field_serializer,
    model_serializer,
    model_validator,
)

from cli.evaluation.campaign_protocol import (
    CampaignProtocol,
    validate_campaign_protocol,
)
from cli.evaluation.catalog_tracks import (
    CATALOG_TRACKS,
    CatalogMethod,
    CatalogMethodEvidenceSource,
)
from cli.evaluation.constants import BUILTIN_SUITE_IDS
from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.execution_contract import (
    FIXTURE_REPLAY_EXECUTOR_ID,
    LIVE_RUNTIME_EXECUTOR_ID,
    MOM_REPLAY_EXECUTOR_ID,
)
from cli.evaluation.executor_contracts import ExecutorContract
from cli.evaluation.live_mom_cases import LIVE_MOM_CASE_COUNT
from cli.evaluation.reporting import EvidenceLevel, TrackID

_ALL_TRACK_IDS = tuple(track.id for track in CATALOG_TRACKS)


class CatalogSuite(StrictModel):
    id: str
    name: str
    description: str
    track_ids: tuple[TrackID, ...]
    modes: tuple[Literal["replay", "live"], ...]
    evidence_level: EvidenceLevel
    executors: Mapping[Literal["replay", "live"], str]
    case_count: int | None = None
    campaign_protocol: CampaignProtocol | None = None
    revision: str | None = None
    tags: tuple[str, ...] = ()
    methods: tuple[CatalogMethod, ...]

    @model_validator(mode="after")
    def modes_have_one_executor(self) -> CatalogSuite:
        canonical_modes = tuple(
            mode for mode in ("replay", "live") if mode in self.modes
        )
        if self.modes != canonical_modes or len(set(self.modes)) != len(self.modes):
            raise ValueError("suite modes must use canonical replay/live order")
        if set(self.executors) != set(self.modes):
            raise ValueError("suite executors must exactly cover declared modes")
        if any(not executor_id for executor_id in self.executors.values()):
            raise ValueError("suite executor identities cannot be empty")
        method_ids = [method.id for method in self.methods]
        method_tracks = {method.track_id for method in self.methods}
        if (
            not self.methods
            or len(method_ids) != len(set(method_ids))
            or method_tracks != set(self.track_ids)
        ):
            raise ValueError(
                "suite methods must uniquely and exactly cover declared tracks"
            )
        if (
            any(
                method.evidence_source is CatalogMethodEvidenceSource.NORMALIZED_IMPORT
                for method in self.methods
            )
            and self.evidence_level != "E0"
        ):
            raise ValueError("normalized import suites must remain E0")
        validate_campaign_protocol(
            self.campaign_protocol,
            modes=self.modes,
            executors=self.executors,
            track_ids=self.track_ids,
            evidence_level=self.evidence_level,
            case_count=self.case_count,
        )
        object.__setattr__(self, "executors", MappingProxyType(dict(self.executors)))
        return self

    @field_serializer("executors")
    def serialize_executors(
        self, value: Mapping[Literal["replay", "live"], str]
    ) -> dict[str, str]:
        return dict(value)

    @model_serializer(mode="wrap")
    def serialize_catalog_suite(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        serialized = handler(self)
        if self.campaign_protocol is None:
            serialized.pop("campaign_protocol", None)
        return serialized


def _method(
    method_id: str,
    track_id: TrackID,
    *,
    gate_ids: tuple[str, ...] = (),
    evidence_source: CatalogMethodEvidenceSource = (
        CatalogMethodEvidenceSource.LIVE_RUNTIME
    ),
    status: Literal["qualified", "configured", "data_required"] = "configured",
    reason: str | None = None,
) -> CatalogMethod:
    return CatalogMethod(
        id=method_id,
        track_id=track_id,
        qualified_gate_ids=gate_ids,
        evidence_source=evidence_source,
        status=status,
        reason=reason,
    )


_BUILTIN_SUITES = (
    CatalogSuite(
        id="evaluation-smoke",
        name="Evaluation setup check",
        description=(
            "A small deterministic workload that verifies every evaluation area is "
            "connected and reportable."
        ),
        track_ids=_ALL_TRACK_IDS,
        modes=("replay",),
        evidence_level="E0",
        executors={"replay": FIXTURE_REPLAY_EXECUTOR_ID},
        case_count=4,
        revision="builtin-v1",
        tags=("smoke", "deterministic"),
        methods=tuple(
            _method(
                f"fixture.{track_id}.v1",
                track_id,
                evidence_source=CatalogMethodEvidenceSource.DIAGNOSTIC_FIXTURE,
            )
            for track_id in _ALL_TRACK_IDS
        ),
    ),
    CatalogSuite(
        id="live-mom-core",
        name="Routing and model-pool setup check",
        description=(
            "A small hidden-answer workload for diagnosing routing, model-pool, and "
            "end-to-end execution. It is not large or diverse enough to support a "
            "release comparison."
        ),
        track_ids=("routing", "model_pool", "joint"),
        modes=("replay", "live"),
        evidence_level="E0",
        executors={
            "replay": MOM_REPLAY_EXECUTOR_ID,
            "live": LIVE_RUNTIME_EXECUTOR_ID,
        },
        case_count=LIVE_MOM_CASE_COUNT,
        revision="mom-diagnostic-cohort-v2",
        tags=("smoke", "mom", "hidden-label", "diagnostic-only"),
        methods=(
            _method("routing.live-diagnostic.v1", "routing"),
            _method("model-pool.live-dense.v1", "model_pool"),
            _method("joint.live-routed-outcome.v1", "joint"),
        ),
    ),
    CatalogSuite(
        id="live-agent-tasks",
        name="Agent task evaluation",
        description=(
            "Repeated tool-use and reasoning tasks with complete provider results. "
            "Measures task completion, tool-policy compliance, reliability, latency, "
            "and cost; it does not invoke tools itself or claim parity with external "
            "agent benchmarks."
        ),
        track_ids=("agentic",),
        modes=("live",),
        evidence_level="E5",
        executors={"live": LIVE_RUNTIME_EXECUTOR_ID},
        revision="executor-v1",
        methods=(
            _method(
                "live-agent-task.v1",
                "agentic",
                status="data_required",
                reason=(
                    "Connect a managed agent-task results source that includes every "
                    "repeated attempt, the required tool policy for each task, and "
                    "provider-confirmed outcomes for the selected Mixture."
                ),
            ),
        ),
    ),
    CatalogSuite(
        id="live-fault-recovery",
        name="Agent fault-recovery evaluation",
        description=(
            "Matched baseline and injected-failure tasks that measure recovery, state "
            "continuity, side effects, retries, and latency."
        ),
        track_ids=("agentic",),
        modes=("live",),
        evidence_level="E5",
        executors={"live": LIVE_RUNTIME_EXECUTOR_ID},
        revision="executor-v1",
        methods=(
            _method(
                "live-fault-recovery.v1",
                "agentic",
                gate_ids=("G6",),
                status="data_required",
                reason=(
                    "Connect a managed fault-recovery results source with complete "
                    "matched baseline and injected-failure attempts at the same task step."
                ),
            ),
        ),
    ),
    CatalogSuite(
        id="live-multimodal",
        name="Multimodal response evaluation",
        description=(
            "Text and non-text requests graded for supported input handling, response "
            "quality, reliability, and latency."
        ),
        track_ids=("multimodal",),
        modes=("live",),
        evidence_level="E0",
        executors={"live": LIVE_RUNTIME_EXECUTOR_ID},
        revision="executor-v1",
        methods=(_method("multimodal.live-chat.v1", "multimodal"),),
    ),
    CatalogSuite(
        id="live-hard-policy",
        name="Policy enforcement evaluation",
        description=(
            "Live policy and adversarial cases that verify required rules are enforced "
            "by the selected system configuration."
        ),
        track_ids=("safety",),
        modes=("live",),
        evidence_level="E4",
        executors={"live": LIVE_RUNTIME_EXECUTOR_ID},
        revision="executor-v1",
        methods=(
            _method(
                "policy.hard-enforcement.v1",
                "safety",
                gate_ids=("G2",),
                status="data_required",
                reason=(
                    "Connect managed policy-test results with the evaluated rules, "
                    "enforcement points, and complete outcomes for the selected configuration."
                ),
            ),
        ),
    ),
    CatalogSuite(
        id="live-production-experiment",
        name="Guarded production evaluation",
        description=(
            "Evaluates connected production experiment results for assignment balance, "
            "exposure controls, risk, stop conditions, rollback readiness, "
            "and preference lift between baseline and candidate."
        ),
        track_ids=("preference",),
        modes=("live",),
        evidence_level="E5",
        executors={"live": LIVE_RUNTIME_EXECUTOR_ID},
        revision="executor-v1",
        methods=(
            _method(
                "production.experiment-controls.v1",
                "preference",
                gate_ids=("G8",),
                evidence_source=CatalogMethodEvidenceSource.LIVE_PRODUCTION,
                status="data_required",
                reason=(
                    "Connect managed production experiment results with complete "
                    "assignment, exposure, and safety-control data."
                ),
            ),
            _method(
                "production.preference-lift.v1",
                "preference",
                gate_ids=("G9",),
                evidence_source=CatalogMethodEvidenceSource.LIVE_PRODUCTION,
                status="data_required",
                reason=(
                    "Connect managed production experiment results with complete "
                    "preference outcomes and the recorded assignment probability for "
                    "each policy."
                ),
            ),
        ),
    ),
    CatalogSuite(
        id="live-capacity",
        name="Capacity setup check",
        description=(
            "A short repeated closed-loop workload for checking load execution, "
            "telemetry, and report generation. It is diagnostic and does not support "
            "a release capacity decision."
        ),
        track_ids=("capacity",),
        modes=("live",),
        evidence_level="E0",
        executors={"live": LIVE_RUNTIME_EXECUTOR_ID},
        revision="executor-v1",
        tags=("smoke", "diagnostic-only"),
        methods=(
            _method(
                "capacity.slo-envelope.v1",
                "capacity",
            ),
        ),
    ),
)

if tuple(suite.id for suite in _BUILTIN_SUITES) != BUILTIN_SUITE_IDS:
    raise RuntimeError(
        "built-in suite catalog order differs from the manifest contract"
    )


def validate_installed_suites(
    installed_suites: tuple[CatalogSuite, ...],
    executor_contracts: tuple[ExecutorContract, ...],
) -> None:
    installed_ids = tuple(suite.id for suite in installed_suites)
    if installed_ids != tuple(sorted(installed_ids)):
        raise ValueError("installed suites must use lexical catalog order")
    if len(installed_ids) != len(set(installed_ids)):
        raise ValueError("installed suite catalog ids must be unique")
    if set(installed_ids).intersection(BUILTIN_SUITE_IDS):
        raise ValueError("installed suite ids cannot shadow built-in suites")
    executor_by_id = {contract.id: contract for contract in executor_contracts}
    for suite in installed_suites:
        expected_modes = ("replay", "live") if "live" in suite.modes else ("replay",)
        if (
            suite.modes != expected_modes
            or suite.case_count is None
            or suite.case_count <= 0
            or suite.revision is None
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", suite.revision)
        ):
            raise ValueError("installed suite catalog entry is not executable")
        for mode in suite.modes:
            executor = executor_by_id.get(suite.executors[mode])
            if (
                executor is None
                or executor.mode != mode
                or not executor.normalized_suite
                or (mode == "replay") != executor.recorded_normalized_import
            ):
                raise ValueError("installed suite catalog executor is not admitted")


def configured_builtin_suites(
    agent_task_ledger_configured: bool,
    fault_recovery_ledger_configured: bool,
    hard_policy_ledger_configured: bool,
    production_experiment_ledger_configured: bool,
) -> tuple[CatalogSuite, ...]:
    configured_ledgers = {
        "live-agent-task.v1": agent_task_ledger_configured,
        "live-fault-recovery.v1": fault_recovery_ledger_configured,
        "policy.hard-enforcement.v1": hard_policy_ledger_configured,
        "production.experiment-controls.v1": production_experiment_ledger_configured,
        "production.preference-lift.v1": production_experiment_ledger_configured,
    }
    return tuple(
        suite.model_copy(
            update={
                "methods": tuple(
                    (
                        method.model_copy(
                            update={"status": "configured", "reason": None}
                        )
                        if configured_ledgers.get(method.id, False)
                        else method
                    )
                    for method in suite.methods
                )
            }
        )
        for suite in _BUILTIN_SUITES
    )
