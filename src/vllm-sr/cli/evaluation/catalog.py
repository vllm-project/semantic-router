"""Evaluation catalog assembly for the CLI and Dashboard."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Literal

from pydantic import field_serializer, model_validator

from cli.evaluation.catalog_suites import (
    CatalogSuite as _CatalogSuite,
)
from cli.evaluation.catalog_suites import (
    configured_builtin_suites,
    validate_installed_suites,
)
from cli.evaluation.catalog_tracks import CATALOG_TRACKS, CatalogTrack
from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.execution_contract import (
    LIVE_RUNTIME_EXECUTOR_ID,
    NORMALIZED_LIVE_EXECUTOR_ID,
)
from cli.evaluation.executor_contracts import (
    BUILTIN_EXECUTOR_CONTRACTS,
    ExecutorContract,
)
from cli.evaluation.gate_contract import (
    CHANGE_PROFILE_DEFINITIONS,
    GATE_CONTRACT_VERSION,
    ChangeProfile,
    GateDisposition,
    gate_applicability,
)
from cli.evaluation.reporting import EvidenceLevel, TrackID
from cli.evaluation.target_capabilities import (
    DEFAULT_TARGET_REGISTRY,
    TargetContract,
    TargetRegistry,
    mixture_target_contract,
)
from cli.evaluation.target_contracts import (
    CatalogMixture,
    EvaluationTarget,
    HTTPServiceEndpoint,
    ManifestMixture,
)


class CatalogTarget(StrictModel):
    id: str
    name: str
    description: str
    kind: str
    track_ids: tuple[TrackID, ...]
    modes: tuple[Literal["replay", "live"], ...]
    accepted_executors: Mapping[Literal["replay", "live"], tuple[str, ...]]
    evidence_level: EvidenceLevel | None = None
    healthy: bool | None = None
    labels: dict[str, str] | None = None
    mixture: CatalogMixture | None = None

    @model_validator(mode="after")
    def executors_exactly_cover_modes(self) -> CatalogTarget:
        canonical_modes = tuple(
            mode for mode in ("replay", "live") if mode in self.modes
        )
        if self.modes != canonical_modes or len(set(self.modes)) != len(self.modes):
            raise ValueError("target modes must use canonical replay/live order")
        if set(self.accepted_executors) != set(self.modes):
            raise ValueError("target executors must exactly cover declared modes")
        frozen: dict[Literal["replay", "live"], tuple[str, ...]] = {}
        for mode in self.modes:
            executors = tuple(self.accepted_executors[mode])
            if (
                not executors
                or len(executors) != len(set(executors))
                or any(
                    re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", executor) is None
                    for executor in executors
                )
            ):
                raise ValueError(
                    "target executor identities must be portable and unique"
                )
            frozen[mode] = executors
        object.__setattr__(self, "accepted_executors", MappingProxyType(frozen))
        return self

    @field_serializer("accepted_executors")
    def serialize_accepted_executors(
        self,
        value: Mapping[Literal["replay", "live"], tuple[str, ...]],
    ) -> dict[str, list[str]]:
        return {mode: list(value[mode]) for mode in self.modes}


CampaignBindingKind = Literal["run", "controlled_pair", "fidelity_pair"]


class CatalogCampaignSlot(StrictModel):
    """One catalog-owned Campaign evidence binding for a release gate."""

    gate_id: Literal["G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"]
    name: str
    description: str
    disposition: GateDisposition
    binding_kind: CampaignBindingKind
    track_id: TrackID
    mode: Literal["replay", "live"] | None = None
    minimum_evidence_level: EvidenceLevel
    accepted_executor_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_slot(self) -> CatalogCampaignSlot:
        if (
            not self.name
            or self.name.strip() != self.name
            or not self.description
            or self.description.strip() != self.description
            or not self.accepted_executor_ids
            or len(self.accepted_executor_ids) != len(set(self.accepted_executor_ids))
            or any(
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", executor_id) is None
                for executor_id in self.accepted_executor_ids
            )
        ):
            raise ValueError("campaign slot identity is invalid")
        if self.binding_kind == "fidelity_pair":
            if self.gate_id != "G5" or self.mode != "live":
                raise ValueError("fidelity pair is reserved for live G5")
        elif self.mode is None:
            raise ValueError("run and controlled-pair slots require an exact mode")
        if self.binding_kind == "controlled_pair" and self.gate_id != "G3":
            raise ValueError("controlled pair is reserved for G3")
        return self


class CatalogChangeProfile(StrictModel):
    id: ChangeProfile
    name: str
    description: str
    campaign_slots: tuple[CatalogCampaignSlot, ...]

    @model_validator(mode="after")
    def validate_campaign_slots(self) -> CatalogChangeProfile:
        expected_gate_ids = tuple(f"G{index}" for index in range(2, 10))
        if tuple(slot.gate_id for slot in self.campaign_slots) != expected_gate_ids:
            raise ValueError(
                "campaign slots must exactly cover G2-G9 in canonical order"
            )
        expected_dispositions = {
            definition.id: disposition
            for definition, disposition in gate_applicability(self.id)
            if definition.id in expected_gate_ids
        }
        if any(
            slot.disposition != expected_dispositions[slot.gate_id]
            for slot in self.campaign_slots
        ):
            raise ValueError(
                "campaign slot disposition must match the release gate matrix"
            )
        return self


class EvaluationCatalog(StrictModel):
    schema_version: Literal[SCHEMA_VERSION]
    generated_at: datetime | None = None
    gate_contract_version: Literal[GATE_CONTRACT_VERSION] = GATE_CONTRACT_VERSION
    change_profiles: tuple[CatalogChangeProfile, ...]
    tracks: tuple[CatalogTrack, ...]
    suites: tuple[_CatalogSuite, ...]
    targets: tuple[CatalogTarget, ...]


_CAMPAIGN_SLOT_TEMPLATES: tuple[dict[str, object], ...] = (
    {
        "gate_id": "G2",
        "name": "Policy enforcement",
        "description": "Checks that required safety and routing policies are enforced on the proposed system.",
        "binding_kind": "run",
        "track_id": "safety",
        "mode": "live",
        "minimum_evidence_level": "E3",
        "accepted_executor_ids": (LIVE_RUNTIME_EXECUTOR_ID,),
    },
    {
        "gate_id": "G3",
        "name": "Controlled value comparison",
        "description": "Compares baseline and candidate outcomes on the same live cases with balanced execution order.",
        "binding_kind": "controlled_pair",
        "track_id": "joint",
        "mode": "live",
        "minimum_evidence_level": "E4",
        "accepted_executor_ids": (LIVE_RUNTIME_EXECUTOR_ID,),
    },
    {
        "gate_id": "G4",
        "name": "Workload-shift robustness",
        "description": "Measures quality and reliability under the workload changes declared for this release.",
        "binding_kind": "run",
        "track_id": "routing",
        "mode": "live",
        "minimum_evidence_level": "E4",
        "accepted_executor_ids": (NORMALIZED_LIVE_EXECUTOR_ID,),
    },
    {
        "gate_id": "G5",
        "name": "Live consistency",
        "description": "Checks that a fresh live run agrees with the saved candidate on the same evaluation cases.",
        "binding_kind": "fidelity_pair",
        "track_id": "joint",
        "mode": "live",
        "minimum_evidence_level": "E5",
        "accepted_executor_ids": (LIVE_RUNTIME_EXECUTOR_ID,),
    },
    {
        "gate_id": "G6",
        "name": "Fault recovery",
        "description": "Measures fallback, retry, state continuity, and side effects during injected failures.",
        "binding_kind": "run",
        "track_id": "agentic",
        "mode": "live",
        "minimum_evidence_level": "E5",
        "accepted_executor_ids": (LIVE_RUNTIME_EXECUTOR_ID,),
    },
    {
        "gate_id": "G7",
        "name": "Cost, latency, and capacity",
        "description": "Measures whether the proposed system meets its service objectives under repeated load.",
        "binding_kind": "run",
        "track_id": "capacity",
        "mode": "live",
        "minimum_evidence_level": "E5",
        "accepted_executor_ids": (LIVE_RUNTIME_EXECUTOR_ID,),
    },
    {
        "gate_id": "G8",
        "name": "Canary safety",
        "description": "Checks production assignment, exposure limits, stop conditions, and rollback controls.",
        "binding_kind": "run",
        "track_id": "preference",
        "mode": "live",
        "minimum_evidence_level": "E5",
        "accepted_executor_ids": (LIVE_RUNTIME_EXECUTOR_ID,),
    },
    {
        "gate_id": "G9",
        "name": "Online preference",
        "description": "Measures assigned user-preference outcomes for the baseline and proposed system.",
        "binding_kind": "run",
        "track_id": "preference",
        "mode": "live",
        "minimum_evidence_level": "E5",
        "accepted_executor_ids": (LIVE_RUNTIME_EXECUTOR_ID,),
    },
)


def _campaign_slots(profile: ChangeProfile) -> tuple[CatalogCampaignSlot, ...]:
    dispositions = {
        definition.id: disposition
        for definition, disposition in gate_applicability(profile)
    }
    slots: list[CatalogCampaignSlot] = []
    for template in _CAMPAIGN_SLOT_TEMPLATES:
        values = {
            **template,
            "disposition": dispositions[str(template["gate_id"])],
        }
        if profile == "agent_multimodal" and template["gate_id"] == "G5":
            values.update(
                {
                    "description": (
                        "Checks that a fresh multimodal run agrees with the saved "
                        "candidate on the same evaluation cases."
                    ),
                    "track_id": "multimodal",
                    "minimum_evidence_level": "E4",
                    "accepted_executor_ids": (NORMALIZED_LIVE_EXECUTOR_ID,),
                }
            )
        slots.append(CatalogCampaignSlot.model_validate(values))
    return tuple(slots)


def _configured_target(
    contract: TargetContract,
    *,
    router_api_url: str | None,
    envoy_url: str | None,
    agent_task_ledger: HTTPServiceEndpoint | None,
    fault_recovery_ledger: HTTPServiceEndpoint | None,
    hard_policy_ledger: HTTPServiceEndpoint | None,
    production_experiment_ledger: HTTPServiceEndpoint | None,
    mixture: ManifestMixture | None,
    backend_topology_digest: str | None,
) -> EvaluationTarget:
    brokered = contract.execution_profile == "brokered-runtime"
    return EvaluationTarget(
        id=contract.id,
        kind=contract.kind,
        router_api_url=router_api_url if brokered else None,
        envoy_url=envoy_url if brokered else None,
        agent_task_ledger=agent_task_ledger if brokered else None,
        fault_recovery_ledger=fault_recovery_ledger if brokered else None,
        hard_policy_ledger=hard_policy_ledger if brokered else None,
        production_experiment_ledger=(
            production_experiment_ledger if brokered else None
        ),
        mixture=mixture if brokered else None,
        backend_topology_digest=backend_topology_digest if brokered else None,
    )


def _catalog_target(
    contract: TargetContract,
    configured: EvaluationTarget,
    installed_suite_count: int,
    *,
    mixture: CatalogMixture | None = None,
) -> CatalogTarget:
    return CatalogTarget(
        id=contract.id,
        name=contract.name,
        description=contract.description,
        kind=contract.kind,
        track_ids=contract.available_tracks(configured),
        modes=contract.modes,
        accepted_executors=contract.accepted_executors,
        evidence_level=contract.evidence_level,
        healthy=contract.healthy(configured, installed_suite_count),
        labels=(dict(contract.labels) if contract.labels is not None else None),
        mixture=mixture,
    )


def get_catalog(
    *,
    generated_at: bool = True,
    router_api_url: str | None = None,
    envoy_url: str | None = None,
    agent_task_ledger: HTTPServiceEndpoint | None = None,
    fault_recovery_ledger: HTTPServiceEndpoint | None = None,
    hard_policy_ledger: HTTPServiceEndpoint | None = None,
    production_experiment_ledger: HTTPServiceEndpoint | None = None,
    mixture: ManifestMixture | None = None,
    backend_topology_digest: str | None = None,
    installed_suites: tuple[_CatalogSuite, ...] = (),
    executor_contracts: tuple[ExecutorContract, ...] = BUILTIN_EXECUTOR_CONTRACTS,
    target_registry: TargetRegistry = DEFAULT_TARGET_REGISTRY,
) -> EvaluationCatalog:
    validate_installed_suites(installed_suites, executor_contracts)
    targets: list[CatalogTarget] = []
    for contract in target_registry.contracts:
        configured = _configured_target(
            contract,
            router_api_url=router_api_url,
            envoy_url=envoy_url,
            agent_task_ledger=agent_task_ledger,
            fault_recovery_ledger=fault_recovery_ledger,
            hard_policy_ledger=hard_policy_ledger,
            production_experiment_ledger=production_experiment_ledger,
            mixture=mixture,
            backend_topology_digest=backend_topology_digest,
        )
        targets.append(_catalog_target(contract, configured, len(installed_suites)))
    if mixture is not None:
        contract = mixture_target_contract(mixture)
        configured = _configured_target(
            contract,
            router_api_url=router_api_url,
            envoy_url=envoy_url,
            agent_task_ledger=agent_task_ledger,
            fault_recovery_ledger=fault_recovery_ledger,
            hard_policy_ledger=hard_policy_ledger,
            production_experiment_ledger=production_experiment_ledger,
            mixture=mixture,
            backend_topology_digest=backend_topology_digest,
        )
        targets.append(
            _catalog_target(
                contract,
                configured,
                len(installed_suites),
                mixture=mixture.public_summary(),
            )
        )
    builtin_suites = configured_builtin_suites(
        agent_task_ledger is not None,
        fault_recovery_ledger is not None,
        hard_policy_ledger is not None,
        production_experiment_ledger is not None,
    )
    return EvaluationCatalog(
        schema_version=SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc) if generated_at else None,
        change_profiles=tuple(
            CatalogChangeProfile(
                id=profile.id,
                name=profile.name,
                description=profile.description,
                campaign_slots=_campaign_slots(profile.id),
            )
            for profile in CHANGE_PROFILE_DEFINITIONS
        ),
        tracks=CATALOG_TRACKS,
        suites=(*builtin_suites, *installed_suites),
        targets=tuple(targets),
    )
