"""Immutable capabilities attached to evaluation executor identities."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from cli.evaluation.builtin_evidence_qualifications import (
    LIVE_RUNTIME_EVIDENCE_QUALIFICATIONS,
    NORMALIZED_LIVE_EVIDENCE_QUALIFICATIONS,
)
from cli.evaluation.constants import TRACK_IDS
from cli.evaluation.evidence_qualification import (
    EMPTY_EVIDENCE_QUALIFICATIONS,
    EvidenceQualificationRegistry,
)
from cli.evaluation.execution_contract import (
    FIXTURE_REPLAY_EXECUTOR_ID,
    LIVE_RUNTIME_EXECUTOR_ID,
    MOM_REPLAY_EXECUTOR_ID,
    NORMALIZED_LIVE_EXECUTOR_ID,
    NORMALIZED_REPLAY_EXECUTOR_ID,
)
from cli.evaluation.reporting import EvidenceLevel

Mode = Literal["replay", "live"]
TargetProfile = Literal["recorded-source", "brokered-runtime"]
LineageProfile = Literal["fixture-replay", "normalized-suite-replay", "runtime"]

_PORTABLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EVIDENCE_LEVELS: tuple[EvidenceLevel, ...] = (
    "E0",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
)


@dataclass(frozen=True)
class ExecutorContract:
    id: str
    mode: Mode
    suite_class: str
    target_profile: TargetProfile
    lineage_profile: LineageProfile
    track_ids: tuple[str, ...]
    normalized_suite: bool = False
    recorded_normalized_import: bool = False
    requires_fixture_ref: bool = False
    case_budget_per_suite: bool = False
    evidence_level_ceiling: EvidenceLevel | None = None
    evidence_qualifications: EvidenceQualificationRegistry = (
        EMPTY_EVIDENCE_QUALIFICATIONS
    )

    def __post_init__(self) -> None:
        canonical_tracks = tuple(
            track for track in TRACK_IDS if track in self.track_ids
        )
        if (
            _PORTABLE_ID_RE.fullmatch(self.id) is None
            or _PORTABLE_ID_RE.fullmatch(self.suite_class) is None
            or self.mode not in {"replay", "live"}
            or self.target_profile not in {"recorded-source", "brokered-runtime"}
            or self.lineage_profile
            not in {"fixture-replay", "normalized-suite-replay", "runtime"}
            or not self.track_ids
            or self.track_ids != canonical_tracks
            or not isinstance(
                self.evidence_qualifications, EvidenceQualificationRegistry
            )
            or (
                self.evidence_level_ceiling is not None
                and self.evidence_level_ceiling not in _EVIDENCE_LEVELS
            )
        ):
            raise ValueError(f"invalid evaluation executor contract: {self.id}")
        if (
            (self.mode == "live" and self.target_profile != "brokered-runtime")
            or (
                self.mode == "replay"
                and self.target_profile == "brokered-runtime"
                and self.suite_class != "mom-cohort"
            )
            or self.requires_fixture_ref
            != (
                self.lineage_profile == "fixture-replay"
                or self.suite_class == "mom-cohort"
            )
            or self.recorded_normalized_import
            != (self.lineage_profile == "normalized-suite-replay")
            or (self.recorded_normalized_import and not self.normalized_suite)
        ):
            raise ValueError(f"inconsistent evaluation executor contract: {self.id}")
        qualification_tracks = {
            track_id
            for contract in self.evidence_qualifications.contracts
            for track_id in contract.allowed_tracks
        }
        if self.mode != "live" and self.evidence_qualifications.contracts:
            raise ValueError(
                f"replay executor cannot register live evidence sources: {self.id}"
            )
        if not qualification_tracks.issubset(self.track_ids):
            raise ValueError(
                f"evidence qualification exceeds executor tracks: {self.id}"
            )
        if self.evidence_level_ceiling is not None and any(
            _EVIDENCE_LEVELS.index(contract.ceiling)
            > _EVIDENCE_LEVELS.index(self.evidence_level_ceiling)
            for contract in self.evidence_qualifications.contracts
        ):
            raise ValueError(
                f"evidence qualification exceeds executor ceiling: {self.id}"
            )


def executor_is_mom_cohort_replay(contract: ExecutorContract) -> bool:
    """Return whether a registered executor may replay a frozen Mixture subject."""

    return (
        contract.mode == "replay"
        and contract.suite_class == "mom-cohort"
        and contract.target_profile == "brokered-runtime"
    )


def builtin_executor_contracts() -> tuple[ExecutorContract, ...]:
    return (
        ExecutorContract(
            id=FIXTURE_REPLAY_EXECUTOR_ID,
            mode="replay",
            suite_class="fixture",
            target_profile="recorded-source",
            lineage_profile="fixture-replay",
            track_ids=TRACK_IDS,
            requires_fixture_ref=True,
        ),
        ExecutorContract(
            id=LIVE_RUNTIME_EXECUTOR_ID,
            mode="live",
            suite_class="runtime",
            target_profile="brokered-runtime",
            lineage_profile="runtime",
            track_ids=(
                "routing",
                "model_pool",
                "joint",
                "agentic",
                "multimodal",
                "preference",
                "safety",
                "capacity",
            ),
            case_budget_per_suite=True,
            evidence_qualifications=LIVE_RUNTIME_EVIDENCE_QUALIFICATIONS,
        ),
        ExecutorContract(
            id=MOM_REPLAY_EXECUTOR_ID,
            mode="replay",
            suite_class="mom-cohort",
            target_profile="brokered-runtime",
            lineage_profile="runtime",
            track_ids=("routing", "model_pool", "joint"),
            requires_fixture_ref=True,
            evidence_level_ceiling="E0",
        ),
        ExecutorContract(
            id=NORMALIZED_REPLAY_EXECUTOR_ID,
            mode="replay",
            suite_class="normalized-suite",
            target_profile="recorded-source",
            lineage_profile="normalized-suite-replay",
            track_ids=TRACK_IDS,
            normalized_suite=True,
            recorded_normalized_import=True,
            case_budget_per_suite=True,
        ),
        ExecutorContract(
            id=NORMALIZED_LIVE_EXECUTOR_ID,
            mode="live",
            suite_class="normalized-suite",
            target_profile="brokered-runtime",
            lineage_profile="runtime",
            track_ids=("routing", "model_pool", "joint", "multimodal", "capacity"),
            normalized_suite=True,
            case_budget_per_suite=True,
            evidence_level_ceiling="E4",
            evidence_qualifications=NORMALIZED_LIVE_EVIDENCE_QUALIFICATIONS,
        ),
    )


BUILTIN_EXECUTOR_CONTRACTS = builtin_executor_contracts()
BUILTIN_NORMALIZED_SUITE_EXECUTORS: Mapping[Mode, str] = MappingProxyType(
    {
        "replay": NORMALIZED_REPLAY_EXECUTOR_ID,
        "live": NORMALIZED_LIVE_EXECUTOR_ID,
    }
)
