"""Target feature contracts and manifest-bound capability resolution."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from cli.evaluation.constants import TRACK_IDS
from cli.evaluation.contract_validation import is_subject_target_id
from cli.evaluation.contracts import RunManifest
from cli.evaluation.execution_contract import (
    FIXTURE_REPLAY_EXECUTOR_ID,
    LIVE_RUNTIME_EXECUTOR_ID,
    MOM_REPLAY_EXECUTOR_ID,
    NORMALIZED_LIVE_EXECUTOR_ID,
    NORMALIZED_REPLAY_EXECUTOR_ID,
)
from cli.evaluation.executor_contracts import (
    BUILTIN_EXECUTOR_CONTRACTS,
    ExecutorContract,
    Mode,
    TargetProfile,
    executor_is_mom_cohort_replay,
)
from cli.evaluation.reporting import EvidenceLevel
from cli.evaluation.target_contracts import EvaluationTarget, ManifestMixture

PolicySnapshotProfile = Literal[
    "fixture", "normalized-suite-revisions", "runtime-config"
]
HealthProfile = Literal["always", "installed-suites", "capabilities"]

_PORTABLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENDPOINT_FEATURES = frozenset(
    {
        "agent_task_ledger",
        "fault_recovery_ledger",
        "hard_policy_ledger",
        "production_experiment_ledger",
    }
)
_MIN_MIXTURE_ARMS = 2
_LEDGER_FEATURES = {
    "agent_task_ledger": "agent_task_ledger",
    "fault_recovery_ledger": "fault_recovery_ledger",
    "hard_policy_ledger": "hard_policy_ledger",
    "production_experiment_ledger": "production_experiment_ledger",
}


@dataclass(frozen=True)
class TargetContract:
    id: str
    name: str
    description: str
    kind: str
    track_requirements: Mapping[str, frozenset[str]]
    modes: tuple[Mode, ...]
    accepted_executors: Mapping[Mode, tuple[str, ...]]
    execution_profile: TargetProfile
    policy_snapshot_profile: PolicySnapshotProfile
    health_profile: HealthProfile
    evidence_level: EvidenceLevel | None = None
    labels: Mapping[str, str] | None = None
    provided_features: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        track_ids = tuple(self.track_requirements)
        canonical_tracks = tuple(track for track in TRACK_IDS if track in track_ids)
        canonical_modes = tuple(
            mode for mode in ("replay", "live") if mode in self.modes
        )
        if (
            _PORTABLE_ID_RE.fullmatch(self.id) is None
            or _PORTABLE_ID_RE.fullmatch(self.kind) is None
            or not self.name.strip()
            or self.name != self.name.strip()
            or self.description != self.description.strip()
            or not track_ids
            or track_ids != canonical_tracks
            or self.modes != canonical_modes
            or set(self.accepted_executors) != set(self.modes)
            or self.execution_profile not in {"recorded-source", "brokered-runtime"}
            or self.policy_snapshot_profile
            not in {"fixture", "normalized-suite-revisions", "runtime-config"}
            or self.health_profile not in {"always", "installed-suites", "capabilities"}
        ):
            raise ValueError(f"invalid evaluation target contract: {self.id}")
        requirements: dict[str, frozenset[str]] = {}
        for track_id, required in self.track_requirements.items():
            features = frozenset(required)
            if any(_PORTABLE_ID_RE.fullmatch(feature) is None for feature in features):
                raise ValueError(
                    f"invalid track capability on evaluation target: {self.id}"
                )
            requirements[track_id] = features
        provided_features = frozenset(self.provided_features)
        if any(
            _PORTABLE_ID_RE.fullmatch(feature) is None for feature in provided_features
        ) or provided_features.intersection(_ENDPOINT_FEATURES):
            raise ValueError(f"invalid provided target feature: {self.id}")
        object.__setattr__(self, "track_requirements", MappingProxyType(requirements))
        object.__setattr__(self, "provided_features", provided_features)
        if (
            self.execution_profile == "brokered-runtime"
            and self.policy_snapshot_profile != "runtime-config"
        ) or (
            self.execution_profile == "recorded-source"
            and self.policy_snapshot_profile == "runtime-config"
        ):
            raise ValueError(f"inconsistent evaluation target contract: {self.id}")
        accepted: dict[Mode, tuple[str, ...]] = {}
        for mode in self.modes:
            executor_ids = tuple(self.accepted_executors[mode])
            if (
                not executor_ids
                or len(executor_ids) != len(set(executor_ids))
                or any(
                    _PORTABLE_ID_RE.fullmatch(value) is None for value in executor_ids
                )
            ):
                raise ValueError(
                    f"invalid executor capability on evaluation target: {self.id}"
                )
            accepted[mode] = executor_ids
        object.__setattr__(self, "accepted_executors", MappingProxyType(accepted))
        object.__setattr__(
            self,
            "labels",
            MappingProxyType(dict(self.labels)) if self.labels is not None else None,
        )

    def available_tracks(self, target: EvaluationTarget) -> tuple[str, ...]:
        features = self.provided_features.union(target_features(target))
        return tuple(
            track_id
            for track_id, required in self.track_requirements.items()
            if required.issubset(features)
        )

    def healthy(self, target: EvaluationTarget, installed_suite_count: int) -> bool:
        if self.health_profile == "always":
            return True
        if self.health_profile == "installed-suites":
            return installed_suite_count > 0
        return bool(self.available_tracks(target))

    @property
    def track_ids(self) -> tuple[str, ...]:
        return tuple(self.track_requirements)


def target_features(target: EvaluationTarget) -> frozenset[str]:
    if target.kind == "mixture-of-models" and target.mixture is None:
        return frozenset()
    features: set[str] = set()
    if target.backend_topology_digest is not None:
        features.add("topology")
    if target.router_api_url is not None:
        features.add("router-api")
    if target.envoy_url is not None:
        features.add("envoy-chat")
    if target.mixture is not None:
        features.update(_mixture_features(target.mixture))
    features.update(_ledger_features(target))
    return frozenset(features)


def _mixture_features(mixture: ManifestMixture) -> set[str]:
    features = set()
    if len(mixture.model_arms) >= _MIN_MIXTURE_ARMS:
        features.add("mixture-pool")
    if any(
        modality != "text" for arm in mixture.model_arms for modality in arm.modalities
    ):
        features.add("multimodal-arm")
    return features


def _ledger_features(target: EvaluationTarget) -> set[str]:
    features = {
        feature
        for attribute, feature in _LEDGER_FEATURES.items()
        if getattr(target, attribute) is not None
    }
    if target.agent_task_ledger is not None or target.fault_recovery_ledger is not None:
        features.add("agentic-ledger")
    return features


class TargetRegistry:
    """Immutable target contracts validated against executor capabilities."""

    def __init__(
        self,
        targets: Iterable[TargetContract],
        executors: Iterable[ExecutorContract],
    ):
        executor_by_id = {contract.id: contract for contract in executors}
        by_id: dict[str, TargetContract] = {}
        for target in targets:
            if target.id in by_id:
                raise ValueError(f"duplicate evaluation target id: {target.id}")
            for mode, executor_ids in target.accepted_executors.items():
                for executor_id in executor_ids:
                    executor = executor_by_id.get(executor_id)
                    if (
                        executor is None
                        or executor.mode != mode
                        or executor.target_profile != target.execution_profile
                    ):
                        raise ValueError(
                            f"target {target.id} accepts incompatible executor {executor_id}"
                        )
            by_id[target.id] = target
        if not by_id:
            raise ValueError("evaluation target registry cannot be empty")
        self._by_id = MappingProxyType(by_id)

    @property
    def contracts(self) -> tuple[TargetContract, ...]:
        return tuple(self._by_id.values())

    def require(self, target_id: str) -> TargetContract:
        try:
            return self._by_id[target_id]
        except KeyError as exc:
            raise ValueError(f"unknown evaluation target: {target_id}") from exc

    def resolve(
        self, manifest: RunManifest, executor: ExecutorContract
    ) -> TargetContract:
        if set(manifest.suite_executors.values()) != {executor.id}:
            raise ValueError(
                "registered executor does not match the manifest-frozen identity"
            )
        contract = self._by_id.get(manifest.target.id)
        if contract is None and manifest.target.mixture is not None:
            contract = mixture_target_contract(manifest.target.mixture)
        elif contract is None:
            contract = self.require(manifest.target.id)
        if manifest.mode == "replay" and manifest.target.mixture is not None:
            executor_admitted = executor_is_mom_cohort_replay(executor)
        else:
            executor_admitted = (
                manifest.mode in contract.modes
                and executor.id in contract.accepted_executors[manifest.mode]
            )
        if (
            manifest.target.kind != contract.kind
            or manifest.mode not in contract.modes
            or not executor_admitted
            or executor.mode != manifest.mode
            or executor.target_profile != contract.execution_profile
        ):
            raise ValueError("manifest target does not accept its frozen executor")
        self._validate_target_shape(manifest, contract)
        unsupported = sorted(
            set(manifest.track_ids) - set(contract.available_tracks(manifest.target))
        )
        if unsupported:
            raise ValueError(
                "manifest target cannot execute selected tracks: "
                + ", ".join(unsupported)
            )
        self._validate_method_endpoints(manifest)
        return contract

    @staticmethod
    def _validate_method_endpoints(manifest: RunManifest) -> None:
        if manifest.mode != "live":
            return
        if (
            "live-agent-tasks" in manifest.suite_ids
            and manifest.target.agent_task_ledger is None
        ):
            raise ValueError(
                "live-agent-tasks requires its dedicated agent_task_ledger capability"
            )
        if (
            "live-fault-recovery" in manifest.suite_ids
            and manifest.target.fault_recovery_ledger is None
        ):
            raise ValueError(
                "live-fault-recovery requires its dedicated fault_recovery_ledger capability"
            )

    @staticmethod
    def _validate_target_shape(manifest: RunManifest, contract: TargetContract) -> None:
        target = manifest.target
        if contract.execution_profile == "recorded-source":
            if (
                target.router_api_url is not None
                or target.envoy_url is not None
                or target.router_api_key is not None
                or target.envoy_api_key is not None
                or target.agent_task_ledger is not None
                or target.fault_recovery_ledger is not None
                or target.hard_policy_ledger is not None
                or target.production_experiment_ledger is not None
                or target.backend_topology_digest is not None
                or target.mixture is not None
            ):
                raise ValueError("recorded-source target contains runtime connectivity")
            return
        if (
            target.envoy_url is None
            or target.backend_topology_digest is None
            or target.mixture is None
            or not is_subject_target_id(target.id, target.mixture.id)
            or target.kind != "mixture-of-models"
        ):
            raise ValueError("brokered-runtime target is incomplete")


def mixture_target_contract(mixture: ManifestMixture) -> TargetContract:
    """Build the one runtime target contract bound to a frozen Mixture."""

    return TargetContract(
        id=mixture.id,
        name=mixture.entrypoint_model,
        description=(
            mixture.recipe_description
            or "Evaluate this routing recipe and its model pool together as one system."
        ),
        kind="mixture-of-models",
        track_requirements={
            "routing": frozenset({"topology", "router-api", "envoy-chat"}),
            "model_pool": frozenset({"topology", "envoy-chat", "mixture-pool"}),
            "joint": frozenset({"topology", "envoy-chat", "mixture-pool"}),
            "agentic": frozenset({"topology", "envoy-chat", "agentic-ledger"}),
            "multimodal": frozenset({"topology", "envoy-chat", "multimodal-arm"}),
            "preference": frozenset(
                {"topology", "envoy-chat", "production_experiment_ledger"}
            ),
            "safety": frozenset({"topology", "envoy-chat", "hard_policy_ledger"}),
            "capacity": frozenset({"topology", "envoy-chat"}),
        },
        modes=("replay", "live"),
        accepted_executors={
            "replay": (MOM_REPLAY_EXECUTOR_ID,),
            "live": (LIVE_RUNTIME_EXECUTOR_ID, NORMALIZED_LIVE_EXECUTOR_ID),
        },
        execution_profile="brokered-runtime",
        policy_snapshot_profile="runtime-config",
        health_profile="capabilities",
        labels={
            "capabilities": "mixture-bound",
            "credentials": "server-brokered",
            "model_arms": "server-owned",
        },
    )


def builtin_target_contracts() -> tuple[TargetContract, ...]:
    return (
        TargetContract(
            id="fixture",
            name="Built-in evaluation sample",
            description=(
                "A small deterministic replay for checking the full evaluation workflow "
                "without calling a live system."
            ),
            kind="builtin-fixture",
            track_requirements={track_id: frozenset() for track_id in TRACK_IDS},
            modes=("replay",),
            accepted_executors={"replay": (FIXTURE_REPLAY_EXECUTOR_ID,)},
            execution_profile="recorded-source",
            policy_snapshot_profile="fixture",
            health_profile="always",
            evidence_level="E0",
            labels={"execution": "local", "network": "none"},
        ),
        TargetContract(
            id="benchmark-source",
            name="Imported benchmark results",
            description=(
                "Replay saved results from pinned benchmark revisions. This evaluates "
                "imported observations, not the live system."
            ),
            kind="normalized-benchmark-source",
            track_requirements={track_id: frozenset() for track_id in TRACK_IDS},
            modes=("replay",),
            accepted_executors={"replay": (NORMALIZED_REPLAY_EXECUTOR_ID,)},
            execution_profile="recorded-source",
            policy_snapshot_profile="normalized-suite-revisions",
            health_profile="installed-suites",
            labels={
                "execution": "recorded-source",
                "identity": "suite-revision-bound",
                "network": "none",
            },
        ),
    )


DEFAULT_TARGET_REGISTRY = TargetRegistry(
    builtin_target_contracts(), BUILTIN_EXECUTOR_CONTRACTS
)
