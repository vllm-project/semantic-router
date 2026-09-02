"""Resolve frozen suite selections into an immutable execution plan."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from cli.evaluation.catalog import get_catalog
from cli.evaluation.catalog_suites import CatalogSuite
from cli.evaluation.contracts import RunManifest
from cli.evaluation.execution_contract import NORMALIZED_LIVE_EXECUTOR_ID
from cli.evaluation.executor_registry import ExecutorRegistry
from cli.evaluation.manifest_identity import require_manifest_digest
from cli.evaluation.normalized_suite_live_admission import (
    normalized_suite_live_tracks,
)
from cli.evaluation.suite_contract import BenchmarkSuiteManifest
from cli.evaluation.suite_store import NormalizedSuiteStore
from cli.evaluation.suite_store_error import SuiteStoreError
from cli.evaluation.target_capabilities import (
    DEFAULT_TARGET_REGISTRY,
    TargetRegistry,
)


@dataclass(frozen=True)
class ExecutionPlan:
    suites: tuple[BenchmarkSuiteManifest, ...]
    suite_revisions: Mapping[str, str]
    suite_executors: Mapping[str, str]
    allowed_tracks: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "suite_revisions", MappingProxyType(dict(self.suite_revisions))
        )
        object.__setattr__(
            self, "suite_executors", MappingProxyType(dict(self.suite_executors))
        )
        if set(self.suite_revisions) != set(self.suite_executors):
            raise ValueError("execution plan suite identities must have equal key sets")
        if len(set(self.suite_executors.values())) != 1:
            raise ValueError("one evaluation run cannot mix executor implementations")

    @property
    def executor_id(self) -> str:
        return next(iter(self.suite_executors.values()))


class SuiteRegistry:
    """Immutable suite contracts used during plan resolution."""

    def __init__(self, suites: Iterable[CatalogSuite]):
        by_id: dict[str, CatalogSuite] = {}
        for suite in suites:
            if suite.id in by_id:
                raise ValueError(f"duplicate evaluation suite id: {suite.id}")
            by_id[suite.id] = suite
        if not by_id:
            raise ValueError("evaluation suite registry cannot be empty")
        self._by_id = MappingProxyType(by_id)

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self._by_id)

    def require_many(self, suite_ids: tuple[str, ...]) -> tuple[CatalogSuite, ...]:
        try:
            return tuple(self._by_id[suite_id] for suite_id in suite_ids)
        except KeyError as exc:
            raise ValueError(f"unknown evaluation suite: {exc.args[0]}") from exc


def _builtin_plan(
    manifest: RunManifest, suites: tuple[CatalogSuite, ...]
) -> ExecutionPlan:
    for suite in suites:
        if manifest.mode not in suite.modes:
            raise ValueError(f"suite {suite.id} does not support mode {manifest.mode}")
        if suite.revision is None:
            raise ValueError(f"suite {suite.id} has no immutable revision")
    return ExecutionPlan(
        suites=(),
        suite_revisions={suite.id: suite.revision for suite in suites},
        suite_executors={suite.id: suite.executors[manifest.mode] for suite in suites},
        allowed_tracks=frozenset(
            track for suite in suites for track in suite.track_ids
        ),
    )


def _installed_plan(
    manifest: RunManifest,
    suite_store: NormalizedSuiteStore | None,
    executor_registry: ExecutorRegistry,
) -> ExecutionPlan:
    if suite_store is None:
        raise ValueError("installed suite execution requires a trusted suite store")
    suites: list[BenchmarkSuiteManifest] = []
    for suite_id in manifest.suite_ids:
        try:
            suites.append(suite_store.get_suite_manifest(suite_id))
        except SuiteStoreError as exc:
            raise ValueError(f"unknown or invalid installed suite: {suite_id}") from exc
    manifests = tuple(suites)
    executor_ids = set(manifest.suite_executors.values())
    if len(executor_ids) != 1:
        raise ValueError("installed suite execution requires one frozen executor")
    executor_id = next(iter(executor_ids))
    executor = executor_registry.contract(executor_id)
    if not executor.normalized_suite or executor.mode != manifest.mode:
        raise ValueError(
            "frozen executor is not admitted for normalized suite execution"
        )
    if executor.recorded_normalized_import:
        for suite in manifests:
            if suite.qualification_receipt.executor_id != executor_id:
                raise ValueError(
                    "installed suite qualification does not admit the frozen executor"
                )
        allowed_tracks = frozenset(
            track for suite in manifests for track in suite.track_ids
        )
    elif executor_id == NORMALIZED_LIVE_EXECUTOR_ID:
        admitted_by_suite = {
            suite.id: normalized_suite_live_tracks(suite_store, suite)
            for suite in manifests
        }
        for suite in manifests:
            inadmissible = sorted(
                set(manifest.track_ids).intersection(suite.track_ids)
                - admitted_by_suite[suite.id]
            )
            if inadmissible:
                raise ValueError(
                    f"suite {suite.id} has no first-party normalized live method for "
                    + ", ".join(inadmissible)
                )
        allowed_tracks = frozenset(
            track for tracks in admitted_by_suite.values() for track in tracks
        )
    else:
        raise ValueError("frozen executor has no normalized suite admission registry")
    return ExecutionPlan(
        suites=manifests,
        suite_revisions={suite.id: suite.revision for suite in manifests},
        suite_executors=dict.fromkeys((suite.id for suite in manifests), executor_id),
        allowed_tracks=allowed_tracks,
    )


def resolve_execution_plan(
    manifest: RunManifest,
    suite_store: NormalizedSuiteStore | None,
    suite_registry: SuiteRegistry,
    executor_registry: ExecutorRegistry,
    target_registry: TargetRegistry = DEFAULT_TARGET_REGISTRY,
) -> ExecutionPlan:
    require_manifest_digest(manifest)
    builtin_ids = set(manifest.suite_ids).intersection(suite_registry.ids)
    if builtin_ids and len(builtin_ids) != len(manifest.suite_ids):
        raise ValueError("builtin and installed suites cannot be mixed")
    plan = (
        _builtin_plan(manifest, suite_registry.require_many(manifest.suite_ids))
        if builtin_ids
        else _installed_plan(manifest, suite_store, executor_registry)
    )
    if manifest.suite_revisions != plan.suite_revisions:
        raise ValueError(
            "manifest suite revisions do not match the active executor catalog"
        )
    if manifest.suite_executors != plan.suite_executors:
        raise ValueError(
            "manifest suite executors do not match the active executor catalog"
        )
    disallowed = sorted(set(manifest.track_ids) - plan.allowed_tracks)
    if disallowed:
        raise ValueError(
            "tracks are not covered by selected suites: " + ", ".join(disallowed)
        )
    executor = executor_registry.contract(plan.executor_id)
    if executor.mode != manifest.mode:
        raise ValueError("frozen executor does not support the manifest mode")
    unsupported_by_executor = sorted(set(manifest.track_ids) - set(executor.track_ids))
    if unsupported_by_executor:
        raise ValueError(
            "frozen executor cannot produce selected tracks: "
            + ", ".join(unsupported_by_executor)
        )
    target_registry.resolve(manifest, executor)
    return plan


DEFAULT_SUITE_REGISTRY = SuiteRegistry(get_catalog(generated_at=False).suites)
