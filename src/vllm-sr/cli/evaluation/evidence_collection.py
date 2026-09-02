"""Validate evidence returned across the executor extension boundary."""

from __future__ import annotations

from cli.evaluation.builtin_executors import DEFAULT_EXECUTOR_REGISTRY
from cli.evaluation.case_plan import require_complete_evidence_plan
from cli.evaluation.contracts import RunManifest
from cli.evaluation.execution_contract import (
    require_discovered_entrypoint_binding,
    require_normalized_identity_binding,
)
from cli.evaluation.execution_plan import ExecutionPlan
from cli.evaluation.executor_registry import CollectedEvidence, ExecutorRegistry
from cli.evaluation.routing_trace import require_routing_trace_binding
from cli.evaluation.store import ArtifactStore
from cli.evaluation.suite_store import NormalizedSuiteStore


def collect_evidence(
    manifest: RunManifest,
    store: ArtifactStore,
    plan: ExecutionPlan,
    *,
    suite_store: NormalizedSuiteStore | None = None,
    registry: ExecutorRegistry = DEFAULT_EXECUTOR_REGISTRY,
) -> CollectedEvidence:
    executor = registry.require(plan.executor_id)
    collected = executor.collect(manifest, store, plan, suite_store)
    _validate_fixture_ref(
        plan.executor_id,
        executor.contract.requires_fixture_ref,
        collected,
    )
    _validate_suite_identity(plan, collected)
    require_normalized_identity_binding(
        manifest,
        collected.inputs.visible,
        collected.records,
        collected.inputs.private_identity_map,
        required=executor.contract.normalized_suite,
        recorded_import=executor.contract.recorded_normalized_import,
    )
    require_discovered_entrypoint_binding(
        manifest,
        collected.discovered_entrypoints,
    )
    _validate_executor_tracks(manifest, plan, collected)
    require_complete_evidence_plan(
        manifest,
        collected.inputs.visible,
        collected.records,
    )
    require_routing_trace_binding(
        manifest,
        collected.inputs.visible,
        collected.records,
        collected.routing_traces,
    )
    return collected


def _validate_fixture_ref(
    executor_id: str,
    requires_fixture_ref: bool,
    collected: CollectedEvidence,
) -> None:
    if requires_fixture_ref and collected.fixture_ref is None:
        raise ValueError(
            f"executor {executor_id!r} omitted its required fixture reference"
        )
    if not requires_fixture_ref and collected.fixture_ref is not None:
        raise ValueError(
            f"executor {executor_id!r} returned an undeclared fixture reference"
        )


def _validate_suite_identity(
    plan: ExecutionPlan,
    collected: CollectedEvidence,
) -> None:
    if collected.inputs.suite_revisions != plan.suite_revisions:
        raise ValueError("executor returned evidence for different suite revisions")
    if collected.inputs.suite_executors != plan.suite_executors:
        raise ValueError("executor returned evidence for different suite executors")


def _validate_executor_tracks(
    manifest: RunManifest,
    plan: ExecutionPlan,
    collected: CollectedEvidence,
) -> None:
    wrong_executor_tracks = sorted(
        track_id
        for track_id in manifest.track_ids
        if collected.inputs.executor_ids.get(track_id) != plan.executor_id
    )
    if wrong_executor_tracks:
        raise ValueError(
            "executor identity does not match the execution plan for tracks: "
            + ", ".join(wrong_executor_tracks)
        )
