"""Resolve, execute, reduce, and finalize evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cli.evaluation.builtin_executors import DEFAULT_EXECUTOR_REGISTRY
from cli.evaluation.canonical import strict_json_loads
from cli.evaluation.capacity_profile import CapacityProfile, build_capacity_profile
from cli.evaluation.case_plan import planned_case_ids_by_track
from cli.evaluation.contract_primitives import ArtifactRef
from cli.evaluation.contracts import ResolvedRunSnapshot, RunManifest
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_collection import collect_evidence
from cli.evaluation.evidence_level import run_evidence_level
from cli.evaluation.execution_contract import ExecutionPlan
from cli.evaluation.execution_plan import (
    DEFAULT_SUITE_REGISTRY,
    SuiteRegistry,
    resolve_execution_plan,
)
from cli.evaluation.executor_registry import CollectedEvidence, ExecutorRegistry
from cli.evaluation.finalize import finalize_report_bundle
from cli.evaluation.gates import compute_gates
from cli.evaluation.method_gate_evidence import derive_method_gate_evidence
from cli.evaluation.metric_model_pool_contract import ModelPoolReductionContext
from cli.evaluation.metrics import compute_metrics
from cli.evaluation.normalized_suite_live_robustness import (
    declared_shift_gate_is_complete,
)
from cli.evaluation.published_bundle import load_published_report_bundle
from cli.evaluation.reporting import (
    EvaluationGate,
    EvaluationMetric,
)
from cli.evaluation.resolution import resolve_snapshot
from cli.evaluation.run_ownership import (
    EventSink,
    RunOwnership,
    StandaloneRunOwnership,
    WorkerRunOwnership,
)
from cli.evaluation.statistics import attach_confidence_intervals
from cli.evaluation.store import ArtifactStore, LocalArtifactStore, WorkerArtifactStore
from cli.evaluation.suite_store import NormalizedSuiteStore
from cli.evaluation.target_capabilities import (
    DEFAULT_TARGET_REGISTRY,
    TargetRegistry,
)
from cli.evaluation.worker_report import (
    WorkerEvent,
    WorkerReportDraft,
    WorkerRunProgress,
    worker_run_state_from_manifest,
)


@dataclass(frozen=True)
class ReducedRunEvidence:
    resolved: ResolvedRunSnapshot
    capacity_profile: CapacityProfile | None
    metrics: list[EvaluationMetric]
    gates: list[EvaluationGate]
    completed_at: datetime


def load_manifest(path: str | Path) -> RunManifest:
    try:
        return RunManifest.model_validate(strict_json_loads(Path(path).read_bytes()))
    except OSError as exc:
        raise ValueError(f"cannot read evaluation manifest: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid evaluation manifest: {exc}") from exc


def validate_manifest(
    manifest: RunManifest,
    suite_store: NormalizedSuiteStore | None = None,
    executor_registry: ExecutorRegistry = DEFAULT_EXECUTOR_REGISTRY,
    suite_registry: SuiteRegistry = DEFAULT_SUITE_REGISTRY,
    target_registry: TargetRegistry = DEFAULT_TARGET_REGISTRY,
) -> None:
    plan = resolve_execution_plan(
        manifest,
        suite_store,
        suite_registry,
        executor_registry,
        target_registry,
    )
    executor_registry.require(plan.executor_id)


def _completed_event(report: WorkerReportDraft) -> WorkerEvent:
    return WorkerEvent(
        type="completed",
        message="Evaluation completed and artifacts were finalized",
        progress=report.run.progress,
        payload={"verdict": report.summary.verdict},
    )


def _emit_track_events(
    manifest: RunManifest,
    ownership: RunOwnership,
    records: list[ExecutionRecord],
) -> None:
    for index, track_id in enumerate(manifest.track_ids, 1):
        record_count = sum(row.track_id == track_id for row in records)
        event_progress = WorkerRunProgress(
            percent=100 * index / len(manifest.track_ids),
            completed=index,
            total=len(manifest.track_ids),
            current_track_id=track_id,
            message=f"Collected {record_count} records",
        )
        ownership.emit(
            manifest.run_id,
            WorkerEvent(
                type="track",
                message=f"Completed {track_id} evidence collection",
                track_id=track_id,
                progress=event_progress,
                payload={"record_count": record_count},
            ),
        )


def _start_run(
    manifest: RunManifest,
    ownership: RunOwnership,
) -> tuple[datetime, WorkerRunProgress, ArtifactRef]:
    started = datetime.now(timezone.utc)
    progress = WorkerRunProgress(
        percent=0,
        completed=0,
        total=len(manifest.track_ids),
        message="Starting evaluation",
    )
    run = worker_run_state_from_manifest(
        manifest,
        status="running",
        progress=progress,
        started_at=started,
    )
    manifest_ref = ownership.bind_manifest(manifest)
    ownership.record_state(run)
    ownership.emit(
        manifest.run_id,
        WorkerEvent(
            type="snapshot", message="Evaluation manifest validated", progress=progress
        ),
    )
    return started, progress, manifest_ref


def _execute_run(
    manifest: RunManifest,
    store: ArtifactStore,
    manifest_ref: ArtifactRef,
    started: datetime,
    ownership: RunOwnership,
    suite_store: NormalizedSuiteStore | None,
    plan: ExecutionPlan,
    executor_registry: ExecutorRegistry,
) -> WorkerReportDraft:
    executor = executor_registry.contract(plan.executor_id)
    collected = collect_evidence(
        manifest,
        store,
        plan,
        suite_store=suite_store,
        registry=executor_registry,
    )
    records = collected.records
    _emit_track_events(
        manifest,
        ownership,
        records,
    )
    reduced = _reduce_run_evidence(manifest, collected)
    final_progress = WorkerRunProgress(
        percent=100,
        completed=len(manifest.track_ids),
        total=len(manifest.track_ids),
        message="Evaluation completed",
    )
    completed_run = worker_run_state_from_manifest(
        manifest,
        status="completed",
        progress=final_progress,
        started_at=started,
        completed_at=reduced.completed_at,
        evidence_level=run_evidence_level(
            manifest.mode,
            executor,
            manifest.track_ids,
            records,
        ),
    )
    report = finalize_report_bundle(
        manifest=manifest,
        executor=executor,
        store=store,
        manifest_ref=manifest_ref,
        inputs=collected.inputs,
        records=records,
        resolved=reduced.resolved,
        metrics=reduced.metrics,
        gates=reduced.gates,
        routing_traces=collected.routing_traces,
        capacity_profile=reduced.capacity_profile,
        run=completed_run,
        completed_at=reduced.completed_at,
        benchmark_revisions=collected.inputs.suite_revisions,
        private_identity_map=collected.inputs.private_identity_map,
    )
    ownership.record_state(completed_run)
    ownership.emit(
        manifest.run_id,
        _completed_event(report),
    )
    return report


def _reduce_run_evidence(
    manifest: RunManifest,
    collected: CollectedEvidence,
) -> ReducedRunEvidence:
    records = collected.records
    resolved = resolve_snapshot(
        manifest,
        collected.inputs,
        collected.visible_ref,
        collected.grading_ref,
        collected.fixture_ref,
        collected.discovered_entrypoints,
    )
    capacity_profile = (
        build_capacity_profile(
            records,
            manifest.capacity_slo,
            manifest.capacity_load_protocol,
        )
        if manifest.capacity_slo is not None
        else None
    )
    planned_case_ids = planned_case_ids_by_track(
        collected.inputs.visible, manifest.track_ids
    )
    model_pool_context = None
    if manifest.target.mixture is not None and "model_pool" in manifest.track_ids:
        model_pool_context = ModelPoolReductionContext(
            frozen_arm_ids=tuple(arm.id for arm in manifest.target.mixture.model_arms),
            planned_case_ids=tuple(planned_case_ids["model_pool"]),
            authoritative=manifest.mode == "live",
        )
    metrics = attach_confidence_intervals(
        compute_metrics(
            records,
            capacity_profile=capacity_profile,
            model_pool_context=model_pool_context,
        ),
        records,
        seed=manifest.seed,
    )
    completed = datetime.now(timezone.utc)
    gates = compute_gates(
        metrics,
        has_records=bool(records),
        change_profile=manifest.change_profile,
        evidence=derive_method_gate_evidence(
            manifest,
            records,
            method_qualified_gate_ids=(
                frozenset({"G4"})
                if manifest.mode == "live"
                and set(collected.inputs.suite_executors.values())
                == {"normalized-suite-live.v1"}
                and declared_shift_gate_is_complete(records)
                else frozenset()
            ),
        ),
        records=records,
        evaluated_at=completed,
    )
    return ReducedRunEvidence(
        resolved=resolved,
        capacity_profile=capacity_profile,
        metrics=metrics,
        gates=gates,
        completed_at=completed,
    )


def _record_failure(
    manifest: RunManifest,
    started: datetime,
    progress: WorkerRunProgress,
    ownership: RunOwnership,
    exc: Exception,
) -> None:
    failed_progress = WorkerRunProgress(
        percent=progress.percent,
        completed=progress.completed,
        total=progress.total,
        message="Evaluation failed",
    )
    failed = worker_run_state_from_manifest(
        manifest,
        status="failed",
        progress=failed_progress,
        started_at=started,
        completed_at=datetime.now(timezone.utc),
        error=f"{type(exc).__name__}: {exc}",
    )
    ownership.record_state(failed)
    ownership.emit(
        manifest.run_id,
        WorkerEvent(type="failed", message="Evaluation failed"),
    )


def _load_existing_report(
    manifest: RunManifest,
    store: ArtifactStore,
    executor_registry: ExecutorRegistry,
) -> WorkerReportDraft | None:
    executor_id = next(iter(manifest.suite_executors.values()))
    bundle = load_published_report_bundle(
        manifest,
        store,
        executor_registry.contract(executor_id),
    )
    if bundle is None:
        return None
    return bundle.report


def _run_with_ownership(
    manifest: RunManifest,
    store: ArtifactStore,
    ownership: RunOwnership,
    *,
    suite_store: NormalizedSuiteStore | None = None,
    executor_registry: ExecutorRegistry = DEFAULT_EXECUTOR_REGISTRY,
    suite_registry: SuiteRegistry = DEFAULT_SUITE_REGISTRY,
    target_registry: TargetRegistry = DEFAULT_TARGET_REGISTRY,
) -> WorkerReportDraft:
    report = _load_existing_report(manifest, store, executor_registry)
    if report is not None:
        ownership.reconcile_completed(report, _completed_event(report))
        return report
    plan = resolve_execution_plan(
        manifest,
        suite_store,
        suite_registry,
        executor_registry,
        target_registry,
    )
    executor_registry.require(plan.executor_id)
    started, progress, manifest_ref = _start_run(manifest, ownership)
    try:
        return _execute_run(
            manifest,
            store=store,
            manifest_ref=manifest_ref,
            started=started,
            ownership=ownership,
            suite_store=suite_store,
            plan=plan,
            executor_registry=executor_registry,
        )
    except Exception as exc:
        _record_failure(
            manifest,
            started,
            progress,
            ownership,
            exc,
        )
        raise


def run_evaluation(
    manifest: RunManifest,
    store: LocalArtifactStore,
    *,
    event_sink: EventSink | None = None,
    suite_store: NormalizedSuiteStore | None = None,
    executor_registry: ExecutorRegistry = DEFAULT_EXECUTOR_REGISTRY,
    suite_registry: SuiteRegistry = DEFAULT_SUITE_REGISTRY,
    target_registry: TargetRegistry = DEFAULT_TARGET_REGISTRY,
) -> WorkerReportDraft:
    """Run standalone with local ownership of recovery and control state."""

    with store.execution_lease(manifest.run_id):
        store.recover_report_bundle(manifest)
        return _run_with_ownership(
            manifest,
            store,
            StandaloneRunOwnership(store, event_sink),
            suite_store=suite_store,
            executor_registry=executor_registry,
            suite_registry=suite_registry,
            target_registry=target_registry,
        )


def run_worker_evaluation(
    manifest: RunManifest,
    store: WorkerArtifactStore,
    *,
    event_sink: EventSink | None = None,
    suite_store: NormalizedSuiteStore | None = None,
    executor_registry: ExecutorRegistry = DEFAULT_EXECUTOR_REGISTRY,
    suite_registry: SuiteRegistry = DEFAULT_SUITE_REGISTRY,
    target_registry: TargetRegistry = DEFAULT_TARGET_REGISTRY,
) -> WorkerReportDraft:
    """Run in a Dashboard staging tree without owning mutable control state."""

    return _run_with_ownership(
        manifest,
        store,
        WorkerRunOwnership(store, event_sink),
        suite_store=suite_store,
        executor_registry=executor_registry,
        suite_registry=suite_registry,
        target_registry=target_registry,
    )
