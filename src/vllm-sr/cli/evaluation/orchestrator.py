"""Run lifecycle, artifact finalization, and executor orchestration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from cli.evaluation.catalog import get_catalog
from cli.evaluation.contracts import ArtifactRef, RunManifest
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_level import run_evidence_level
from cli.evaluation.execution import collect_evidence
from cli.evaluation.finalize import finalize_report_bundle
from cli.evaluation.gates import GateEvidenceContext, compute_gates
from cli.evaluation.metrics import compute_metrics
from cli.evaluation.report_builder import build_costs
from cli.evaluation.reporting import (
    EvaluationReport,
    EvaluationRun,
    EvaluationRunProgress,
    WorkerEvent,
)
from cli.evaluation.resolution import resolve_snapshot
from cli.evaluation.statistics import attach_confidence_intervals
from cli.evaluation.store import LocalArtifactStore, StoreError
from cli.evaluation.suite_contract import BenchmarkSuiteManifest
from cli.evaluation.suite_store import NormalizedSuiteStore, SuiteStoreError

EventSink = Callable[[WorkerEvent], None]


def load_manifest(path: str | Path) -> RunManifest:
    try:
        return RunManifest.model_validate_json(Path(path).read_bytes())
    except OSError as exc:
        raise ValueError(f"cannot read evaluation manifest: {exc}") from exc
    except ValidationError as exc:
        raise ValueError(f"invalid evaluation manifest: {exc}") from exc


def _external_suites(
    manifest: RunManifest,
    suite_store: NormalizedSuiteStore | None,
) -> tuple[BenchmarkSuiteManifest, ...]:
    catalog = get_catalog(generated_at=False)
    suites = {suite.id: suite for suite in catalog.suites}
    builtin_ids = set(manifest.suite_ids).intersection(suites)
    external_ids = sorted(set(manifest.suite_ids) - set(suites))
    if builtin_ids and external_ids:
        raise ValueError("builtin and installed external suites cannot be mixed")
    if not external_ids:
        return ()
    if suite_store is None:
        raise ValueError(f"unknown suite ids: {', '.join(external_ids)}")
    manifests: list[BenchmarkSuiteManifest] = []
    try:
        for suite_id in external_ids:
            manifests.append(suite_store.get_suite_manifest(suite_id))
    except SuiteStoreError as exc:
        raise ValueError(f"unknown or invalid installed suite: {suite_id}") from exc
    return tuple(manifests)


def validate_manifest(
    manifest: RunManifest,
    suite_store: NormalizedSuiteStore | None = None,
) -> None:
    catalog = get_catalog(generated_at=False)
    suites = {suite.id: suite for suite in catalog.suites}
    external_suites = _external_suites(manifest, suite_store)
    actual_revisions = (
        {suite.id: suite.revision for suite in external_suites}
        if external_suites
        else {suite_id: suites[suite_id].revision for suite_id in manifest.suite_ids}
    )
    if manifest.suite_revisions != actual_revisions:
        raise ValueError(
            "manifest suite revisions do not match the active executor catalog"
        )
    allowed_tracks: set[str] = set()
    if external_suites:
        if manifest.mode != "replay":
            raise ValueError("installed external suites support replay mode only")
        for suite in external_suites:
            allowed_tracks.update(suite.track_ids)
    else:
        for suite_id in manifest.suite_ids:
            suite = suites[suite_id]
            if manifest.mode not in suite.modes:
                raise ValueError(
                    f"suite {suite_id} does not support mode {manifest.mode}"
                )
            allowed_tracks.update(suite.track_ids)
    disallowed = sorted(set(manifest.track_ids) - allowed_tracks)
    if disallowed:
        raise ValueError(
            "tracks are not covered by selected suites: " + ", ".join(disallowed)
        )
    if manifest.mode == "live":
        runtime_catalog = get_catalog(
            generated_at=False,
            router_api_url=manifest.target.router_api_url,
            envoy_url=manifest.target.envoy_url,
            model_arms=manifest.target.model_arms,
        )
        runtime = next(
            target for target in runtime_catalog.targets if target.id == "runtime"
        )
        unsupported = sorted(set(manifest.track_ids) - set(runtime.track_ids))
        if unsupported:
            raise ValueError(
                "live target cannot produce selected tracks: " + ", ".join(unsupported)
            )


def _emit(
    store: LocalArtifactStore,
    run_id: str,
    sink: EventSink | None,
    event: WorkerEvent,
) -> None:
    store.append_event(run_id, event.model_dump(mode="json", exclude_none=True))
    if sink:
        sink(event)


def _run_model(
    manifest: RunManifest,
    *,
    status: str,
    progress: EvaluationRunProgress,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    error: str | None = None,
    evidence_level: str = "E0",
) -> EvaluationRun:
    return EvaluationRun(
        id=manifest.run_id,
        name=manifest.run_id,
        description="Evaluation suites: " + ", ".join(manifest.suite_ids),
        status=status,
        mode=manifest.mode,
        evidence_level=evidence_level,
        target_id=manifest.target.id,
        change_profile=manifest.change_profile,
        suite_ids=manifest.suite_ids,
        track_ids=manifest.track_ids,
        sample_limit=manifest.sample_limit,
        concurrency=manifest.concurrency,
        seed=manifest.seed,
        baseline_run_id=manifest.baseline_run_id,
        progress=progress,
        created_at=manifest.created_at,
        started_at=started_at,
        completed_at=completed_at,
        error=error,
    )


def _emit_track_events(
    manifest: RunManifest,
    store: LocalArtifactStore,
    sink: EventSink | None,
    records: list[ExecutionRecord],
) -> None:
    for index, track_id in enumerate(manifest.track_ids, 1):
        record_count = sum(row.track_id == track_id for row in records)
        event_progress = EvaluationRunProgress(
            percent=100 * index / len(manifest.track_ids),
            completed=index,
            total=len(manifest.track_ids),
            current_track_id=track_id,
            message=f"Collected {record_count} records",
        )
        _emit(
            store,
            manifest.run_id,
            sink,
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
    store: LocalArtifactStore,
    event_sink: EventSink | None,
    manage_control_state: bool,
) -> tuple[datetime, EvaluationRunProgress, ArtifactRef]:
    started = datetime.now(timezone.utc)
    progress = EvaluationRunProgress(
        percent=0,
        completed=0,
        total=len(manifest.track_ids),
        message="Starting evaluation",
    )
    run = _run_model(manifest, status="running", progress=progress, started_at=started)
    if manage_control_state:
        store.set_status(manifest.run_id, run)
        store.update_index(run)
    try:
        staged_manifest = RunManifest.model_validate(
            store.read_run_json(manifest.run_id, "run-manifest.json")
        )
        if staged_manifest != manifest:
            raise StoreError("staged run-manifest.json does not match worker input")
        manifest_ref = store.reference_run_artifact(
            manifest.run_id, "run-manifest.json"
        )
    except StoreError:
        manifest_ref = store.write_run_json(
            manifest.run_id, "run-manifest.json", manifest
        )
    _emit(
        store,
        manifest.run_id,
        event_sink,
        WorkerEvent(
            type="snapshot", message="Evaluation manifest validated", progress=progress
        ),
    )
    return started, progress, manifest_ref


def _execute_run(
    manifest: RunManifest,
    store: LocalArtifactStore,
    manifest_ref: ArtifactRef,
    started: datetime,
    event_sink: EventSink | None,
    manage_control_state: bool,
    suite_store: NormalizedSuiteStore | None,
    external_suites: tuple[BenchmarkSuiteManifest, ...],
) -> EvaluationReport:
    collected = collect_evidence(
        manifest,
        store,
        suite_store=suite_store,
        external_suites=external_suites,
    )
    records = collected.records
    _emit_track_events(manifest, store, event_sink, records)
    resolved = resolve_snapshot(
        manifest,
        collected.inputs,
        collected.visible_ref,
        collected.grading_ref,
        collected.fixture_ref,
        collected.discovered_entrypoints,
    )
    metrics = attach_confidence_intervals(
        compute_metrics(records), records, seed=manifest.seed
    )
    costs = build_costs(records)
    cost_accounted = all(
        amount is not None
        for amount in (
            costs.runtime.amount,
            costs.evaluation_overhead.amount,
            costs.capacity_tco.amount,
        )
    )
    completed = datetime.now(timezone.utc)
    gates = compute_gates(
        metrics,
        has_records=bool(records),
        cost_accounted=cost_accounted,
        change_profile=manifest.change_profile,
        evidence=GateEvidenceContext(
            manifest_validated=True,
            snapshots_complete=True,
            artifact_lineage_complete=True,
        ),
        records=records,
        evaluated_at=completed,
    )
    final_progress = EvaluationRunProgress(
        percent=100,
        completed=len(manifest.track_ids),
        total=len(manifest.track_ids),
        message="Evaluation completed",
    )
    completed_run = _run_model(
        manifest,
        status="completed",
        progress=final_progress,
        started_at=started,
        completed_at=completed,
        evidence_level=run_evidence_level(manifest.mode, manifest.track_ids, records),
    )
    report = finalize_report_bundle(
        manifest=manifest,
        store=store,
        manifest_ref=manifest_ref,
        inputs=collected.inputs,
        records=records,
        resolved=resolved,
        metrics=metrics,
        gates=gates,
        routing_traces=collected.routing_traces,
        run=completed_run,
        completed_at=completed,
        benchmark_revisions=collected.benchmark_revisions,
        private_lineage=collected.private_lineage,
    )
    if manage_control_state:
        store.set_status(manifest.run_id, completed_run)
        store.update_index(completed_run)
    _emit(
        store,
        manifest.run_id,
        event_sink,
        WorkerEvent(
            type="completed",
            message="Evaluation completed and artifacts were finalized",
            progress=final_progress,
            payload={"verdict": report.summary.verdict},
        ),
    )
    return report


def _record_failure(
    manifest: RunManifest,
    store: LocalArtifactStore,
    started: datetime,
    progress: EvaluationRunProgress,
    event_sink: EventSink | None,
    manage_control_state: bool,
    exc: Exception,
) -> None:
    failed_progress = EvaluationRunProgress(
        percent=progress.percent,
        completed=progress.completed,
        total=progress.total,
        message="Evaluation failed",
    )
    failed = _run_model(
        manifest,
        status="failed",
        progress=failed_progress,
        started_at=started,
        completed_at=datetime.now(timezone.utc),
        error=f"{type(exc).__name__}: {exc}",
    )
    if manage_control_state:
        store.set_status(manifest.run_id, failed)
        store.update_index(failed)
    _emit(
        store,
        manifest.run_id,
        event_sink,
        WorkerEvent(type="failed", message=failed.error or "Evaluation failed"),
    )


def run_evaluation(
    manifest: RunManifest,
    store: LocalArtifactStore,
    *,
    event_sink: EventSink | None = None,
    manage_control_state: bool = True,
    suite_store: NormalizedSuiteStore | None = None,
) -> EvaluationReport:
    validate_manifest(manifest, suite_store)
    external_suites = _external_suites(manifest, suite_store)
    try:
        existing = store.read_run_json(manifest.run_id, "report.json")
    except StoreError:
        existing = None
    if existing is not None:
        try:
            staged_manifest = RunManifest.model_validate(
                store.read_run_json(manifest.run_id, "run-manifest.json")
            )
        except (StoreError, ValidationError) as exc:
            raise StoreError(
                "existing report cannot be tied to a valid staged run manifest"
            ) from exc
        if staged_manifest != manifest:
            raise StoreError("existing report belongs to a different run manifest")
        return EvaluationReport.model_validate(existing)
    started, progress, manifest_ref = _start_run(
        manifest, store, event_sink, manage_control_state
    )
    try:
        return _execute_run(
            manifest,
            store=store,
            manifest_ref=manifest_ref,
            started=started,
            event_sink=event_sink,
            manage_control_state=manage_control_state,
            suite_store=suite_store,
            external_suites=external_suites,
        )
    except Exception as exc:
        _record_failure(
            manifest,
            store,
            started,
            progress,
            event_sink,
            manage_control_state,
            exc,
        )
        raise
