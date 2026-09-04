"""Explicit control-state ownership for evaluation execution contexts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from cli.evaluation.contract_primitives import ArtifactRef
from cli.evaluation.contracts import RunManifest
from cli.evaluation.errors import StoreError
from cli.evaluation.store import LocalArtifactStore, WorkerArtifactStore
from cli.evaluation.worker_report import (
    WorkerEvent,
    WorkerReportDraft,
    WorkerRunState,
)

EventSink = Callable[[WorkerEvent], None]


class RunOwnership(Protocol):
    """Own only the mutable control-plane actions for one execution context."""

    def bind_manifest(
        self,
        manifest: RunManifest,
    ) -> ArtifactRef: ...

    def record_state(
        self,
        run: WorkerRunState,
    ) -> None: ...

    def emit(
        self,
        run_id: str,
        event: WorkerEvent,
    ) -> None: ...

    def reconcile_completed(
        self,
        report: WorkerReportDraft,
        event: WorkerEvent,
    ) -> None: ...


def _validate_worker_staged_manifest(
    store: WorkerArtifactStore,
    manifest: RunManifest,
) -> None:
    try:
        staged = RunManifest.model_validate(
            store.read_run_json(manifest.run_id, "run-manifest.json")
        )
    except (StoreError, TypeError, ValueError) as exc:
        raise StoreError(
            "worker requires a valid server-staged run-manifest.json"
        ) from exc
    if staged != manifest:
        raise StoreError("staged run-manifest.json does not match worker input")


def _require_staged_manifest(
    store: WorkerArtifactStore,
    manifest: RunManifest,
) -> ArtifactRef:
    _validate_worker_staged_manifest(store, manifest)
    return store.reference_staged_manifest(manifest.run_id)


def _send(sink: EventSink | None, event: WorkerEvent) -> None:
    if sink is not None:
        sink(event)


@dataclass(frozen=True, slots=True)
class StandaloneRunOwnership:
    """Standalone owns manifest, status, and durable event publication."""

    store: LocalArtifactStore
    event_sink: EventSink | None = None

    def bind_manifest(
        self,
        manifest: RunManifest,
    ) -> ArtifactRef:
        return self.store.stage_run_manifest(manifest)

    def record_state(
        self,
        run: WorkerRunState,
    ) -> None:
        self.store.write_run_status(run)

    def emit(
        self,
        run_id: str,
        event: WorkerEvent,
    ) -> None:
        self.store.append_event(run_id, event)
        _send(self.event_sink, event)

    def reconcile_completed(
        self,
        report: WorkerReportDraft,
        event: WorkerEvent,
    ) -> None:
        self.store.write_run_status(report.run)
        if self.store.append_event_if_changed(report.run.id, event):
            _send(self.event_sink, event)


@dataclass(frozen=True, slots=True)
class WorkerRunOwnership:
    """Dashboard worker writes evidence and streams events, never control state."""

    store: WorkerArtifactStore
    event_sink: EventSink | None = None

    def bind_manifest(
        self,
        manifest: RunManifest,
    ) -> ArtifactRef:
        return _require_staged_manifest(self.store, manifest)

    def record_state(
        self,
        run: WorkerRunState,
    ) -> None:
        """Dashboard owns mutable worker status publication."""

        del run

    def emit(
        self,
        run_id: str,
        event: WorkerEvent,
    ) -> None:
        del run_id
        _send(self.event_sink, event)

    def reconcile_completed(
        self,
        report: WorkerReportDraft,
        event: WorkerEvent,
    ) -> None:
        """Dashboard seals and publishes the terminal server state."""

        del report, event
