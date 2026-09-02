"""Validate and rebuild one immutable published evaluation report bundle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cli.evaluation.artifact_store_error import StoreError
from cli.evaluation.bundle import (
    artifact_media_type,
    checksum_bytes,
    failure_summary,
    private_receipt_names,
    public_artifacts,
    public_receipt_names,
)
from cli.evaluation.canonical import digest_value, sha256_digest, strict_json_loads
from cli.evaluation.capacity_profile import CapacityProfile, build_capacity_profile
from cli.evaluation.case_plan import require_complete_evidence_plan
from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.contract_primitives import ArtifactRef
from cli.evaluation.contracts import (
    CaseGrading,
    CaseVisible,
    ExecutorMetadata,
    GradingCaseSet,
    ResolvedRunSnapshot,
    RunManifest,
    VisibleCaseSet,
)
from cli.evaluation.evidence import ExecutionRecord, ReplayFixture, RoutingDiagnostic
from cli.evaluation.execution_contract import (
    NormalizedSuiteIdentities,
    PublishedLineage,
    require_discovered_entrypoint_binding,
    require_normalized_identity_binding,
)
from cli.evaluation.executor_contracts import (
    ExecutorContract,
    executor_is_mom_cohort_replay,
)
from cli.evaluation.report_builder import build_worker_report_draft
from cli.evaluation.reporting import (
    EvaluationArtifact,
    EvaluationGate,
    EvaluationMetric,
    EvaluationProvenance,
)
from cli.evaluation.routing_trace import require_routing_trace_binding
from cli.evaluation.runtime_factors import runtime_factors
from cli.evaluation.store import ArtifactStore
from cli.evaluation.worker_report import (
    WorkerReportDraft,
    require_manifest_run_state,
)


@dataclass(frozen=True, slots=True)
class ValidatedReportBundle:
    report: WorkerReportDraft
    records: tuple[ExecutionRecord, ...]


def _artifact_reference(name: str, data: bytes) -> ArtifactRef:
    return ArtifactRef(
        digest=sha256_digest(data),
        media_type=artifact_media_type(name),
        size_bytes=len(data),
    )


def _require_executor_contract(
    manifest: RunManifest,
    executor: ExecutorContract,
) -> None:
    expected_executor_id = next(iter(manifest.suite_executors.values()))
    has_replay_mixture = (
        manifest.mode == "replay" and manifest.target.mixture is not None
    )
    if (
        executor.id != expected_executor_id
        or executor.mode != manifest.mode
        or not set(manifest.track_ids).issubset(executor.track_ids)
        or has_replay_mixture != executor_is_mom_cohort_replay(executor)
    ):
        raise StoreError("executor contract does not match the published run manifest")


def _strict_object(
    data: bytes,
    expected_fields: frozenset[str],
    name: str,
) -> dict[str, Any]:
    try:
        value = strict_json_loads(data)
    except (TypeError, ValueError) as exc:
        raise StoreError(f"{name} contains invalid JSON") from exc
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise StoreError(f"{name} has an invalid root contract")
    return value


def _jsonl_values(data: bytes, name: str) -> tuple[Any, ...]:
    if not data:
        return ()
    if not data.endswith(b"\n"):
        raise StoreError(f"{name} is not newline framed")
    try:
        return tuple(strict_json_loads(line) for line in data.splitlines())
    except (TypeError, ValueError) as exc:
        raise StoreError(f"{name} contains invalid JSON") from exc


def _decode_inputs(
    files: Mapping[str, bytes],
) -> tuple[VisibleCaseSet, GradingCaseSet, list[ExecutionRecord]]:
    try:
        visible = VisibleCaseSet(
            cases=tuple(
                CaseVisible.model_validate(value)
                for value in _jsonl_values(files["cases.jsonl"], "cases.jsonl")
            )
        )
        grading = GradingCaseSet(
            cases=tuple(
                CaseGrading.model_validate(value)
                for value in _jsonl_values(
                    files["grading-cases.jsonl"],
                    "grading-cases.jsonl",
                )
            )
        )
        records = [
            ExecutionRecord.model_validate(value)
            for value in _jsonl_values(files["records.jsonl"], "records.jsonl")
        ]
    except (TypeError, ValueError) as exc:
        raise StoreError("published evidence artifacts are invalid") from exc
    if tuple(case.id for case in visible.cases) != tuple(
        case.case_id for case in grading.cases
    ):
        raise StoreError("visible and grading case order does not match")
    return visible, grading, records


def _decode_report_facts(
    files: Mapping[str, bytes],
) -> tuple[
    tuple[EvaluationMetric, ...],
    tuple[EvaluationGate, ...],
    EvaluationProvenance,
]:
    try:
        metric_payload = _strict_object(
            files["metrics.json"],
            frozenset({"schema_version", "metrics"}),
            "metrics.json",
        )
        gate_payload = _strict_object(
            files["gates.json"],
            frozenset({"schema_version", "gates"}),
            "gates.json",
        )
        if (
            metric_payload["schema_version"] != SCHEMA_VERSION
            or gate_payload["schema_version"] != SCHEMA_VERSION
            or not isinstance(metric_payload["metrics"], list)
            or not isinstance(gate_payload["gates"], list)
        ):
            raise StoreError("report fact envelope version is invalid")
        metrics = tuple(
            EvaluationMetric.model_validate(value)
            for value in metric_payload["metrics"]
        )
        gates = tuple(
            EvaluationGate.model_validate(value) for value in gate_payload["gates"]
        )
        provenance = EvaluationProvenance.model_validate(
            strict_json_loads(files["provenance.json"])
        )
    except (TypeError, ValueError) as exc:
        raise StoreError("published report facts are invalid") from exc
    return metrics, gates, provenance


def _decode_lineage(
    data: bytes,
) -> PublishedLineage:
    try:
        return PublishedLineage.model_validate(strict_json_loads(data))
    except (TypeError, ValueError) as exc:
        raise StoreError("lineage.json has an invalid contract") from exc


def _validate_replay_fixture_lineage(
    resolved: ResolvedRunSnapshot,
    store: ArtifactStore,
    visible: VisibleCaseSet,
) -> None:
    fixture_ref = resolved.fixture_ref
    if fixture_ref is None:
        return
    if fixture_ref.media_type != "application/json":
        raise StoreError("lineage replay fixture has an invalid media type")
    try:
        fixture = ReplayFixture.model_validate(store.read_json(fixture_ref))
    except (TypeError, ValueError) as exc:
        raise StoreError("lineage replay fixture is invalid") from exc
    if tuple(case.case_id for case in fixture.cases) != tuple(
        case.id for case in visible.cases
    ):
        raise StoreError("lineage replay fixture does not match planned cases")


def _validate_lineage(
    manifest: RunManifest,
    executor: ExecutorContract,
    store: ArtifactStore,
    visible: VisibleCaseSet,
    grading: GradingCaseSet,
    records: list[ExecutionRecord],
    resolved: ResolvedRunSnapshot,
    identities: NormalizedSuiteIdentities | None,
) -> None:
    expected_executors = tuple(
        ExecutorMetadata(track_id=track_id, executor_id=executor.id, mode=manifest.mode)
        for track_id in manifest.track_ids
    )
    if (
        resolved.manifest_digest != manifest.manifest_digest
        or resolved.executors != expected_executors
        or resolved.environment.target_id != manifest.target.id
        or resolved.policy.recipe_digest != manifest.policy_snapshot_digest
    ):
        raise StoreError("lineage does not bind the immutable run manifest")
    visible_ref = resolved.workload.visible_cases
    grading_ref = resolved.workload.grading_cases
    if (
        visible_ref.media_type != "application/json"
        or grading_ref.media_type != "application/json"
    ):
        raise StoreError("lineage workload snapshots have an invalid media type")
    try:
        visible_snapshot = VisibleCaseSet.model_validate(store.read_json(visible_ref))
        grading_snapshot = GradingCaseSet.model_validate(store.read_json(grading_ref))
    except (TypeError, ValueError) as exc:
        raise StoreError("lineage workload snapshot is invalid") from exc
    if visible_snapshot != visible or grading_snapshot != grading:
        raise StoreError("lineage workload snapshots do not match report evidence")
    workload_suffix = digest_value(
        {
            "visible_cases": visible_ref.digest,
            "grading_cases": grading_ref.digest,
        }
    ).removeprefix("sha256:")[:16]
    if resolved.workload.id != f"workload-{workload_suffix}":
        raise StoreError("lineage workload identity is invalid")
    _validate_replay_fixture_lineage(resolved, store, visible)
    try:
        require_normalized_identity_binding(
            manifest,
            visible,
            records,
            identities,
            required=executor.normalized_suite,
            recorded_import=executor.recorded_normalized_import,
        )
    except ValueError as exc:
        raise StoreError("lineage normalized identity binding is invalid") from exc
    if (resolved.fixture_ref is not None) != executor.requires_fixture_ref:
        raise StoreError("lineage replay fixture ownership is invalid")
    try:
        require_discovered_entrypoint_binding(
            manifest,
            resolved.discovered_entrypoints,
        )
    except ValueError as exc:
        raise StoreError("lineage runtime entrypoint binding is invalid") from exc
    if manifest.target.mixture is not None:
        factors = runtime_factors(manifest)
        if (
            resolved.policy != factors.policy
            or resolved.arms != factors.arms
            or resolved.pool != factors.pool
            or resolved.binding != factors.binding
            or resolved.environment != factors.environment
        ):
            raise StoreError("lineage Mixture factors do not match the frozen target")


def _validate_provenance(
    manifest: RunManifest,
    report: WorkerReportDraft,
    resolved: ResolvedRunSnapshot,
    provenance: EvaluationProvenance,
) -> None:
    mixture = manifest.target.mixture
    expected_policy = (
        mixture.recipe_digest if mixture is not None else digest_value(resolved.policy)
    )
    expected_binding = (
        mixture.binding_digest
        if mixture is not None
        else digest_value(resolved.binding)
    )
    expected_pool = (
        mixture.pool_digest
        if mixture is not None
        else digest_value({"pool": resolved.pool, "arms": resolved.arms})
    )
    if (
        provenance.generated_at != report.run.completed_at
        or provenance.code_revision != manifest.code_revision
        or provenance.benchmark_revisions != manifest.suite_revisions
        or provenance.workload_snapshot_digest != digest_value(resolved.workload)
        or provenance.policy_snapshot_digest != expected_policy
        or provenance.binding_snapshot_digest != expected_binding
        or provenance.pool_snapshot_digest != expected_pool
        or provenance.environment_snapshot_digest != digest_value(resolved.environment)
        or provenance.target_id != manifest.target.id
        or provenance.seed != manifest.seed
        or provenance.redaction_policy != manifest.redaction_policy
    ):
        raise StoreError("provenance does not match the immutable run lineage")


def _validate_optional_artifacts(
    manifest: RunManifest,
    files: Mapping[str, bytes],
    visible: VisibleCaseSet,
    records: list[ExecutionRecord],
) -> None:
    routing_data = files.get("routing-traces.jsonl")
    traces: tuple[RoutingDiagnostic, ...] = ()
    if routing_data is not None:
        try:
            traces = tuple(
                RoutingDiagnostic.model_validate(value)
                for value in _jsonl_values(routing_data, "routing-traces.jsonl")
            )
        except (TypeError, ValueError) as exc:
            raise StoreError("routing traces are invalid") from exc
        if not traces:
            raise StoreError("routing traces artifact cannot be empty")
    try:
        require_routing_trace_binding(manifest, visible, records, traces)
    except ValueError as exc:
        raise StoreError("routing trace binding is invalid") from exc
    capacity_data = files.get("capacity-profile.json")
    capacity_required = manifest.mode == "live" and "capacity" in manifest.track_ids
    if (capacity_data is not None) != capacity_required:
        raise StoreError("capacity profile presence does not match the run contract")
    if capacity_data is not None:
        if manifest.capacity_slo is None or manifest.capacity_load_protocol is None:
            raise StoreError("capacity profile is missing its frozen load contract")
        try:
            capacity = CapacityProfile.model_validate(strict_json_loads(capacity_data))
            expected = build_capacity_profile(
                records,
                manifest.capacity_slo,
                manifest.capacity_load_protocol,
            )
        except (TypeError, ValueError) as exc:
            raise StoreError("capacity profile is invalid") from exc
        if capacity != expected:
            raise StoreError("capacity profile does not match recorded load evidence")


def _validate_receipts_and_cas(
    store: ArtifactStore,
    files: Mapping[str, bytes],
) -> tuple[EvaluationArtifact, ...]:
    references = {name: _artifact_reference(name, data) for name, data in files.items()}
    for name, reference in references.items():
        if store.read_bytes(reference) != files[name]:
            raise StoreError(f"CAS object does not match published artifact {name}")
    public_names = public_receipt_names(files)
    public_rows = [(name, references[name]) for name in public_names]
    if files["checksums.sha256"] != checksum_bytes(public_rows):
        raise StoreError("public checksum receipt does not match report artifacts")
    private_names = private_receipt_names(files)
    private_rows = [(name, references[name]) for name in private_names]
    if files["private-checksums.sha256"] != checksum_bytes(private_rows):
        raise StoreError("private checksum receipt does not match report artifacts")
    report_rows = [
        *private_rows,
        ("private-checksums.sha256", references["private-checksums.sha256"]),
    ]
    return public_artifacts(report_rows)


def load_published_report_bundle(
    manifest: RunManifest,
    store: ArtifactStore,
    executor: ExecutorContract,
) -> ValidatedReportBundle | None:
    """Return only a fully coherent, CAS-backed, reproducible report draft."""

    _require_executor_contract(manifest, executor)
    files = store.snapshot_published_report_bundle(manifest.run_id)
    if files is None:
        return None
    try:
        staged = RunManifest.model_validate(
            strict_json_loads(files["run-manifest.json"])
        )
        report = WorkerReportDraft.model_validate(
            strict_json_loads(files["report.json"])
        )
    except (TypeError, ValueError) as exc:
        raise StoreError("published report or manifest contract is invalid") from exc
    if staged != manifest:
        raise StoreError("published report belongs to another run manifest")
    try:
        require_manifest_run_state(manifest, report.run)
    except ValueError as exc:
        raise StoreError(str(exc)) from exc
    visible, grading, records = _decode_inputs(files)
    try:
        planned_case_ids = require_complete_evidence_plan(manifest, visible, records)
    except ValueError as exc:
        raise StoreError("published evidence violates the immutable case plan") from exc
    failure_payload = _strict_object(
        files["failure-summary.json"],
        frozenset(
            {"schema_version", "total_records", "failed", "unavailable", "by_track"}
        ),
        "failure-summary.json",
    )
    if failure_payload != failure_summary(records):
        raise StoreError("failure summary does not match published records")
    metrics, gates, provenance = _decode_report_facts(files)
    lineage = _decode_lineage(files["lineage.json"])
    _validate_lineage(
        manifest,
        executor,
        store,
        visible,
        grading,
        records,
        lineage.resolved_snapshot,
        lineage.normalized_suite_identities,
    )
    _validate_provenance(
        manifest,
        report,
        lineage.resolved_snapshot,
        provenance,
    )
    _validate_optional_artifacts(manifest, files, visible, records)
    expected_artifacts = _validate_receipts_and_cas(store, files)
    try:
        expected = build_worker_report_draft(
            manifest=manifest,
            executor=executor,
            run=report.run,
            records=records,
            metrics=list(metrics),
            gates=list(gates),
            provenance=provenance,
            artifacts=expected_artifacts,
            planned_case_ids=planned_case_ids,
        )
    except (TypeError, ValueError) as exc:
        raise StoreError("published report evidence cannot be reduced") from exc
    if report != expected:
        raise StoreError("published report does not match its typed evidence bundle")
    return ValidatedReportBundle(report=report, records=tuple(records))
