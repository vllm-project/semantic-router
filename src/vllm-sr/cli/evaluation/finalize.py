"""Finalize immutable evidence files and one server-sealable worker draft."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from cli.evaluation.bundle import (
    ReportBundleWriter,
    checksum_bytes,
    failure_summary,
    private_receipt_names,
    public_artifacts,
    public_receipt_names,
)
from cli.evaluation.canonical import digest_value
from cli.evaluation.capacity_profile import CapacityProfile
from cli.evaluation.case_plan import planned_case_ids_by_track
from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.contract_primitives import ArtifactRef
from cli.evaluation.contracts import ResolvedRunSnapshot, RunManifest
from cli.evaluation.evidence import ExecutionRecord, RoutingDiagnostic
from cli.evaluation.execution_contract import (
    EvaluationInputs,
    NormalizedSuiteIdentities,
    PublishedLineage,
)
from cli.evaluation.executor_contracts import ExecutorContract
from cli.evaluation.report_builder import (
    build_worker_report_draft,
    select_report_metrics,
)
from cli.evaluation.reporting import (
    EvaluationGate,
    EvaluationMetric,
    EvaluationProvenance,
)
from cli.evaluation.store import ArtifactStore
from cli.evaluation.worker_report import WorkerReportDraft, WorkerRunState


def _provenance(
    manifest: RunManifest,
    resolved: ResolvedRunSnapshot,
    completed_at: datetime,
    benchmark_revisions: Mapping[str, str],
) -> EvaluationProvenance:
    mixture = manifest.target.mixture
    return EvaluationProvenance(
        schema_version=SCHEMA_VERSION,
        generated_at=completed_at,
        code_revision=manifest.code_revision,
        benchmark_revisions=benchmark_revisions,
        workload_snapshot_digest=digest_value(resolved.workload),
        policy_snapshot_digest=(
            mixture.recipe_digest
            if mixture is not None
            else digest_value(resolved.policy)
        ),
        binding_snapshot_digest=(
            mixture.binding_digest
            if mixture is not None
            else digest_value(resolved.binding)
        ),
        pool_snapshot_digest=(
            mixture.pool_digest
            if mixture is not None
            else digest_value({"pool": resolved.pool, "arms": resolved.arms})
        ),
        environment_snapshot_digest=digest_value(resolved.environment),
        target_id=manifest.target.id,
        seed=manifest.seed,
        redaction_policy=manifest.redaction_policy,
    )


def _core_artifacts(
    manifest: RunManifest,
    transaction: ReportBundleWriter,
    manifest_ref: ArtifactRef,
    inputs: EvaluationInputs,
    records: list[ExecutionRecord],
    resolved: ResolvedRunSnapshot,
    metrics: list[EvaluationMetric],
    gates: list[EvaluationGate],
    provenance: EvaluationProvenance,
    private_identity_map: NormalizedSuiteIdentities | None,
) -> list[tuple[str, ArtifactRef]]:
    lineage = PublishedLineage(
        schema_version=SCHEMA_VERSION,
        resolved_snapshot=resolved,
        normalized_suite_identities=private_identity_map,
    )
    lineage_value = lineage.model_dump(mode="json", exclude_none=True)
    lineage_value.setdefault("normalized_suite_identities", None)
    resolved_value = resolved.model_dump(mode="json", exclude_none=True)
    resolved_value.setdefault("fixture_ref", None)
    lineage_value["resolved_snapshot"] = resolved_value
    return [
        ("run-manifest.json", manifest_ref),
        (
            "cases.jsonl",
            transaction.write_jsonl("cases.jsonl", inputs.visible.cases),
        ),
        (
            "records.jsonl",
            transaction.write_jsonl("records.jsonl", records),
        ),
        (
            "grading-cases.jsonl",
            transaction.write_jsonl(
                "grading-cases.jsonl",
                inputs.grading.cases,
            ),
        ),
        (
            "metrics.json",
            transaction.write_json(
                "metrics.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "metrics": [
                        metric.model_dump(mode="json", exclude_none=False)
                        for metric in metrics
                    ],
                },
            ),
        ),
        (
            "gates.json",
            transaction.write_json(
                "gates.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "gates": [
                        gate.model_dump(mode="json", exclude_none=False)
                        for gate in gates
                    ],
                },
            ),
        ),
        (
            "lineage.json",
            transaction.write_json("lineage.json", lineage_value),
        ),
        (
            "provenance.json",
            transaction.write_json("provenance.json", provenance),
        ),
        (
            "failure-summary.json",
            transaction.write_json(
                "failure-summary.json",
                failure_summary(records),
            ),
        ),
    ]


def _live_artifacts(
    manifest: RunManifest,
    transaction: ReportBundleWriter,
    records: list[ExecutionRecord],
    routing_traces: tuple[RoutingDiagnostic, ...],
    capacity_profile: CapacityProfile | None,
) -> list[tuple[str, ArtifactRef]]:
    if manifest.mode != "live":
        return []
    rows: list[tuple[str, ArtifactRef]] = []
    if routing_traces:
        rows.append(
            (
                "routing-traces.jsonl",
                transaction.write_jsonl("routing-traces.jsonl", routing_traces),
            )
        )
    capacity_rows = [row for row in records if row.track_id == "capacity"]
    if capacity_rows and capacity_profile is None:
        raise ValueError("live capacity records require a typed SLO profile")
    if capacity_profile is not None:
        rows.append(
            (
                "capacity-profile.json",
                transaction.write_json(
                    "capacity-profile.json",
                    capacity_profile.model_dump(mode="json", exclude_none=False),
                ),
            )
        )
    return rows


def finalize_report_bundle(
    *,
    manifest: RunManifest,
    executor: ExecutorContract,
    store: ArtifactStore,
    manifest_ref: ArtifactRef,
    inputs: EvaluationInputs,
    records: list[ExecutionRecord],
    resolved: ResolvedRunSnapshot,
    metrics: list[EvaluationMetric],
    gates: list[EvaluationGate],
    routing_traces: tuple[RoutingDiagnostic, ...],
    capacity_profile: CapacityProfile | None,
    run: WorkerRunState,
    completed_at: datetime,
    benchmark_revisions: Mapping[str, str],
    private_identity_map: NormalizedSuiteIdentities | None = None,
) -> WorkerReportDraft:
    """Write the exact report.json draft accepted by the Dashboard worker envelope."""

    selected_metrics = select_report_metrics(manifest, metrics)
    provenance = _provenance(manifest, resolved, completed_at, benchmark_revisions)
    with store.report_bundle_transaction(manifest) as transaction:
        artifact_rows = _core_artifacts(
            manifest,
            transaction,
            manifest_ref,
            inputs,
            records,
            resolved,
            selected_metrics,
            gates,
            provenance,
            private_identity_map,
        )
        artifact_rows.extend(
            _live_artifacts(
                manifest,
                transaction,
                records,
                routing_traces,
                capacity_profile,
            )
        )
        planned_case_ids = planned_case_ids_by_track(
            inputs.visible,
            manifest.track_ids,
        )
        report_options = {
            "manifest": manifest,
            "executor": executor,
            "run": run,
            "records": records,
            "metrics": selected_metrics,
            "gates": gates,
            "provenance": provenance,
            "planned_case_ids": planned_case_ids,
        }
        artifact_references = dict(artifact_rows)
        public_checksum_rows = [
            (name, artifact_references[name])
            for name in public_receipt_names(artifact_references)
        ]
        checksum_ref = transaction.write_bytes(
            "checksums.sha256",
            checksum_bytes(public_checksum_rows),
        )
        artifact_rows.append(("checksums.sha256", checksum_ref))
        artifact_references["checksums.sha256"] = checksum_ref
        private_checksum_ref = transaction.write_bytes(
            "private-checksums.sha256",
            checksum_bytes(
                [
                    (name, artifact_references[name])
                    for name in private_receipt_names(artifact_references)
                ]
            ),
        )
        artifact_rows.append(("private-checksums.sha256", private_checksum_ref))
        report = build_worker_report_draft(
            **report_options,
            artifacts=public_artifacts(artifact_rows),
        )
        transaction.write_json(
            "report.json",
            report.model_dump(mode="json", exclude_none=False),
        )
        transaction.commit()
    return report
