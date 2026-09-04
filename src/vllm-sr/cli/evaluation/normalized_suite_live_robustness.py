"""Qualify exact installed declared-shift relations against live broker records.

This module never executes upstream code and never imports historical router
decisions. The installed normalized relation is an immutable input corpus;
current source/target actions come only from this run's broker-bound records.
"""

from __future__ import annotations

from collections import defaultdict
from typing import cast

from cli.evaluation.canonical import digest_value
from cli.evaluation.contracts import CaseVisible
from cli.evaluation.errors import SuiteStoreError
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_source_ids import (
    DECLARED_SHIFT_LIVE_EVIDENCE_SOURCE_ID,
)
from cli.evaluation.method_evidence import RobustnessMethodEvidence
from cli.evaluation.normalized_suite_inputs import SelectedCase
from cli.evaluation.suite_contract import (
    BenchmarkSuiteManifest,
    NormalizedPerturbation,
)
from cli.evaluation.suite_store import NormalizedSuiteStore
from cli.evaluation.target_arm_resolution import resolve_target_arm_id
from cli.evaluation.target_contracts import EvaluationTargetArm

DECLARED_SHIFT_LIVE_METHOD_ID = "declared-shift.server-live.v1"


def _qualified_manifest_pairs(
    store: NormalizedSuiteStore,
    manifest: BenchmarkSuiteManifest,
) -> tuple[NormalizedPerturbation, ...] | None:
    artifact = manifest.artifacts.perturbations
    if (
        artifact is None
        or not manifest.qualification_receipt.qualification.parser_verified
        or "routing" not in manifest.track_ids
    ):
        return None
    pairs = tuple(
        cast(NormalizedPerturbation, row)
        for row in store.load_jsonl(manifest.id, "perturbations")
    )
    if not pairs:
        return None
    native_count = len(pairs)
    if any(pair.native_pair_count != native_count for pair in pairs):
        raise SuiteStoreError("installed declared-shift native pair count drifted")
    pair_ids = {pair.pair_id for pair in pairs}
    coordinates = {(pair.source_case_id, pair.perturbed_case_id) for pair in pairs}
    case_ids = [
        case_id
        for pair in pairs
        for case_id in (pair.source_case_id, pair.perturbed_case_id)
    ]
    if (
        len(pair_ids) != native_count
        or len(coordinates) != native_count
        or len(set(case_ids)) != native_count * 2
    ):
        raise SuiteStoreError(
            "installed declared-shift pairs reuse an identity or coordinate"
        )
    visible: dict[str, CaseVisible] = {}
    for row in store.load_jsonl(manifest.id, "visible_cases"):
        case = cast(CaseVisible, row)
        if case.id in visible:
            raise SuiteStoreError(
                "installed declared-shift visible identity is duplicated"
            )
        visible[case.id] = case
    if len(visible) != manifest.case_count:
        raise SuiteStoreError("installed declared-shift visible cohort is incomplete")
    if any(
        case_id not in visible or "routing" not in visible[case_id].track_ids
        for case_id in case_ids
    ):
        raise SuiteStoreError(
            "installed declared-shift pair references a non-routing case"
        )
    return pairs


def declared_shift_source_is_eligible(
    store: NormalizedSuiteStore,
    manifest: BenchmarkSuiteManifest,
) -> bool:
    """Return whether an installed suite can advertise the live G4 method."""

    return _qualified_manifest_pairs(store, manifest) is not None


def _routing_pair_is_complete(
    source: ExecutionRecord,
    target: ExecutionRecord,
) -> bool:
    return all(
        row.status == "succeeded"
        and row.success is True
        and row.selected_arm_id is not None
        and row.broker_receipt is not None
        for row in (source, target)
    )


def _manifest_pair_methods(
    *,
    manifest: BenchmarkSuiteManifest,
    pairs: tuple[NormalizedPerturbation, ...],
    selected_by_source: dict[tuple[str, str], SelectedCase],
    routing: dict[str, list[ExecutionRecord]],
    arms: tuple[EvaluationTargetArm, ...],
    receipt_digest: str,
    artifact_digest: str,
) -> tuple[dict[str, RobustnessMethodEvidence], set[str]] | None:
    methods: dict[str, RobustnessMethodEvidence] = {}
    qualified_records: set[str] = set()
    for pair in pairs:
        source_case = selected_by_source.get((manifest.id, pair.source_case_id))
        target_case = selected_by_source.get((manifest.id, pair.perturbed_case_id))
        if source_case is None or target_case is None:
            return None
        source_rows = routing.get(source_case.visible.id, [])
        target_rows = routing.get(target_case.visible.id, [])
        if len(source_rows) != 1 or len(target_rows) != 1:
            return None
        source, target = source_rows[0], target_rows[0]
        if not _routing_pair_is_complete(source, target):
            return None
        expected_action = resolve_target_arm_id(pair.expected_action_id, arms)
        if pair.relation == "expected_change" and expected_action is None:
            return None
        if target.case_id in methods:
            return None
        methods[target.case_id] = RobustnessMethodEvidence(
            method_id=DECLARED_SHIFT_LIVE_METHOD_ID,
            suite_id=manifest.id,
            suite_revision=manifest.revision,
            qualification_receipt_digest=receipt_digest,
            perturbation_artifact_digest=artifact_digest,
            pair_id=pair.pair_id,
            source_case_id=source.case_id,
            target_case_id=target.case_id,
            shift_type="paraphrase",
            relation=pair.relation,
            source_action_id=source.selected_arm_id,
            expected_action_id=expected_action,
            slice_ids=pair.slice_ids,
            native_pair_count=pair.native_pair_count,
            source_record_digest=pair.source_record_digest,
        )
        qualified_records.update((source.case_id, target.case_id))
    return methods, qualified_records


def attach_live_declared_shift_evidence(
    *,
    records: list[ExecutionRecord],
    selected: tuple[SelectedCase, ...],
    manifests: tuple[BenchmarkSuiteManifest, ...],
    store: NormalizedSuiteStore,
    arms: tuple[EvaluationTargetArm, ...],
) -> list[ExecutionRecord]:
    """Attach E4 candidates only for a complete, exact live pair matrix.

    Returning the original records means the method is unavailable. Structural
    corruption remains the suite store's responsibility; scientific
    incompleteness is deliberately a fail-closed E0 result.
    """

    routing: dict[str, list[ExecutionRecord]] = defaultdict(list)
    for record in records:
        if record.track_id == "routing":
            routing[record.case_id].append(record)
    selected_by_source = {
        (case.manifest.id, case.source_visible.id): case for case in selected
    }
    methods_by_target: dict[str, RobustnessMethodEvidence] = {}
    qualified_pair_records: set[str] = set()
    for manifest in manifests:
        pairs = _qualified_manifest_pairs(store, manifest)
        if pairs is None:
            if "routing" in manifest.track_ids:
                return records
            continue
        receipt_digest = digest_value(manifest.qualification_receipt)
        artifact = manifest.artifacts.perturbations
        assert artifact is not None
        resolved = _manifest_pair_methods(
            manifest=manifest,
            pairs=pairs,
            selected_by_source=selected_by_source,
            routing=routing,
            arms=arms,
            receipt_digest=receipt_digest,
            artifact_digest=artifact.digest,
        )
        if resolved is None:
            return records
        manifest_methods, manifest_records = resolved
        if set(methods_by_target).intersection(manifest_methods):
            return records
        methods_by_target.update(manifest_methods)
        qualified_pair_records.update(manifest_records)
    if not methods_by_target:
        return records
    if set(routing) != qualified_pair_records or any(
        len(rows) != 1 for rows in routing.values()
    ):
        return records
    # Every declared source/target coordinate has one successful receipt. The
    # server independently validates each receipt and repeats this reduction.
    return [
        (
            record.model_copy(
                update={
                    "robustness": methods_by_target.get(record.case_id),
                    "evidence_kind": DECLARED_SHIFT_LIVE_EVIDENCE_SOURCE_ID,
                }
            )
            if record.track_id == "routing" and record.case_id in qualified_pair_records
            else record
        )
        for record in records
    ]


def declared_shift_gate_is_complete(records: list[ExecutionRecord]) -> bool:
    """Worker-side proposal only; the dashboard server is authoritative."""

    methods = [row.robustness for row in records if row.robustness is not None]
    if not methods:
        return False
    native_counts = {method.native_pair_count for method in methods}
    return (
        len(native_counts) == 1
        and len(methods) == next(iter(native_counts))
        and all(
            method.method_id == DECLARED_SHIFT_LIVE_METHOD_ID
            and row.evidence_kind == DECLARED_SHIFT_LIVE_EVIDENCE_SOURCE_ID
            and row.broker_receipt is not None
            for row in records
            for method in ([row.robustness] if row.robustness is not None else [])
        )
    )
