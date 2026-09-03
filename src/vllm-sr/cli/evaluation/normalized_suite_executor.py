"""Orchestrate data-only replay of installed normalized suites."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.execution_contract import EvaluationInputs
from cli.evaluation.method_contract_v2 import (
    R2_COMPOUND_MODEL_BUDGET_PLUGIN,
    EvaluationMethodPlugin,
)
from cli.evaluation.method_registry_v2 import method_plugin_for_benchmark
from cli.evaluation.normalized_suite_behavior_records import (
    agentic_records,
    multimodal_records,
    preference_records,
)
from cli.evaluation.normalized_suite_inputs import (
    SelectedCase,
    SuiteEvidence,
    build_inputs,
    load_selected_cases,
)
from cli.evaluation.normalized_suite_loading import load_suite_evidence
from cli.evaluation.normalized_suite_operations_records import (
    capacity_records,
    safety_records,
)
from cli.evaluation.normalized_suite_r2_records import r2_compound_model_budget_records
from cli.evaluation.normalized_suite_routing_records import (
    joint_records,
    model_pool_records,
    routing_records,
)
from cli.evaluation.suite_contract import BenchmarkSuiteManifest
from cli.evaluation.suite_store import NormalizedSuiteStore
from cli.evaluation.suite_store_error import SuiteStoreError

TrackRecordBuilder = Callable[[SelectedCase, SuiteEvidence], list[ExecutionRecord]]


@dataclass(frozen=True)
class NormalizedMethodRecordBuilder:
    """Bind one exact v2 method contract to one normalized record reducer."""

    method: EvaluationMethodPlugin
    track_id: str
    build_records: TrackRecordBuilder


TRACK_RECORD_BUILDERS: Mapping[str, TrackRecordBuilder] = MappingProxyType(
    {
        "routing": routing_records,
        "model_pool": model_pool_records,
        "joint": joint_records,
        "agentic": agentic_records,
        "multimodal": multimodal_records,
        "preference": preference_records,
        "safety": safety_records,
        "capacity": capacity_records,
    }
)

METHOD_RECORD_BUILDERS = (
    NormalizedMethodRecordBuilder(
        method=R2_COMPOUND_MODEL_BUDGET_PLUGIN,
        track_id="model_pool",
        build_records=r2_compound_model_budget_records,
    ),
)


@dataclass(frozen=True)
class NormalizedSuiteExecution:
    inputs: EvaluationInputs
    records: list[ExecutionRecord]


def _validated_track_builders(
    track_ids: tuple[str, ...],
) -> tuple[tuple[str, TrackRecordBuilder], ...]:
    missing = [
        track_id for track_id in track_ids if track_id not in TRACK_RECORD_BUILDERS
    ]
    if missing:
        raise SuiteStoreError(
            f"normalized suite executor does not implement track {missing[0]!r}"
        )
    return tuple((track_id, TRACK_RECORD_BUILDERS[track_id]) for track_id in track_ids)


def _method_record_builder(
    manifest: BenchmarkSuiteManifest,
    track_id: str,
) -> TrackRecordBuilder | None:
    """Resolve method-specific reducers only for exact parser-bound imports."""

    qualification = manifest.qualification_receipt.qualification
    if (
        qualification.origin != "registered_parser_import"
        or not qualification.parser_verified
    ):
        return None
    method = method_plugin_for_benchmark(manifest.adapter_id)
    matches = tuple(
        contract.build_records
        for contract in METHOD_RECORD_BUILDERS
        if contract.method == method and contract.track_id == track_id
    )
    if len(matches) > 1:
        raise SuiteStoreError(
            "normalized suite method matches multiple record reducers"
        )
    return matches[0] if matches else None


def execute_normalized_suites(
    *,
    store: NormalizedSuiteStore,
    manifests: tuple[BenchmarkSuiteManifest, ...],
    track_ids: tuple[str, ...],
    sample_limit: int,
    seed: int,
    executor_id: str,
    target_id: str,
) -> NormalizedSuiteExecution:
    """Replay exact installed suite revisions without loading upstream code."""

    if not manifests:
        raise SuiteStoreError("normalized suite execution requires at least one suite")
    builders = _validated_track_builders(track_ids)
    manifests = tuple(sorted(manifests, key=lambda item: item.id))
    selected, known_case_ids = load_selected_cases(
        store,
        manifests,
        sample_limit,
        seed,
        track_ids,
        executor_id,
    )
    if not selected:
        raise SuiteStoreError("normalized suite sampling selected no cases")
    evidence_by_suite = {
        manifest.id: load_suite_evidence(store, manifest, known_case_ids[manifest.id])
        for manifest in manifests
    }
    records: list[ExecutionRecord] = []
    for case in selected:
        evidence = evidence_by_suite[case.manifest.id]
        for track_id, build_records in builders:
            if track_id in case.visible.track_ids:
                method_builder = _method_record_builder(case.manifest, track_id)
                records.extend((method_builder or build_records)(case, evidence))
    inputs = build_inputs(
        manifests,
        selected,
        evidence_by_suite,
        track_ids,
        executor_id,
        target_id,
    )
    return NormalizedSuiteExecution(inputs=inputs, records=records)
