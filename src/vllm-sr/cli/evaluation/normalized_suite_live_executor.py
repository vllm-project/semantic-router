"""Execute an installed normalized workload against the current runtime."""

from __future__ import annotations

from dataclasses import dataclass

from cli.evaluation.broker_client import require_broker_for_authenticated_target
from cli.evaluation.contracts import RunManifest, VisibleCaseSet
from cli.evaluation.evidence import ExecutionRecord, RoutingDiagnostic
from cli.evaluation.evidence_source_ids import (
    NORMALIZED_LIVE_MULTIMODAL_EVIDENCE_SOURCE_ID,
)
from cli.evaluation.execution_contract import (
    NORMALIZED_LIVE_EXECUTOR_ID,
    EvaluationInputs,
)
from cli.evaluation.live_executor import (
    LIVE_RUNTIME_TRACKS,
    execute_live_raw,
    grade_live_execution,
)
from cli.evaluation.normalized_suite_inputs import load_selected_cases
from cli.evaluation.normalized_suite_live_admission import (
    normalized_suite_live_tracks,
)
from cli.evaluation.normalized_suite_live_robustness import (
    attach_live_declared_shift_evidence,
)
from cli.evaluation.normalized_suite_target_inputs import build_target_inputs
from cli.evaluation.runtime_factors import runtime_factors
from cli.evaluation.suite_contract import BenchmarkSuiteManifest
from cli.evaluation.suite_store import NormalizedSuiteStore
from cli.evaluation.suite_store_error import SuiteStoreError
from cli.evaluation.target_arm_resolution import resolve_target_arm_id


@dataclass(frozen=True)
class NormalizedSuiteLiveExecution:
    inputs: EvaluationInputs
    records: list[ExecutionRecord]
    discovered_entrypoints: tuple[str, ...]
    routing_traces: tuple[RoutingDiagnostic, ...]


def _grade_target_records(
    records: list[ExecutionRecord],
    inputs: EvaluationInputs,
) -> list[ExecutionRecord]:
    labels = {case.case_id: case for case in inputs.grading.cases}
    graded: list[ExecutionRecord] = []
    for record in records:
        updates: dict[str, object] = {
            "evidence_kind": (
                record.evidence_kind
                if record.track_id == "capacity"
                else NORMALIZED_LIVE_MULTIMODAL_EVIDENCE_SOURCE_ID
            )
        }
        label = labels[record.case_id]
        if record.track_id == "routing":
            selected_arm_id = resolve_target_arm_id(record.selected_arm_id, inputs.arms)
            updates["selected_arm_id"] = selected_arm_id
            if selected_arm_id is not None and label.expected_route is not None:
                updates["quality"] = float(selected_arm_id == label.expected_route)
                updates["grader"] = "normalized-suite-hidden-route-label.v1"
        elif (
            record.track_id in {"model_pool", "joint", "multimodal"}
            and record.quality is not None
        ):
            updates["grader"] = "normalized-suite-hidden-answer-exact.v1"
        graded.append(record.model_copy(update=updates))
    return graded


def execute_normalized_suite_live(
    *,
    manifest: RunManifest,
    store: NormalizedSuiteStore,
    manifests: tuple[BenchmarkSuiteManifest, ...],
    executor_id: str,
) -> NormalizedSuiteLiveExecution:
    """Run selected visible cases first, then join their hidden grading labels."""

    require_broker_for_authenticated_target(manifest.target)
    if manifest.mode != "live":
        raise SuiteStoreError("normalized target execution requires live mode")
    if executor_id != NORMALIZED_LIVE_EXECUTOR_ID:
        raise SuiteStoreError("normalized target execution requires its exact executor")
    unsupported = sorted(set(manifest.track_ids) - LIVE_RUNTIME_TRACKS)
    if unsupported:
        raise SuiteStoreError(
            f"normalized target executor does not implement track {unsupported[0]!r}"
        )
    manifests = tuple(sorted(manifests, key=lambda item: item.id))
    admitted_by_suite = {
        suite.id: normalized_suite_live_tracks(store, suite) for suite in manifests
    }
    for suite in manifests:
        inadmissible = sorted(
            set(manifest.track_ids).intersection(suite.track_ids)
            - admitted_by_suite[suite.id]
        )
        if inadmissible:
            raise SuiteStoreError(
                f"suite {suite.id} has no first-party normalized live method for "
                + ", ".join(inadmissible)
            )
    admitted_tracks = frozenset(
        track for tracks in admitted_by_suite.values() for track in tracks
    )
    uncovered = sorted(set(manifest.track_ids) - admitted_tracks)
    if uncovered:
        raise SuiteStoreError(
            "normalized target executor has no admitted source for "
            + ", ".join(uncovered)
        )
    selected, _ = load_selected_cases(
        store,
        manifests,
        manifest.sample_limit,
        manifest.seed,
        manifest.track_ids,
        executor_id,
    )
    if not selected:
        raise SuiteStoreError("normalized suite sampling selected no cases")
    visible = VisibleCaseSet(cases=tuple(case.visible for case in selected))
    if manifest.target.envoy_url is None:
        raise SuiteStoreError("normalized target execution requires envoy_url")
    if manifest.target.mixture is None:
        raise SuiteStoreError("normalized target execution requires a frozen mixture")
    raw = execute_live_raw(
        visible,
        track_ids=manifest.track_ids,
        router_api_url=manifest.target.router_api_url,
        envoy_url=manifest.target.envoy_url,
        concurrency=manifest.concurrency,
        capacity_load_protocol=manifest.capacity_load_protocol,
        mixture=manifest.target.mixture,
    )
    factors = runtime_factors(manifest)
    inputs = build_target_inputs(manifest, manifests, selected, factors, executor_id)
    hidden_graded = grade_live_execution(raw, inputs.grading)
    records = _grade_target_records(hidden_graded.records, inputs)
    records = attach_live_declared_shift_evidence(
        records=records,
        selected=selected,
        manifests=manifests,
        store=store,
        arms=inputs.arms,
    )
    return NormalizedSuiteLiveExecution(
        inputs=inputs,
        records=records,
        discovered_entrypoints=hidden_graded.discovered_entrypoints,
        routing_traces=hidden_graded.routing_traces,
    )
