"""Strict normalization for continuity-bench paired failover artifacts."""

from __future__ import annotations

from pathlib import Path

from cli.evaluation.benchmark_normalization_io import (
    NormalizationError,
    boolean,
    number,
    string,
)
from cli.evaluation.benchmark_normalization_types import (
    BenchmarkNormalizerDescriptor,
    NormalizedAdapterPayload,
)
from cli.evaluation.benchmark_normalizer_continuity_source import (
    CONTINUITY_FAULT_KINDS,
    ContinuityContext,
    build_continuity_context,
    continuity_metric_matrix,
    continuity_source_bundle,
    validate_continuity_log_coverage,
)
from cli.evaluation.benchmark_normalizers_common import (
    applicable_track_ids,
    native_digest,
    opaque_id,
)
from cli.evaluation.contracts import CaseGrading, CaseVisible
from cli.evaluation.suite_contract import NormalizedFault, NormalizedTrajectoryStep


def _case_artifacts(
    context: ContinuityContext,
    descriptor: BenchmarkNormalizerDescriptor,
) -> tuple[
    CaseVisible,
    CaseGrading,
    tuple[NormalizedTrajectoryStep, ...],
    str,
    str,
]:
    trajectory_id = opaque_id("trajectory", "continuity", *context.identity)
    case_id = opaque_id("case", "continuity", *context.identity)
    visible = CaseVisible(
        id=case_id,
        track_ids=applicable_track_ids(descriptor.track_ids, modality="text"),
        messages=context.turns,
        tags=(
            "continuity-bench",
            context.system,
            f"concurrency-{context.concurrency}",
        ),
        trajectory_id=trajectory_id,
    )
    grading = CaseGrading(
        case_id=case_id,
        expected_answer=string(context.source["expected_fact"], "expected_fact"),
    )
    state_before = native_digest(
        {
            "expected_fact": context.source["expected_fact"],
            "conversation_id": context.conversation_id,
        }
    )
    state_after = (
        state_before
        if context.preserved
        else native_digest(
            {
                "response_text": context.log["response_text"],
                "error": context.log["error"],
            }
        )
    )
    record_digest = native_digest(
        {"plan": context.plan, "log": context.log, "metrics": context.row}
    )
    steps = (
        NormalizedTrajectoryStep(
            trajectory_id=trajectory_id,
            step_id=opaque_id("step", "continuity", *context.identity, "context"),
            sequence=0,
            case_id=case_id,
            state_digest_after=state_before,
            source_record_digest=native_digest(context.source),
        ),
        NormalizedTrajectoryStep(
            trajectory_id=trajectory_id,
            step_id=opaque_id("step", "continuity", *context.identity, "fault"),
            sequence=1,
            case_id=case_id,
            selected_action_id=opaque_id(
                "action",
                "continuity",
                context.system,
                str(context.log["failover_from"]),
            ),
            state_digest_before=state_before,
            source_record_digest=native_digest(context.log),
        ),
        NormalizedTrajectoryStep(
            trajectory_id=trajectory_id,
            step_id=opaque_id("step", "continuity", *context.identity, "terminal"),
            sequence=2,
            case_id=case_id,
            selected_action_id=opaque_id(
                "action", "continuity", context.system, str(context.log["provider"])
            ),
            state_digest_before=state_before,
            state_digest_after=state_after,
            terminal=True,
            terminal_success=context.recovered and context.preserved,
            task_score=1.0 if context.preserved else 0.0,
            source_record_digest=record_digest,
        ),
    )
    return visible, grading, steps, record_digest, trajectory_id


def _fault_artifact(
    context: ContinuityContext,
    trajectory_id: str,
    record_digest: str,
    native_pair_count: int,
) -> NormalizedFault | None:
    if context.system != "treatment":
        return None
    baseline_recovered = context.counterpart_log["error"] is None and bool(
        context.counterpart_log["response_text"]
    )
    baseline_success = baseline_recovered and boolean(
        context.counterpart_row["preserved"], "continuity baseline preserved"
    )
    treatment_success = context.recovered and context.preserved
    pair_id = opaque_id(
        "cohort", "continuity", context.conversation_id, str(context.concurrency)
    )
    return NormalizedFault(
        id=opaque_id("fault", "continuity", *context.identity),
        trajectory_id=trajectory_id,
        sequence=1,
        kind=CONTINUITY_FAULT_KINDS[context.mode],
        diagnostic_scope="provider_fallback_label_and_context_preservation",
        method_id="continuity.labeled-failover.v1",
        cohort_pair_id=pair_id,
        conversation_id=context.conversation_id,
        system_role="treatment",
        concurrency=context.concurrency,
        failure_turn=context.failure_turn,
        native_repetition_count=1,
        repeated_seed_evidence=False,
        native_pair_count=native_pair_count,
        failover_labeled=True,
        context_preserved=treatment_success,
        experiment_manifest_digest=native_digest(context.plan),
        baseline_record_digest=native_digest(
            {
                "plan": context.plan,
                "log": context.counterpart_log,
                "metrics": context.counterpart_row,
            }
        ),
        treatment_record_digest=record_digest,
        baseline_terminal_success=baseline_success,
        treatment_terminal_success=treatment_success,
        baseline_latency_ms=number(
            context.counterpart_log["latency_ms"], "baseline latency"
        ),
        treatment_latency_ms=number(context.log["latency_ms"], "treatment latency"),
        source_record_digest=record_digest,
    )


def normalize_continuitybench(
    root: Path, descriptor: BenchmarkNormalizerDescriptor
) -> NormalizedAdapterPayload:
    artifacts = {item.id: item for item in descriptor.required_artifacts}
    conversations, manifest, logs = continuity_source_bundle(root, artifacts)
    metric_rows, metric_by_key, native_pair_count = continuity_metric_matrix(
        root, artifacts["raw-metrics"]
    )
    visible = []
    grading = []
    trajectories = []
    faults = []
    seen: set[tuple[str, str, int]] = set()
    for row in metric_rows:
        context = build_continuity_context(
            row, conversations, manifest, logs, metric_by_key, seen
        )
        (
            case_visible,
            case_grading,
            steps,
            record_digest,
            trajectory_id,
        ) = _case_artifacts(context, descriptor)
        visible.append(case_visible)
        grading.append(case_grading)
        trajectories.extend(steps)
        fault = _fault_artifact(
            context,
            trajectory_id,
            record_digest,
            native_pair_count,
        )
        if fault is not None:
            faults.append(fault)
    if not visible:
        raise NormalizationError("continuity-bench raw metrics are empty")
    validate_continuity_log_coverage(logs, seen)
    return NormalizedAdapterPayload(
        visible_cases=tuple(visible),
        grading_cases=tuple(grading),
        trajectories=tuple(trajectories),
        faults=tuple(faults),
        split_protocol=(
            "Fixed conversation suite; manifest-selected provider fallback labels are "
            "cross-bound to logs and judged context preservation. This source does not "
            "execute a real timeout, HTTP error, retry, or partial-stream fault."
        ),
    )
