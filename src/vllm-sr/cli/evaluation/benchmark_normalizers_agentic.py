"""Strict parsers for AceBench and continuity-bench summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cli.evaluation.benchmark_normalization_io import (
    NormalizationError,
    exact_object,
    load_json,
    number,
    require_array,
    require_object,
    required_file,
    string,
)
from cli.evaluation.benchmark_normalization_types import (
    BenchmarkNormalizerDescriptor,
    NormalizedAdapterPayload,
)
from cli.evaluation.benchmark_normalizer_continuity import (
    normalize_continuitybench,
)
from cli.evaluation.benchmark_normalizers_common import (
    applicable_track_ids,
    native_digest,
    opaque_id,
)
from cli.evaluation.contract_primitives import Message
from cli.evaluation.contracts import CaseGrading, CaseVisible
from cli.evaluation.suite_contract import NormalizedTrajectoryStep

__all__ = ["normalize_acebench", "normalize_continuitybench"]

_ACE_REQUIRED = {
    "category",
    "model",
    "total_tasks",
    "returned_tasks",
    "scored_tasks",
    "overall_average",
    "overall_bar",
    "pass_threshold",
    "pass_count",
    "pass_rate",
    "privacy_average",
    "privacy_bar",
    "usage",
    "final_scores",
    "results",
}
_ACE_OPTIONAL = {"edge_usage", "cloud_usage", "combined_usage", "_routing_note"}


def _ace_score(row: dict[str, Any], index: int) -> tuple[float, str | None]:
    error = row["error"]
    if error is not None:
        error = string(error, f"AceBench results[{index}].error")
    scores = require_object(row["scores"], f"AceBench results[{index}].scores")
    if "overall_score" not in scores:
        if error is None:
            raise NormalizationError("successful AceBench result lacks overall_score")
        return 0.0, error
    overall = number(scores["overall_score"], "AceBench overall_score", maximum=1)
    if "privacy_score" in scores:
        number(scores["privacy_score"], "AceBench privacy_score", maximum=1)
    return overall, error


def normalize_acebench(
    root: Path, descriptor: BenchmarkNormalizerDescriptor
) -> NormalizedAdapterPayload:
    requirement = descriptor.required_artifacts[0]
    summary = exact_object(
        load_json(required_file(root, requirement), max_bytes=requirement.max_bytes),
        required=_ACE_REQUIRED,
        optional=_ACE_OPTIONAL,
        label="AceBench summary",
    )
    threshold = number(summary["pass_threshold"], "AceBench pass_threshold", maximum=1)
    visible = []
    grading = []
    trajectories = []
    seen: set[str] = set()
    for index, value in enumerate(
        require_array(summary["results"], "AceBench results")
    ):
        row = exact_object(
            value,
            required={"task_id", "scores", "error"},
            optional={"output_dir", "usage", "cloud_usage", "task_route"},
            label=f"AceBench results[{index}]",
        )
        task_id = string(row["task_id"], "AceBench task_id")
        if task_id in seen:
            raise NormalizationError("AceBench summary repeats task_id")
        seen.add(task_id)
        overall, error = _ace_score(row, index)
        trajectory_id = opaque_id("trajectory", "acebench", task_id)
        case_id = opaque_id("case", "acebench", task_id)
        visible.append(
            CaseVisible(
                id=case_id,
                track_ids=applicable_track_ids(
                    descriptor.track_ids,
                    modality="text",
                ),
                messages=(
                    Message(role="user", content=f"AceBench agent task {task_id}"),
                ),
                tags=("acebench", string(summary["category"], "AceBench category")),
                trajectory_id=trajectory_id,
            )
        )
        grading.append(CaseGrading(case_id=case_id))
        trajectories.append(
            NormalizedTrajectoryStep(
                trajectory_id=trajectory_id,
                step_id=opaque_id("step", "acebench", task_id),
                sequence=0,
                case_id=case_id,
                selected_action_id=(
                    opaque_id("action", "acebench", row["task_route"])
                    if row.get("task_route")
                    else None
                ),
                terminal=True,
                terminal_success=error is None and overall >= threshold,
                task_score=overall,
                source_record_digest=native_digest(row),
            )
        )
    if not visible:
        raise NormalizationError("AceBench summary has no results")
    return NormalizedAdapterPayload(
        visible_cases=tuple(visible),
        grading_cases=tuple(grading),
        trajectories=tuple(trajectories),
        split_protocol=f"Native run summary; pass threshold fixed by artifact at {threshold}.",
    )
