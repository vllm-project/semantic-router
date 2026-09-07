"""Frozen model-pool coordinates and dynamic metric identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_analysis_catalog import (
    decode_metric_subject_id,
    encode_metric_subject_id,
)

MIN_DENSE_POOL_ARMS = 2
MAX_DENSE_POOL_ARMS = 64
MAX_DENSE_POOL_CELLS = 50_000
ARM_METRIC_PART_COUNT = 2
ARM_MEASURES = ("marginal_contribution", "quality", "success_rate")
_EVIDENCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class ModelPoolReductionContext:
    """Immutable cohort supplied by the manifest/case plan, never records."""

    frozen_arm_ids: tuple[str, ...]
    planned_case_ids: tuple[str, ...]
    authoritative: bool

    def __post_init__(self) -> None:
        arms = tuple(sorted(self.frozen_arm_ids))
        cases = tuple(sorted(self.planned_case_ids))
        if (
            not isinstance(self.authoritative, bool)
            or len(arms) < MIN_DENSE_POOL_ARMS
            or len(arms) > MAX_DENSE_POOL_ARMS
            or len(arms) != len(set(arms))
            or not cases
            or len(cases) != len(set(cases))
            or len(arms) * len(cases) > MAX_DENSE_POOL_CELLS
            or any(_EVIDENCE_ID.fullmatch(value) is None for value in (*arms, *cases))
        ):
            raise ValueError("model-pool reduction context is not a frozen dense plan")
        object.__setattr__(self, "frozen_arm_ids", arms)
        object.__setattr__(self, "planned_case_ids", cases)


def model_pool_arm_segment(arm_id: str) -> str:
    """Return the catalog-owned one-segment portable arm encoding."""

    return encode_metric_subject_id(arm_id)


def decode_model_pool_arm_segment(segment: str) -> str:
    return decode_metric_subject_id(segment)


def model_pool_arm_metric_id(arm_id: str, measure: str) -> str:
    if measure not in ARM_MEASURES:
        raise ValueError("unknown model-pool arm metric measure")
    return f"model_pool.arm.{model_pool_arm_segment(arm_id)}.{measure}"


def parse_model_pool_arm_metric_id(metric_id: str) -> tuple[str, str] | None:
    """Decode a dynamic metric without ever splitting a raw dotted arm ID."""

    prefix = "model_pool.arm."
    if not metric_id.startswith(prefix):
        return None
    segment_and_measure = metric_id.removeprefix(prefix).rsplit(".", 1)
    if len(segment_and_measure) != ARM_METRIC_PART_COUNT:
        return None
    segment, measure = segment_and_measure
    if measure not in ARM_MEASURES:
        return None
    try:
        arm_id = decode_model_pool_arm_segment(segment)
    except ValueError:
        return None
    if _EVIDENCE_ID.fullmatch(arm_id) is None:
        return None
    return arm_id, measure


def build_dense_model_pool_matrix(
    records: list[ExecutionRecord], context: ModelPoolReductionContext
) -> dict[str, dict[str, ExecutionRecord | None]]:
    """Validate coordinates against the immutable case x arm plan."""

    arm_ids = frozenset(context.frozen_arm_ids)
    case_ids = frozenset(context.planned_case_ids)
    matrix: dict[str, dict[str, ExecutionRecord | None]] = {
        case_id: dict.fromkeys(context.frozen_arm_ids)
        for case_id in context.planned_case_ids
    }
    for record in records:
        if record.track_id != "model_pool" or record.arm_id is None:
            raise ValueError(
                "model-pool reducer received a record without one pool coordinate"
            )
        if record.case_id not in case_ids or record.arm_id not in arm_ids:
            raise ValueError("model-pool record lies outside frozen matrix")
        if matrix[record.case_id][record.arm_id] is not None:
            raise ValueError("model-pool record duplicates frozen coordinate")
        matrix[record.case_id][record.arm_id] = record
    return matrix
