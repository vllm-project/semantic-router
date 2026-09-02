"""Shared routing grader derived from a complete dense model-pool cohort."""

from __future__ import annotations

from cli.evaluation.evidence import ExecutionRecord

DENSE_POOL_ORACLE_GRADER = "dense-pool-oracle.v1"
_MIN_DENSE_POOL_ARMS = 2


def grade_routing_with_dense_pool_oracle(
    records: list[ExecutionRecord],
    arm_ids: tuple[str, ...],
) -> list[ExecutionRecord]:
    """Grade route choice against per-case best-arm outcomes.

    A route receives no inferred grade unless every frozen arm has one
    successful quality-bearing observation for that exact case. This keeps the
    replay and live Mixture cohort on one metric while failing closed whenever
    the counterfactual pool is incomplete.
    """

    expected_arms = set(arm_ids)
    if len(expected_arms) < _MIN_DENSE_POOL_ARMS or len(expected_arms) != len(arm_ids):
        raise ValueError("dense pool oracle requires at least two unique arms")
    pool_by_case: dict[str, list[ExecutionRecord]] = {}
    for row in records:
        if row.track_id == "model_pool":
            pool_by_case.setdefault(row.case_id, []).append(row)
    graded: list[ExecutionRecord] = []
    for row in records:
        if (
            row.track_id != "routing"
            or row.quality is not None
            or not row.success
            or row.selected_arm_id is None
        ):
            graded.append(row)
            continue
        outcomes = pool_by_case.get(row.case_id, [])
        if (
            len(outcomes) != len(expected_arms)
            or {outcome.arm_id for outcome in outcomes} != expected_arms
            or any(
                not outcome.success or outcome.quality is None for outcome in outcomes
            )
        ):
            graded.append(row)
            continue
        best = max(outcome.quality or 0 for outcome in outcomes)
        oracle_arms = {
            outcome.arm_id
            for outcome in outcomes
            if outcome.arm_id is not None and outcome.quality == best
        }
        graded.append(
            row.model_copy(
                update={
                    "quality": float(row.selected_arm_id in oracle_arms),
                    "grader": DENSE_POOL_ORACLE_GRADER,
                }
            )
        )
    return graded
