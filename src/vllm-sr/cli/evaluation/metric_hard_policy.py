"""Live hard-policy proof and dynamic enforcement reduction."""

from __future__ import annotations

from dataclasses import dataclass

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_core import MetricDraft, build_metric


@dataclass(frozen=True)
class HardPolicyReduction:
    observation_count: int
    static_proof_passed: bool | None
    dynamic_passed: bool | None
    target_id: str | None
    backend_topology_digest: str | None
    mixture_snapshot_digest: str | None


def reduce_hard_policy(
    records: list[ExecutionRecord],
    *,
    policy_snapshot_digest: str | None = None,
    config_digest: str | None = None,
) -> HardPolicyReduction:
    rows = [
        row
        for row in records
        if row.track_id == "safety" and row.hard_policy is not None
    ]
    if not rows:
        return HardPolicyReduction(0, None, None, None, None, None)
    first = rows[0].hard_policy
    if first is None:
        raise ValueError("hard-policy row lost typed evidence")
    proof = first.proof
    if (
        policy_snapshot_digest is not None
        and proof.policy_snapshot_digest != policy_snapshot_digest
    ) or (config_digest is not None and proof.config_digest != config_digest):
        raise ValueError("hard-policy proof belongs to a different runtime snapshot")
    observations: set[str] = set()
    attacks: set[str] = set()
    receipts: set[str] = set()
    covered_bindings: set[tuple[str, str]] = set()
    dynamic_passed = True
    for row in rows:
        method = row.hard_policy
        if method is None or method.proof != proof:
            raise ValueError("hard-policy records mix static proofs")
        if (
            method.observation_id in observations
            or method.attack_id in attacks
            or method.decision_receipt_id in receipts
        ):
            raise ValueError("hard-policy records repeat dynamic identities")
        observations.add(method.observation_id)
        attacks.add(method.attack_id)
        receipts.add(method.decision_receipt_id)
        covered_bindings.add((method.rule_id, method.enforcement_point))
        dynamic_passed = dynamic_passed and (
            method.blocked == method.should_block and method.violations == 0
        )
    complete = len(
        rows
    ) == proof.ledger_total_observation_count and covered_bindings == {
        (binding.rule_id, binding.enforcement_point)
        for binding in proof.required_bindings
    }
    return HardPolicyReduction(
        observation_count=len(rows),
        static_proof_passed=complete,
        dynamic_passed=dynamic_passed and complete,
        target_id=proof.target_id,
        backend_topology_digest=proof.backend_topology_digest,
        mixture_snapshot_digest=proof.mixture_snapshot_digest,
    )


def hard_policy_metrics(records: list[ExecutionRecord]) -> list[MetricDraft]:
    reduced = reduce_hard_policy(records)
    return [
        build_metric(
            "safety.hard_policy_static_passed",
            "Runtime hard-policy static proof result",
            "safety",
            (
                float(reduced.static_proof_passed)
                if reduced.static_proof_passed is not None
                else None
            ),
            "boolean",
            "higher_is_better",
            reduced.observation_count,
        ),
        build_metric(
            "safety.hard_policy_observation_count",
            "Hard-policy dynamic observation count",
            "safety",
            float(reduced.observation_count) if reduced.observation_count else None,
            "observations",
            "higher_is_better",
            reduced.observation_count,
        ),
    ]
