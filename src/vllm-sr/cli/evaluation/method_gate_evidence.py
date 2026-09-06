"""Derive release-gate facts only from typed method reducers."""

from __future__ import annotations

from cli.evaluation.contracts import RunManifest
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.gate_context import GateEvidenceContext
from cli.evaluation.metric_hard_policy import reduce_hard_policy
from cli.evaluation.metric_production_experiment import (
    reduce_production_experiment,
)
from cli.evaluation.metric_recovery import (
    MINIMUM_RECOVERY_PASS_RATE_LOWER_BOUND,
    reduce_recovery,
)
from cli.evaluation.metric_robustness import reduce_robustness


def derive_method_gate_evidence(
    manifest: RunManifest,
    records: list[ExecutionRecord],
    *,
    method_qualified_gate_ids: frozenset[str],
) -> GateEvidenceContext:
    """Return method outcomes; absent typed evidence remains ``None``."""

    hard_policy = reduce_hard_policy(
        records,
        policy_snapshot_digest=manifest.policy_snapshot_digest,
        config_digest=manifest.config_digest,
    )
    robustness = reduce_robustness(records)
    recovery = reduce_recovery(
        records,
        policy_snapshot_digest=manifest.policy_snapshot_digest,
        config_digest=manifest.config_digest,
    )
    production = reduce_production_experiment(records)
    return GateEvidenceContext(
        manifest_validated=True,
        snapshots_complete=True,
        artifact_lineage_complete=True,
        hard_policy_static_passed=hard_policy.static_proof_passed,
        robustness_qualified=(
            robustness.passed if "G4" in method_qualified_gate_ids else None
        ),
        recovery_cluster_qualified=recovery.passed,
        recovery_cluster_pass_rate_lower_bound=(
            recovery.cluster_pass_rate_lower_confidence_bound
        ),
        recovery_cluster_minimum_pass_rate_lower_bound=(
            MINIMUM_RECOVERY_PASS_RATE_LOWER_BOUND if recovery.cluster_count else None
        ),
        production_candidate_safe=production.candidate_safe,
        online_preference_qualified=(
            production.preference_passed if production.assignment_count else None
        ),
        production_assignment_support=production.assignment_support,
        production_balance_p_value=production.assignment_balance_p_value,
        production_risk_event_rate=production.risk_event_rate,
        production_risk_event_upper_confidence_bound=(
            production.risk_event_upper_confidence_bound
        ),
        production_risk_budget_max_rate=production.risk_budget_max_rate,
        online_outcome_coverage=production.outcome_coverage,
        online_effective_sample_size=production.effective_sample_size,
        online_minimum_effective_sample_size=(production.minimum_effective_sample_size),
        online_effective_sample_ratio=production.effective_sample_ratio,
        online_minimum_effective_sample_ratio=(
            production.minimum_effective_sample_ratio
        ),
        online_segment_coverage=production.segment_coverage,
        online_snips_reward=production.snips_reward,
        online_reference_snips_reward=production.reference_snips_reward,
        online_causal_eligible=(
            production.causal_eligible if production.assignment_count else None
        ),
        online_reward_lift=production.reward_lift,
        online_reward_lift_lower_bound=(
            production.reward_lift_confidence_interval[0]
            if production.reward_lift_confidence_interval is not None
            else None
        ),
        online_minimum_reward_lift=production.minimum_reward_lift,
        method_qualified_gate_ids=method_qualified_gate_ids,
    )
