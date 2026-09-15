"""Deterministic sequential replay for Router Learning policy comparisons."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace

from cli.evaluation.case_plan import project_visible_case_set
from cli.evaluation.contracts import RunManifest
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.execution_contract import EvaluationInputs
from cli.evaluation.fixtures import fixture_inputs
from cli.evaluation.resolution import sample_case_sets
from cli.evaluation.router_learning_corpus import (
    ROUTER_LEARNING_CORPUS,
    RouterLearningCase,
    RouterLearningOutcome,
    router_learning_case_sets,
)
from cli.evaluation.router_learning_evidence import (
    ROUTER_LEARNING_POLICY_IDS,
    RouterLearningMethodEvidence,
)
from cli.evaluation.router_learning_policy import (
    ArmState as _ArmState,
)
from cli.evaluation.router_learning_policy import (
    apply_outcome,
    apply_telemetry,
    cost_penalty,
    score_candidate,
    select_winner,
)


def _beta_sample(rng: random.Random, alpha: float, beta: float) -> float:
    return rng.betavariate(alpha, beta)


def _static_proposal(base_arm_id: str) -> str:
    return base_arm_id


def _beta_bernoulli_proposal(
    case: RouterLearningCase,
    state: dict[str, _ArmState],
    rng: random.Random,
) -> str:
    return max(
        case.eligible_arm_ids,
        key=lambda arm_id: (
            _beta_sample(
                rng,
                1.0 + state[arm_id].successes,
                1.0 + state[arm_id].failures,
            ),
            arm_id,
        ),
    )


def _routing_sampling_proposal(
    case: RouterLearningCase,
    state: dict[str, _ArmState],
    rng: random.Random,
) -> str:
    """Complete production equation; distribution-equivalent seeded Beta draws.

    Parity checks compare the production Go helpers with identical posterior
    samples, state, telemetry, candidate scope, and base route. Python and Go
    random streams are deliberately not claimed to be bit-identical.
    """
    corpus = ROUTER_LEARNING_CORPUS
    arms = {arm.id: arm for arm in corpus.candidate_arms}
    costs = {
        arm_id: arms[arm_id].input_cost_per_million_tokens_usd
        + arms[arm_id].output_cost_per_million_tokens_usd
        for arm_id in case.eligible_arm_ids
    }
    # A protected apply-mode request suppresses sampling in production.
    use_sampling = case.protected_arm_id is None
    sample = (
        (lambda alpha, beta: _beta_sample(rng, alpha, beta)) if use_sampling else None
    )
    scores = [
        score_candidate(
            arm_id,
            state[arm_id],
            cost_penalty(costs[arm_id], max(costs.values()), corpus.candidate_set),
            arm_id == corpus.base_arm_id,
            sample,
        )
        for arm_id in case.eligible_arm_ids
    ]
    return select_winner(
        scores, corpus.base_arm_id, corpus.candidate_set, use_sampling
    ).model


def _proposal(
    policy_id: str,
    case: RouterLearningCase,
    state: dict[str, _ArmState],
    rng: random.Random,
) -> str:
    if policy_id == "static-base":
        return _static_proposal(ROUTER_LEARNING_CORPUS.base_arm_id)
    if policy_id == "routing-sampling":
        return _routing_sampling_proposal(case, state, rng)
    if policy_id == "beta-bernoulli":
        return _beta_bernoulli_proposal(case, state, rng)
    raise ValueError(f"unknown Router Learning policy: {policy_id}")


def _apply_feedback(
    state: dict[str, _ArmState],
    arm_id: str,
    outcome: RouterLearningOutcome,
) -> None:
    experience = state[arm_id]
    apply_outcome(experience, outcome.feedback_verdict, outcome.feedback_weight)
    if outcome.success:
        experience.successes += 1
    else:
        experience.failures += 1


def _observe_telemetry(experience: _ArmState, outcome: RouterLearningOutcome) -> None:
    apply_telemetry(
        experience,
        latency_seconds=outcome.latency_ms / 1000 if outcome.latency_ms > 0 else None,
        cache_hit_ratio=outcome.cache_hit_ratio,
        cache_write_pressure=outcome.cache_write_pressure,
        input_cost_multiplier=outcome.input_cost_multiplier,
        provider_failed=outcome.provider_failed,
    )


def _execute_policy_trial(
    *,
    policy_id: str,
    trial_index: int,
    trial_seed: int,
    cases: tuple[RouterLearningCase, ...],
) -> list[ExecutionRecord]:
    corpus = ROUTER_LEARNING_CORPUS
    rng = random.Random(trial_seed)
    state = {arm.id: _ArmState() for arm in corpus.candidate_arms}
    pending: dict[int, list[tuple[str, RouterLearningOutcome]]] = defaultdict(list)
    records: list[ExecutionRecord] = []
    candidate_ids = tuple(arm.id for arm in corpus.candidate_arms)
    trial_id = f"trial-{trial_index + 1:02d}"
    for round_index, case in enumerate(cases):
        for arm_id, feedback in pending.pop(round_index, ()):
            _apply_feedback(state, arm_id, feedback)
        proposed = _proposal(policy_id, case, state, rng)
        selected = proposed
        if selected not in case.eligible_arm_ids:
            selected = case.eligible_arm_ids[0]
        if case.protected_arm_id is not None:
            selected = case.protected_arm_id
        outcome = case.outcomes[selected]
        _observe_telemetry(state[selected], outcome)
        if case.feedback_observed:
            due_round = round_index + case.feedback_delay_rounds + 1
            pending[due_round].append((selected, outcome))
        protection_violation = (
            case.protected_arm_id is not None and selected != case.protected_arm_id
        )
        hard_violation = selected not in case.eligible_arm_ids
        method = RouterLearningMethodEvidence(
            policy_id=policy_id,
            trial_id=trial_id,
            trial_seed=trial_seed,
            round_index=round_index,
            candidate_arm_ids=candidate_ids,
            eligible_arm_ids=case.eligible_arm_ids,
            protected_arm_id=case.protected_arm_id,
            proposed_arm_id=proposed,
            selected_arm_id=selected,
            outcome_success=outcome.success,
            feedback_delay_rounds=case.feedback_delay_rounds,
            feedback_observed=case.feedback_observed,
            protection_required=case.protected_arm_id is not None,
            protection_violation=protection_violation,
            hard_constraint_violation=hard_violation,
            call_count=outcome.call_count,
            lifecycle_cost_usd=outcome.cost_usd,
        )
        records.append(
            ExecutionRecord(
                id=f"rl-{policy_id}-{trial_index + 1:02d}-{round_index + 1:02d}",
                track_id="joint",
                case_id=case.id,
                attempt_id=f"{policy_id}-{trial_id}-round-{round_index + 1:02d}",
                status="succeeded" if outcome.success else "failed",
                selected_arm_id=selected,
                selection_status="selected",
                selection_method=policy_id,
                success=outcome.success,
                quality=outcome.quality,
                latency_ms=outcome.latency_ms,
                runtime_cost=outcome.cost_usd,
                fallback=selected != proposed,
                evidence_kind="router-learning-replay.v1",
                router_learning=method,
            )
        )
    return records


def collect_router_learning_evidence(
    manifest: RunManifest,
) -> tuple[EvaluationInputs, list[ExecutionRecord]]:
    all_visible, all_grading, _ = router_learning_case_sets()
    visible, grading = sample_case_sets(
        all_visible, all_grading, manifest.sample_limit, manifest.seed
    )
    selected_ids = {case.id for case in visible.cases}
    selected_cases = tuple(
        case for case in ROUTER_LEARNING_CORPUS.cases if case.id in selected_ids
    )
    visible, grading, fixture = router_learning_case_sets(selected_cases)
    visible = project_visible_case_set(visible, manifest.track_ids)
    records: list[ExecutionRecord] = []
    for trial_index, offset in enumerate(ROUTER_LEARNING_CORPUS.trial_seeds):
        trial_seed = (manifest.seed + offset) % (2**32)
        for policy_id in ROUTER_LEARNING_POLICY_IDS:
            records.extend(
                _execute_policy_trial(
                    policy_id=policy_id,
                    trial_index=trial_index,
                    trial_seed=trial_seed,
                    cases=selected_cases,
                )
            )
    base = fixture_inputs()
    inputs = replace(
        base,
        visible=visible,
        grading=grading,
        fixture=fixture,
        suite_revisions=dict(manifest.suite_revisions),
        suite_executors=dict(manifest.suite_executors),
        executor_ids=dict.fromkeys(manifest.track_ids, "router-learning-replay.v1"),
    )
    return inputs, records
