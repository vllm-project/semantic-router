"""Deterministic replay of the built-in Mixture-of-Models campaign cohort."""

from __future__ import annotations

import hashlib

from cli.evaluation.canonical import digest_value
from cli.evaluation.contracts import (
    CaseGrading,
    CaseVisible,
    GradingCaseSet,
    RunManifest,
    VisibleCaseSet,
)
from cli.evaluation.evidence import (
    ArmEvidence,
    CapacityEvidence,
    FixtureCaseEvidence,
    MultimodalEvidence,
    PreferenceEvidence,
    ReplayFixture,
    RouteEvidence,
    SafetyEvidence,
    TrajectoryEvidence,
)
from cli.evaluation.target_contracts import EvaluationTargetArm, ManifestMixture

_MIN_MIXTURE_ARMS = 2
_ARM_FAILURE_RATE = 0.08


def _fraction(*parts: str) -> float:
    payload = "\x00".join(parts).encode("utf-8")
    numerator = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return numerator / float(1 << 64)


def _selector_identity(manifest: RunManifest) -> str:
    mixture = manifest.target.mixture
    if mixture is None:
        raise ValueError("Mixture replay requires a frozen target mixture")
    return digest_value(
        {
            "recipe": mixture.recipe_digest,
            "selector": mixture.selector_digest,
            "adaptation": mixture.adaptation_digest,
            "binding": mixture.binding_digest,
        }
    )


def _case_snapshot(case: CaseVisible, grading: CaseGrading) -> str:
    return digest_value(
        {
            "visible": case.model_dump(mode="json", exclude_none=True),
            "grading": grading.model_dump(mode="json", exclude_none=True),
        }
    )


def _arm_evidence(arm: EvaluationTargetArm, case_snapshot: str) -> ArmEvidence:
    reliability = _fraction(
        arm.provider_model_id_digest,
        arm.config_digest or "",
        case_snapshot,
        "reliability",
    )
    success = reliability >= _ARM_FAILURE_RATE
    quality = (
        round(
            _fraction(
                arm.provider_model_id_digest,
                arm.config_digest or "",
                case_snapshot,
                "quality",
            ),
            6,
        )
        if success
        else None
    )
    latency_ms = 12.0 + 108.0 * _fraction(arm.id, case_snapshot, "latency")
    return ArmEvidence(
        arm_id=arm.id,
        success=success,
        quality=quality,
        latency_ms=round(latency_ms, 6),
        input_tokens=16,
        output_tokens=4,
        runtime_cost=(
            16 * arm.input_cost_per_million_tokens_usd
            + 4 * arm.output_cost_per_million_tokens_usd
        )
        / 1_000_000,
    )


def _case_evidence(
    mixture: ManifestMixture,
    selector: str,
    case: CaseVisible,
    grading: CaseGrading,
) -> FixtureCaseEvidence:
    snapshot = _case_snapshot(case, grading)
    selected_index = int(
        _fraction(selector, snapshot, "selection") * len(mixture.model_arms)
    )
    selected_index = min(selected_index, len(mixture.model_arms) - 1)
    selected_arm = mixture.model_arms[selected_index]
    arms = tuple(_arm_evidence(arm, snapshot) for arm in mixture.model_arms)
    selected = next(arm for arm in arms if arm.arm_id == selected_arm.id)
    return FixtureCaseEvidence(
        case_id=case.id,
        route=RouteEvidence(
            selected_model=selected_arm.id,
            selection_status="selected",
            success=True,
            latency_ms=round(
                1.0 + 9.0 * _fraction(selector, snapshot, "route-latency"),
                6,
            ),
        ),
        arms=arms,
        trajectory=TrajectoryEvidence(
            success=selected.success,
            task_score=selected.quality,
            steps=1,
            tool_calls=0,
        ),
        multimodal=MultimodalEvidence(
            supported=selected.success,
            quality=selected.quality,
        ),
        preference=PreferenceEvidence(
            chosen_arm_id=selected_arm.id,
            preferred_arm_id=selected_arm.id,
            reward=selected.quality,
        ),
        safety=SafetyEvidence(
            violations=0,
            should_block=False,
            blocked=False,
        ),
        capacity=CapacityEvidence(
            concurrency=1,
            success=selected.success,
            latency_ms=selected.latency_ms or 0,
            throughput_rps=0,
        ),
    )


def mom_replay_fixture(
    manifest: RunManifest,
    visible: VisibleCaseSet,
    grading: GradingCaseSet,
) -> ReplayFixture:
    """Build dense, reproducible E0 outcomes from only frozen Mixture factors.

    This is a diagnostic counterfactual, never a claim that a provider was
    called. Direct-arm quality is independent of the selector while routed
    choice is keyed by the frozen Recipe/selector/adaptation binding.
    """

    mixture = manifest.target.mixture
    if mixture is None or len(mixture.model_arms) < _MIN_MIXTURE_ARMS:
        raise ValueError("Mixture replay requires at least two frozen model arms")
    selector = _selector_identity(manifest)
    grading_by_id = {case.case_id: case for case in grading.cases}
    if {case.id for case in visible.cases} != set(grading_by_id):
        raise ValueError("Mixture replay visible and grading cohorts must align")
    return ReplayFixture(
        cases=tuple(
            _case_evidence(mixture, selector, case, grading_by_id[case.id])
            for case in visible.cases
        )
    )
