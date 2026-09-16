"""Bind an installed normalized workload to the current runtime factor graph."""

from __future__ import annotations

from cli.evaluation.contracts import (
    CaseGrading,
    GradingCaseSet,
    RunManifest,
    VisibleCaseSet,
)
from cli.evaluation.execution_contract import (
    EvaluationInputs,
    NormalizedIdentity,
    NormalizedSuiteIdentities,
)
from cli.evaluation.normalized_suite_inputs import SelectedCase
from cli.evaluation.runtime_factors import RunFactors
from cli.evaluation.suite_contract import BenchmarkSuiteManifest
from cli.evaluation.target_arm_resolution import resolve_target_arm_id
from cli.evaluation.target_contracts import EvaluationTargetArm


def _target_grading(
    case: SelectedCase,
    arms: tuple[EvaluationTargetArm, ...],
) -> CaseGrading:
    """Map only exact source selectors onto the current server-owned arm IDs."""

    return case.source_grading.model_copy(
        update={
            "case_id": case.visible.id,
            "expected_route": resolve_target_arm_id(
                case.source_grading.expected_route, arms
            ),
            "preferred_arm_id": resolve_target_arm_id(
                case.source_grading.preferred_arm_id, arms
            ),
        }
    )


def build_target_inputs(
    manifest: RunManifest,
    manifests: tuple[BenchmarkSuiteManifest, ...],
    selected: tuple[SelectedCase, ...],
    factors: RunFactors,
    executor_id: str,
) -> EvaluationInputs:
    """Create target-bound inputs without importing historical decisions/outcomes."""

    revisions = {suite.id: suite.revision for suite in manifests}
    return EvaluationInputs(
        visible=VisibleCaseSet(cases=tuple(case.visible for case in selected)),
        grading=GradingCaseSet(
            cases=tuple(_target_grading(case, factors.arms) for case in selected)
        ),
        fixture=None,
        policy=factors.policy,
        pool=factors.pool,
        arms=factors.arms,
        binding=factors.binding,
        environment=factors.environment,
        suite_revisions=revisions,
        suite_executors=dict.fromkeys(revisions, executor_id),
        executor_ids=dict.fromkeys(manifest.track_ids, executor_id),
        private_identity_map=NormalizedSuiteIdentities(
            suite_revisions=revisions,
            case_identities=tuple(
                NormalizedIdentity(
                    suite_id=case.manifest.id,
                    opaque_id=case.visible.id,
                    source_id=case.source_visible.id,
                )
                for case in selected
            ),
            arm_identities=(),
            action_identities=(),
        ),
    )
