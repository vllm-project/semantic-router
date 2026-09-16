"""Target-independent hidden-label cohort for live Mixture-of-Models evaluation."""

from __future__ import annotations

from cli.evaluation.contract_primitives import Message
from cli.evaluation.contracts import (
    CaseGrading,
    CaseVisible,
    GradingCaseSet,
    VisibleCaseSet,
)
from cli.evaluation.live_mom_case_data import LIVE_MOM_CASE_ROWS

_MOM_TRACKS = ("routing", "model_pool", "joint")
LIVE_MOM_CASE_COUNT = 64


def live_mom_case_sets() -> tuple[VisibleCaseSet, GradingCaseSet]:
    """Return prompts whose requested canonical answers make exact grading valid."""

    cases = LIVE_MOM_CASE_ROWS
    if len(cases) != LIVE_MOM_CASE_COUNT:
        raise AssertionError("live MoM campaign cohort size drifted")
    return (
        VisibleCaseSet(
            cases=tuple(
                CaseVisible(
                    id=case_id,
                    track_ids=_MOM_TRACKS,
                    messages=(Message(role="user", content=prompt),),
                    tags=(
                        "live-mom-core",
                        "canonical-exact-answer",
                        f"domain:{domain}",
                        f"difficulty:{difficulty}",
                    ),
                )
                for case_id, prompt, _, domain, difficulty in cases
            )
        ),
        GradingCaseSet(
            cases=tuple(
                CaseGrading(case_id=case_id, expected_answer=answer)
                for case_id, _, answer, _, _ in cases
            )
        ),
    )
