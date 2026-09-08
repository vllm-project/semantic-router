"""Resolve source selectors against a frozen server-owned target arm set."""

from __future__ import annotations

from cli.evaluation.target_contracts import EvaluationTargetArm


def resolve_target_arm(
    selector: str | None,
    arms: tuple[EvaluationTargetArm, ...],
) -> EvaluationTargetArm | None:
    """Return the unique target arm matching an exact public ID or model name."""

    if selector is None:
        return None
    matches = [arm for arm in arms if selector in {arm.id, arm.model}]
    return matches[0] if len(matches) == 1 else None


def resolve_target_arm_id(
    selector: str | None,
    arms: tuple[EvaluationTargetArm, ...],
) -> str | None:
    """Return the public ID of the uniquely resolved target arm."""

    arm = resolve_target_arm(selector, arms)
    return arm.id if arm is not None else None
