"""Every planned known-issue waiver of a CI result, and what its evidence must show.

A waiver names exact cases and one accepted outcome for them; it never changes
a threshold, deadline or pass rate. The release Guard waiver (#4120) covers one
release-only case. The recipe CPU waiver (#4706) covers the preview deadline on
named long-context probes, in every profile that runs live recipe conformance.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from release_guard_waiver import planned_waiver as planned_release_waiver
from release_guard_waiver import waiver_evidence_errors as guard_evidence_errors

RECIPE_VERIFICATION = "recipe-conformance"
RECIPE_TOOLS = Path(__file__).resolve().parents[1] / "calibration" / "recipe"
DEADLINE_OUTCOMES = frozenset({"request_timeout", "signals_cut_at_deadline"})


def _recipe_waivers() -> ModuleType:
    if str(RECIPE_TOOLS) not in sys.path:
        sys.path.insert(0, str(RECIPE_TOOLS))
    import recipe_conformance_waivers  # noqa: PLC0415 - the policy lives with the recipe tooling

    return recipe_conformance_waivers


def recipe_waiver() -> dict:
    return _recipe_waivers().waiver_policy()


def planned_waiver(ci_profile: str, verification_id: str) -> dict | None:
    if waiver := planned_release_waiver(ci_profile, verification_id):
        return waiver
    if verification_id == RECIPE_VERIFICATION:
        return recipe_waiver()
    return None


def is_recipe_waiver(waiver: dict) -> bool:
    return waiver.get("kind") == _recipe_waivers().KIND


def waived_cases(waiver: dict) -> frozenset[str]:
    if is_recipe_waiver(waiver):
        return frozenset(waiver.get("cases", ()))
    return frozenset({waiver["case"]})


def waiver_evidence_errors(evidence: dict, waiver: dict) -> list[str]:
    if not is_recipe_waiver(waiver):
        return guard_evidence_errors(evidence, waiver)
    errors = []
    if waiver != recipe_waiver():
        errors.append("unknown recipe CPU waiver")
    if evidence.get("known_issue_waiver") != waiver:
        errors.append("recipe CPU waiver differs from plan")
    failures = evidence.get("waived_failures")
    if not isinstance(failures, list) or not failures:
        return [*errors, "recipe CPU waiver records no waived failure"]
    named = waived_cases(waiver)
    recorded = {}
    for failure in failures:
        case = failure.get("case") if isinstance(failure, dict) else None
        if case not in named:
            errors.append(f"waived failure {case!r} is not a named long-context probe")
        elif failure.get("outcome") not in DEADLINE_OUTCOMES:
            errors.append(f"{case}: waived outcome is not the preview deadline")
        else:
            recorded[case] = failure["outcome"]
    for case in evidence.get("cases", []):
        if (
            case.get("status") == "failed"
            and recorded.get(case.get("id")) != "request_timeout"
        ):
            errors.append(
                f"{case.get('id')}: failed without a recorded preview timeout"
            )
    return errors
