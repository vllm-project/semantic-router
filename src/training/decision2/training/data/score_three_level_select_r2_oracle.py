"""Independent mechanical rules for private three-level Score SELECT r2.

The private source facts and answers are never checked into this repository.
This module deliberately does not import the question authoring code.
"""

from __future__ import annotations

from typing import Any


def score(operation: str, facts: dict[str, Any]) -> int:
    if operation == "waiver_precedence":
        day = facts["review_day"]
        veto = facts["veto_active"]
        waiver = facts["waiver"]
        valid_waiver = (
            waiver["lead_signed"]
            and waiver["second_signed"]
            and waiver["signed_day"] <= day <= waiver["expires_day"]
        )
        if veto and not valid_waiver:
            return 0
        return 2 if facts["secondary_check"] == "passed" else 1
    if operation == "inclusive_coverage":
        request = facts["request"]
        return sum(
            window[0] <= request[0] and request[1] <= window[1]
            for window in facts["current_windows"].values()
        )
    if operation == "independent_quorum":
        lineages = {
            report["lineage"]
            for report in facts["reports"]
            if report["signed"] and report["current"] and report["affirmative"]
        }
        return min(2, len(lineages))
    if operation == "allocation_caps":
        return sum(
            pool["committed"] + pool["request"] + pool["reserve_floor"]
            <= pool["capacity"]
            for pool in facts["pools"].values()
        )
    raise ValueError(f"Unknown operation: {operation}")
