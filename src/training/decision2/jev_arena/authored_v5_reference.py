"""Separate semantic oracle for the v5 feasibility registry.

The primary evaluator never calls this implementation. It is deliberately
written with different selection and integer arithmetic for cross-checks.
"""

from __future__ import annotations

from typing import Any

from .authored_v5_registry import PAIRS, validate_facts


def evaluate_reference(
    pair_id: str, facts: dict[str, Any], *, archived: bool = False
) -> str | bool | int:
    pair = PAIRS[pair_id]
    validate_facts(pair, facts)
    if pair.kind == "choice":
        current_metric, old_metric, feasible_field = pair.fields
        metric = old_metric if archived else current_metric
        # The procurement current rule is a benefit maximum; its first field
        # is cost, so it has the opposite metric order from the other pairs.
        if pair_id == "procurement-benefit":
            metric = "costs" if archived else "benefits"
            maximize = not archived
        elif pair_id == "delivery-speed":
            metric = "reliability" if archived else "lead_days"
            maximize = archived
        elif pair_id == "grant-impact":
            metric = "cofund_pct" if archived else "community_impact"
            maximize = True
        else:
            metric = "start_hours" if archived else "exposure_reduced"
            maximize = not archived
        winner = None
        for candidate in sorted(facts[feasible_field]):
            if winner is None or (
                facts[metric][candidate] > facts[metric][winner]
                if maximize
                else facts[metric][candidate] < facts[metric][winner]
            ):
                winner = candidate
        assert winner is not None
        return winner
    if pair_id == "committee-quorum":
        return (
            facts["signed_votes"] > facts["quorum"]
            if archived
            else facts["signed_votes"] >= facts["quorum"]
        )
    if pair_id == "inventory-buffer":
        required = facts["reserve_min"] + (0 if archived else facts["demand_next_day"])
        return facts["remaining_stock"] >= required
    if pair_id == "expense-receipt":
        within_budget = facts["expense"] <= facts["approved_budget"]
        return within_budget and (archived or facts["receipt_verified"])
    if pair_id == "incident-escalation":
        serious = facts["severity"] >= 3
        unsigned_mitigation = not facts["mitigation_signed"]
        return facts["incident_signed"] and (
            serious and unsigned_mitigation
            if archived
            else serious or unsigned_mitigation
        )
    if pair_id == "audit-ratio":
        numerator = 4 * facts["passed"]
        denominator = facts["reviewed"]
        return min(
            4,
            (
                numerator // denominator
                if archived
                else (numerator + denominator - 1) // denominator
            ),
        )
    if pair_id == "restoration-lateness":
        penalty = (facts["late_days"] + 1) // 2 if archived else facts["late_days"]
        return max(0, 4 - penalty)
    if pair_id == "risk-product":
        if archived:
            return max(facts["likelihood"], facts["impact"])
        product = facts["likelihood"] * facts["impact"]
        return min(4, (product + 3) // 4)
    if pair_id == "quality-critical":
        raw = facts["passed_checks"] - (0 if archived else facts["critical_failures"])
        return min(4, max(0, raw))
    raise AssertionError(pair_id)


def governing_reference(
    state: str, pair_id: str, facts: dict[str, Any]
) -> str | bool | int:
    """Read which of two policy texts is visibly current before executing it."""
    pair = PAIRS[pair_id]
    lines = state.splitlines()
    current = [
        line.removeprefix("Current signed rule: ")
        for line in lines
        if line.startswith("Current signed rule: ")
    ]
    old = [
        line.removeprefix("Archived rule: ")
        for line in lines
        if line.startswith("Archived rule: ")
    ]
    if len(current) != 1 or len(old) != 1:
        raise ValueError("Visible policy provenance is ambiguous")
    if current[0] == pair.current_text and old[0] == pair.archived_text:
        return evaluate_reference(pair_id, facts)
    if current[0] == pair.archived_text and old[0] == pair.current_text:
        return evaluate_reference(pair_id, facts, archived=True)
    raise ValueError("Visible policy text differs from registered executable pair")


def swap_priority(state: str, pair_id: str) -> str:
    """Change only visible policy-priority metadata, leaving sources intact."""
    pair = PAIRS[pair_id]
    first = f"Current signed rule: {pair.current_text}"
    second = f"Archived rule: {pair.archived_text}"
    if state.count(first) != 1 or state.count(second) != 1:
        raise ValueError("Cannot locate the unique paired-policy provenance")
    return (
        state.replace(first, "__POLICY_SWAP_A__", 1)
        .replace(second, first.replace("Current signed rule: ", "Archived rule: "), 1)
        .replace(
            "__POLICY_SWAP_A__",
            second.replace("Archived rule: ", "Current signed rule: "),
            1,
        )
    )
