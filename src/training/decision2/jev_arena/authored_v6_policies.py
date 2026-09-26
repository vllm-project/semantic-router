"""Redesigned semantic policies for a small, blocked-until-reviewed v6 pilot.

The six policies intentionally use every named field in the current decision.
The archived policy is executable from the same facts. A separate reference
oracle, policy-text parser, and author-supplied interventions are required by
the builder; this registry alone cannot approve an item for release.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


@dataclass(frozen=True)
class Policy:
    id: str
    kind: str
    domain: str
    fields: tuple[str, str, str]
    current_text: str
    archived_text: str


POLICIES = {
    policy.id: policy
    for policy in (
        Policy(
            "benefit-per-cost",
            "choice",
            "procurement",
            ("costs", "benefits", "qualified_ids"),
            "Among qualified bids with positive total cost, choose the greatest verified service-benefit points per cost unit (benefit divided by cost); break an exact ratio tie by alphabetical bid ID. If admissible signed worlds choose different bids, return hold.",
            "Among the same qualified bids, choose the lowest total cost; break an exact tie by alphabetical bid ID.",
        ),
        Policy(
            "reliable-delivery",
            "choice",
            "delivery",
            ("lead_days", "reliability", "approved_ids"),
            "Among approved plans whose audited reliability is at least 80, choose the shortest signed lead time; break an exact tie by alphabetical plan ID. If no plan qualifies or admissible signed worlds choose different plans, return hold.",
            "Among the same approved plans, choose the highest audited reliability; break an exact tie by alphabetical plan ID.",
        ),
        Policy(
            "certified-quorum",
            "noul",
            "committee",
            ("signed_votes", "quorum", "committee_size"),
            "A resolution is certified true only if signed votes are at least the required quorum AND twice the signed votes is strictly greater than committee size. Across multiple admissible signed worlds, certify true only if this condition is true in every world; otherwise return false.",
            "A resolution passes only if signed votes strictly exceed quorum AND twice the signed votes is strictly greater than committee size.",
        ),
        Policy(
            "safe-stock-release",
            "noul",
            "inventory",
            ("remaining_stock", "reserve_min", "demand_next_day"),
            "Release is certified safe only if remaining stock units are at least minimum reserve units PLUS signed next-day demand units: stock >= reserve + demand. Across multiple admissible signed worlds, certify true only if this inequality holds in every world; otherwise return false.",
            "Release is allowed if remaining stock units are at least the minimum reserve units, without adding next-day demand.",
        ),
        Policy(
            "signed-audit-ratio",
            "score",
            "audit",
            ("passed", "reviewed", "audit_signed"),
            "For a signed audit, grade min(4, ceil(4 * passed control count / reviewed control count)); an unsigned audit has grade 0. Across multiple admissible signed worlds, publish the lowest computed grade.",
            "For a signed audit, grade min(4, floor(4 * passed control count / reviewed control count)); an unsigned audit has grade 0.",
        ),
        Policy(
            "signed-risk-product",
            "score",
            "risk-review",
            ("likelihood", "impact", "assessment_signed"),
            "For a signed assessment, grade min(4, ceil(likelihood level * impact level / 4)); an unsigned assessment has grade 0. Across multiple admissible signed worlds, publish the lowest computed grade.",
            "For a signed assessment, grade the larger of likelihood level and impact level; an unsigned assessment has grade 0.",
        ),
    )
}


def validate(policy: Policy, facts: dict[str, Any]) -> None:
    if set(facts) != set(policy.fields):
        raise ValueError("Policy fact schema is incomplete or has extra fields")
    if policy.kind == "choice":
        first, second, allowed = (facts[name] for name in policy.fields)
        if (
            not isinstance(first, dict)
            or not isinstance(second, dict)
            or not isinstance(allowed, list)
            or not 2 <= len(first) <= 5
            or set(first) != set(second)
            or not allowed
            or len(set(allowed)) != len(allowed)
            or set(allowed) - set(first)
            or any(
                re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", key) is None or key == "hold"
                for key in first
            )
            or any(
                type(value) is not int or not 1 <= value <= 100
                for value in first.values()
            )
            or any(
                type(value) is not int or not 0 <= value <= 100
                for value in second.values()
            )
        ):
            raise ValueError("Invalid choice register or feasible set")
        return
    if policy.id == "certified-quorum":
        votes, quorum, size = (facts[name] for name in policy.fields)
        valid = (
            all(type(x) is int for x in (votes, quorum, size))
            and 1 <= quorum <= size <= 20
            and 0 <= votes <= size
        )
    elif policy.id == "safe-stock-release":
        stock, reserve, demand = (facts[name] for name in policy.fields)
        valid = (
            all(type(x) is int for x in (stock, reserve, demand))
            and min(stock, reserve, demand) >= 0
            and max(stock, reserve, demand) <= 1000
        )
    elif policy.id == "signed-audit-ratio":
        passed, reviewed, signed = (facts[name] for name in policy.fields)
        valid = (
            type(passed) is int
            and type(reviewed) is int
            and 0 <= passed <= reviewed <= 100
            and reviewed > 0
            and type(signed) is bool
        )
    elif policy.id == "signed-risk-product":
        likelihood, impact, signed = (facts[name] for name in policy.fields)
        valid = (
            type(likelihood) is int
            and type(impact) is int
            and 1 <= likelihood <= 4
            and 1 <= impact <= 4
            and type(signed) is bool
        )
    else:
        raise ValueError("Unknown v6 policy")
    if not valid:
        raise ValueError("Domain-invalid v6 facts")


def evaluate(
    policy: Policy, facts: dict[str, Any], *, archived: bool = False
) -> str | bool | int:
    validate(policy, facts)
    if policy.id == "benefit-per-cost":
        choices = facts["qualified_ids"]
        if archived:
            return min(choices, key=lambda key: (facts["costs"][key], key))
        return min(
            choices,
            key=lambda key: (
                -Fraction(facts["benefits"][key], facts["costs"][key]),
                key,
            ),
        )
    if policy.id == "reliable-delivery":
        choices = facts["approved_ids"]
        if archived:
            return min(choices, key=lambda key: (-facts["reliability"][key], key))
        reliable = [key for key in choices if facts["reliability"][key] >= 80]
        return (
            min(reliable, key=lambda key: (facts["lead_days"][key], key))
            if reliable
            else "hold"
        )
    if policy.id == "certified-quorum":
        sufficient_votes = (
            facts["signed_votes"] > facts["quorum"]
            if archived
            else facts["signed_votes"] >= facts["quorum"]
        )
        return sufficient_votes and 2 * facts["signed_votes"] > facts["committee_size"]
    if policy.id == "safe-stock-release":
        required = facts["reserve_min"] + (0 if archived else facts["demand_next_day"])
        return facts["remaining_stock"] >= required
    if policy.id == "signed-audit-ratio":
        if not facts["audit_signed"]:
            return 0
        share = Fraction(4 * facts["passed"], facts["reviewed"])
        return min(4, math.floor(share) if archived else math.ceil(share))
    if policy.id == "signed-risk-product":
        if not facts["assessment_signed"]:
            return 0
        return (
            max(facts["likelihood"], facts["impact"])
            if archived
            else min(4, math.ceil(Fraction(facts["likelihood"] * facts["impact"], 4)))
        )
    raise AssertionError(policy.id)


def reference(
    policy: Policy, facts: dict[str, Any], *, archived: bool = False
) -> str | bool | int:
    """Independent arithmetic and selection path over validated visible facts."""
    validate(policy, facts)
    if policy.id == "benefit-per-cost":
        winner = None
        for key in sorted(facts["qualified_ids"]):
            if (
                winner is None
                or (archived and facts["costs"][key] < facts["costs"][winner])
                or (
                    not archived
                    and facts["benefits"][key] * facts["costs"][winner]
                    > facts["benefits"][winner] * facts["costs"][key]
                )
            ):
                winner = key
        assert winner is not None
        return winner
    if policy.id == "reliable-delivery":
        ordered = sorted(facts["approved_ids"])
        if archived:
            return sorted(
                ordered, key=lambda key: facts["reliability"][key], reverse=True
            )[0]
        feasible = [key for key in ordered if facts["reliability"][key] >= 80]
        return (
            sorted(feasible, key=lambda key: facts["lead_days"][key])[0]
            if feasible
            else "hold"
        )
    if policy.id == "certified-quorum":
        threshold = facts["quorum"] + (1 if archived else 0)
        return (
            min(
                facts["signed_votes"] - threshold,
                2 * facts["signed_votes"] - facts["committee_size"] - 1,
            )
            >= 0
        )
    if policy.id == "safe-stock-release":
        available = facts["remaining_stock"] - facts["reserve_min"]
        return available >= (0 if archived else facts["demand_next_day"])
    if policy.id == "signed-audit-ratio":
        if not facts["audit_signed"]:
            return 0
        numerator, denominator = 4 * facts["passed"], facts["reviewed"]
        return min(
            4,
            (
                numerator // denominator
                if archived
                else (numerator + denominator - 1) // denominator
            ),
        )
    if policy.id == "signed-risk-product":
        if not facts["assessment_signed"]:
            return 0
        if archived:
            return (
                facts["likelihood"]
                if facts["likelihood"] >= facts["impact"]
                else facts["impact"]
            )
        product = facts["likelihood"] * facts["impact"]
        return min(4, (product + 3) // 4)
    raise AssertionError(policy.id)


def aggregate(policy: Policy, outputs: list[str | bool | int]) -> str | bool | int:
    if not outputs:
        raise ValueError("No admissible worlds")
    if policy.kind == "choice":
        return outputs[0] if all(output == outputs[0] for output in outputs) else "hold"
    if policy.kind == "noul":
        return all(output is True for output in outputs)
    return min(outputs)


def governing(state: str, policy: Policy, facts: dict[str, Any]) -> str | bool | int:
    current = [
        line.removeprefix("Current signed rule: ")
        for line in state.splitlines()
        if line.startswith("Current signed rule: ")
    ]
    old = [
        line.removeprefix("Archived rule: ")
        for line in state.splitlines()
        if line.startswith("Archived rule: ")
    ]
    if len(current) != 1 or len(old) != 1:
        raise ValueError("Ambiguous v6 policy provenance")
    if current == [policy.current_text] and old == [policy.archived_text]:
        return reference(policy, facts)
    if current == [policy.archived_text] and old == [policy.current_text]:
        return reference(policy, facts, archived=True)
    raise ValueError("Visible policy wording differs from executable v6 pair")


def swap_priority(state: str, policy: Policy) -> str:
    current = f"Current signed rule: {policy.current_text}"
    old = f"Archived rule: {policy.archived_text}"
    if state.count(current) != 1 or state.count(old) != 1:
        raise ValueError("Policy swap requires one visible copy of each version")
    return (
        state.replace(current, "__V6_SWAP__", 1)
        .replace(old, f"Archived rule: {policy.current_text}", 1)
        .replace("__V6_SWAP__", f"Current signed rule: {policy.archived_text}", 1)
    )
