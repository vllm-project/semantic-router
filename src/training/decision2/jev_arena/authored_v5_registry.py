"""Small, explicit policy-pair registry for authored v5 feasibility work.

The registry contains policy semantics only. Scenario facts, prompts, proof
traces and keys belong in a separate private candidate workspace. These twelve
pairs are an audit target, not an approved release operation count.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyPair:
    id: str
    kind: str
    domain: str
    fields: tuple[str, ...]
    current_text: str
    archived_text: str
    fallback: str | bool | int


PAIRS = {
    pair.id: pair
    for pair in (
        PolicyPair(
            "procurement-benefit",
            "choice",
            "procurement",
            ("costs", "benefits", "qualified_ids"),
            "Among qualified bids, select the bid with the highest verified service benefit; break an exact tie by bid ID.",
            "Among qualified bids, select the bid with the lowest total cost; break an exact tie by bid ID.",
            "hold",
        ),
        PolicyPair(
            "delivery-speed",
            "choice",
            "delivery",
            ("lead_days", "reliability", "approved_ids"),
            "Among approved delivery plans, select the one with the shortest signed lead time; break an exact tie by plan ID.",
            "Among approved delivery plans, select the one with the highest audited reliability; break an exact tie by plan ID.",
            "hold",
        ),
        PolicyPair(
            "grant-impact",
            "choice",
            "grants",
            ("community_impact", "cofund_pct", "eligible_ids"),
            "Among eligible proposals, select the highest reviewed community-impact score; break an exact tie by proposal ID.",
            "Among eligible proposals, select the highest committed co-funding percentage; break an exact tie by proposal ID.",
            "hold",
        ),
        PolicyPair(
            "incident-exposure",
            "choice",
            "incident-response",
            ("exposure_reduced", "start_hours", "cleared_ids"),
            "Among cleared response plans, select the one removing the most measured exposure; break an exact tie by plan ID.",
            "Among cleared response plans, select the one with the earliest start; break an exact tie by plan ID.",
            "hold",
        ),
        PolicyPair(
            "committee-quorum",
            "noul",
            "committee",
            ("signed_votes", "quorum", "committee_size"),
            "A signed resolution passes when signed votes meet or exceed quorum.",
            "A signed resolution passes only when signed votes strictly exceed quorum.",
            False,
        ),
        PolicyPair(
            "inventory-buffer",
            "noul",
            "inventory",
            ("remaining_stock", "reserve_min", "demand_next_day"),
            "Release is allowed only if remaining stock covers both the reserve minimum and the signed next-day demand.",
            "Release is allowed if remaining stock covers the reserve minimum, without the next-day demand buffer.",
            False,
        ),
        PolicyPair(
            "expense-receipt",
            "noul",
            "expense-review",
            ("expense", "approved_budget", "receipt_verified"),
            "Reimbursement is allowed only when the expense is within approved budget and its receipt is verified.",
            "Reimbursement is allowed when the expense is within approved budget, whether or not its receipt is verified.",
            False,
        ),
        PolicyPair(
            "incident-escalation",
            "noul",
            "incident-response",
            ("severity", "mitigation_signed", "incident_signed"),
            "For a signed incident, escalate if severity is at least three or a mitigation sign-off is missing.",
            "For a signed incident, escalate only if severity is at least three and a mitigation sign-off is missing.",
            False,
        ),
        PolicyPair(
            "audit-ratio",
            "score",
            "audit",
            ("passed", "reviewed", "audit_signed"),
            "For a signed audit, grade the passing share on a zero-to-four scale by rounding upward.",
            "For a signed audit, grade the passing share on a zero-to-four scale by rounding downward.",
            0,
        ),
        PolicyPair(
            "restoration-lateness",
            "score",
            "service-restoration",
            ("late_days", "deadline_signed", "closure_signed"),
            "For a signed deadline and closure, begin at grade four and deduct one grade per late day, stopping at zero.",
            "For a signed deadline and closure, begin at grade four and deduct one grade per two late days rounded upward, stopping at zero.",
            0,
        ),
        PolicyPair(
            "risk-product",
            "score",
            "risk-review",
            ("likelihood", "impact", "assessment_signed"),
            "For a signed assessment, multiply likelihood by impact, divide by four and round upward to a grade of at most four.",
            "For a signed assessment, use the larger of likelihood and impact as the grade.",
            0,
        ),
        PolicyPair(
            "quality-critical",
            "score",
            "quality-control",
            ("passed_checks", "critical_failures", "review_signed"),
            "For a signed review, grade passed checks minus critical failures, clamped to zero through four.",
            "For a signed review, grade passed checks alone, clamped to zero through four.",
            0,
        ),
    )
}


def validate_facts(pair: PolicyPair, facts: dict[str, Any]) -> None:
    """Reject incomplete or domain-impossible facts before either evaluator."""
    if set(facts) != set(pair.fields):
        raise ValueError(f"{pair.id}: fact schema differs from both policies")
    if pair.kind == "choice":
        metric_names = pair.fields[:2]
        left, right = (facts[name] for name in metric_names)
        allowed = facts[pair.fields[2]]
        if (
            not isinstance(left, dict)
            or not isinstance(right, dict)
            or not isinstance(allowed, list)
            or not 2 <= len(left) <= 5
            or set(left) != set(right)
            or set(allowed) - set(left)
            or len(set(allowed)) != len(allowed)
            or not allowed
            or any(
                not isinstance(key, str)
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", key) is None
                or key == "hold"
                for key in left
            )
            or any(
                type(value) is not int or not 0 <= value <= 100
                for value in left.values()
            )
            or any(
                type(value) is not int or not 0 <= value <= 100
                for value in right.values()
            )
        ):
            raise ValueError(f"{pair.id}: impossible candidate register")
        return
    if pair.id == "committee-quorum":
        votes, quorum, size = (facts[field] for field in pair.fields)
        valid = (
            all(type(x) is int for x in (votes, quorum, size))
            and 1 <= quorum <= size <= 99
            and 0 <= votes <= size
        )
    elif pair.id == "inventory-buffer":
        stock, reserve, demand = (facts[field] for field in pair.fields)
        valid = (
            all(type(x) is int for x in (stock, reserve, demand))
            and min(stock, reserve, demand) >= 0
            and max(stock, reserve, demand) <= 1000
        )
    elif pair.id == "expense-receipt":
        expense, budget, verified = (facts[field] for field in pair.fields)
        valid = (
            type(expense) is int
            and type(budget) is int
            and 0 <= expense <= 10000
            and 0 < budget <= 10000
            and type(verified) is bool
        )
    elif pair.id == "incident-escalation":
        severity, mitigation, signed = (facts[field] for field in pair.fields)
        valid = (
            type(severity) is int
            and 1 <= severity <= 5
            and type(mitigation) is bool
            and type(signed) is bool
        )
    elif pair.id == "audit-ratio":
        passed, reviewed, signed = (facts[field] for field in pair.fields)
        valid = (
            type(passed) is int
            and type(reviewed) is int
            and 0 <= passed <= reviewed <= 100
            and reviewed > 0
            and signed is True
        )
    elif pair.id == "restoration-lateness":
        late, deadline, closure = (facts[field] for field in pair.fields)
        valid = (
            type(late) is int
            and 0 <= late <= 30
            and deadline is True
            and closure is True
        )
    elif pair.id == "risk-product":
        likelihood, impact, signed = (facts[field] for field in pair.fields)
        valid = (
            type(likelihood) is int
            and type(impact) is int
            and 1 <= likelihood <= 4
            and 1 <= impact <= 4
            and signed is True
        )
    elif pair.id == "quality-critical":
        passed, failed, signed = (facts[field] for field in pair.fields)
        valid = (
            type(passed) is int
            and type(failed) is int
            and 0 <= passed <= 4
            and 0 <= failed <= 4
            and signed is True
        )
    else:
        raise ValueError(f"Unknown policy pair {pair.id}")
    if not valid:
        raise ValueError(f"{pair.id}: domain-inconsistent facts")


def _pick(facts: dict[str, Any], metric: str, allowed: str, *, maximum: bool) -> str:
    choices = facts[allowed]
    ranked = sorted(
        choices, key=lambda key: ((-1 if maximum else 1) * facts[metric][key], key)
    )
    return ranked[0]


def evaluate_current(pair: PolicyPair, facts: dict[str, Any]) -> str | bool | int:
    validate_facts(pair, facts)
    if pair.id == "procurement-benefit":
        return _pick(facts, "benefits", "qualified_ids", maximum=True)
    if pair.id == "delivery-speed":
        return _pick(facts, "lead_days", "approved_ids", maximum=False)
    if pair.id == "grant-impact":
        return _pick(facts, "community_impact", "eligible_ids", maximum=True)
    if pair.id == "incident-exposure":
        return _pick(facts, "exposure_reduced", "cleared_ids", maximum=True)
    if pair.id == "committee-quorum":
        return facts["signed_votes"] >= facts["quorum"]
    if pair.id == "inventory-buffer":
        return (
            facts["remaining_stock"] >= facts["reserve_min"] + facts["demand_next_day"]
        )
    if pair.id == "expense-receipt":
        return (
            facts["expense"] <= facts["approved_budget"] and facts["receipt_verified"]
        )
    if pair.id == "incident-escalation":
        return facts["incident_signed"] and (
            facts["severity"] >= 3 or not facts["mitigation_signed"]
        )
    if pair.id == "audit-ratio":
        return min(4, math.ceil(4 * facts["passed"] / facts["reviewed"]))
    if pair.id == "restoration-lateness":
        return max(0, 4 - facts["late_days"])
    if pair.id == "risk-product":
        return min(4, math.ceil(facts["likelihood"] * facts["impact"] / 4))
    if pair.id == "quality-critical":
        return max(0, min(4, facts["passed_checks"] - facts["critical_failures"]))
    raise AssertionError(pair.id)


def evaluate_archived(pair: PolicyPair, facts: dict[str, Any]) -> str | bool | int:
    """Independent archived-rule implementation over the same validated facts."""
    validate_facts(pair, facts)
    if pair.id == "procurement-benefit":
        return _pick(facts, "costs", "qualified_ids", maximum=False)
    if pair.id == "delivery-speed":
        return _pick(facts, "reliability", "approved_ids", maximum=True)
    if pair.id == "grant-impact":
        return _pick(facts, "cofund_pct", "eligible_ids", maximum=True)
    if pair.id == "incident-exposure":
        return _pick(facts, "start_hours", "cleared_ids", maximum=False)
    if pair.id == "committee-quorum":
        return facts["signed_votes"] > facts["quorum"]
    if pair.id == "inventory-buffer":
        return facts["remaining_stock"] >= facts["reserve_min"]
    if pair.id == "expense-receipt":
        return facts["expense"] <= facts["approved_budget"]
    if pair.id == "incident-escalation":
        return (
            facts["incident_signed"]
            and facts["severity"] >= 3
            and not facts["mitigation_signed"]
        )
    if pair.id == "audit-ratio":
        return min(4, math.floor(4 * facts["passed"] / facts["reviewed"]))
    if pair.id == "restoration-lateness":
        return max(0, 4 - math.ceil(facts["late_days"] / 2))
    if pair.id == "risk-product":
        return max(facts["likelihood"], facts["impact"])
    if pair.id == "quality-critical":
        return max(0, min(4, facts["passed_checks"]))
    raise AssertionError(pair.id)


def prove_conflict(pair: PolicyPair, facts: dict[str, Any]) -> dict[str, Any]:
    """Current/archive must both execute and differ; priority swap must flip."""
    current = evaluate_current(pair, facts)
    archived = evaluate_archived(pair, facts)
    if type(current) is not type(archived) or current == archived:
        raise ValueError(f"{pair.id}: both policy versions do not conflict")
    if pair.kind == "choice" and (
        current not in facts[pair.fields[2]] or archived not in facts[pair.fields[2]]
    ):
        raise ValueError(f"{pair.id}: selection left the feasible set")
    return {
        "pair_id": pair.id,
        "required_fields": list(pair.fields),
        "current_output": current,
        "archived_output": archived,
        "priority_swap_changes_output": True,
    }
