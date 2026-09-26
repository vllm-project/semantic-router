"""Independent policy archetypes for an unqualified authored v7 DEV pilot.

These definitions expose rule semantics, never private questions or gold. A
separate visible-evidence parser and sealed blind review are still required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


@dataclass(frozen=True)
class Policy:
    id: str
    kind: str
    fields: tuple[str, str, str]
    current: str
    archived: str


POLICIES = {
    row.id: row
    for row in (
        Policy(
            "relief-supplier",
            "choice",
            ("costs", "capacity", "certified"),
            "Among certified suppliers, select the highest capacity per cost unit; break exact ratio ties by supplier ID. If eligible winners differ across admissible source worlds, hold.",
            "Among certified suppliers, select the lowest cost; break ties by supplier ID.",
        ),
        Policy(
            "cold-chain-dispatch",
            "choice",
            ("drive_minutes", "seal_ok", "authorized"),
            "Among authorized carriers with a valid cold-chain seal, select the shortest drive; break ties by carrier ID. If no carrier qualifies or admissible winners differ, hold.",
            "Among authorized carriers, select the shortest drive whether or not the seal is valid; break ties by carrier ID.",
        ),
        Policy(
            "grant-panel",
            "choice",
            ("merit", "matching_funds", "eligible"),
            "Among eligible applications with at least 50 matching-fund units, select highest merit; break ties by application ID. If none qualifies or admissible winners differ, hold.",
            "Among eligible applications, select the highest matching-fund amount; break ties by application ID.",
        ),
        Policy(
            "backup-routing",
            "choice",
            ("outage_risk", "restore_hours", "available"),
            "Among available recovery teams, select lowest audited outage risk, then shortest restore time, then team ID. If admissible winners differ, hold.",
            "Among available teams, select shortest restore time, then lowest outage risk, then team ID.",
        ),
        Policy(
            "sterile-release",
            "noul",
            ("signoffs", "open_defects", "seal_intact"),
            "Certify release only when at least two signoffs exist, no defect remains open, and the tamper seal is intact. Certify true across admissible worlds only if every world passes.",
            "Certify release with at least one signoff, at most one open defect, and an intact seal.",
        ),
        Policy(
            "safe-exit",
            "noul",
            ("occupied_levels", "inspected_exits", "certified_capacity"),
            "Certify evacuation readiness only if inspected exits are at least ceiling(occupied levels / 2) and certified exit capacity is at least 12 times occupied levels. Certify true across admissible worlds only if every world passes.",
            "Certify readiness if inspected exits are at least ceiling(occupied levels / 2), without the capacity test.",
        ),
        Policy(
            "supplier-renewal",
            "noul",
            ("delivery_pct", "open_complaints", "bond_current"),
            "Certify renewal only if on-time delivery is at least 96 percent, open complaints are at most two, and the bond is current. Certify true across admissible worlds only if every world passes.",
            "Certify renewal if on-time delivery is at least 90 percent, open complaints are at most two, and the bond is current.",
        ),
        Policy(
            "incident-severity",
            "score",
            ("exposure", "containment", "signed"),
            "If the assessment is signed, grade max(0, min(4, exposure minus containment)); otherwise grade zero. Across admissible worlds report the minimum grade.",
            "If the assessment is signed, grade min(4, exposure); otherwise grade zero.",
        ),
        Policy(
            "inspection-deficit",
            "score",
            ("critical", "major", "reviewer_signed"),
            "If review is signed, grade max(0, 4 minus twice critical findings minus major findings); otherwise grade zero. Across admissible worlds report the minimum grade.",
            "If review is signed, grade max(0, 4 minus critical findings minus major findings); otherwise grade zero.",
        ),
        Policy(
            "readiness-evidence",
            "score",
            ("verified_stages", "blocked_stages", "executive_signed"),
            "If executive signed, grade min(4, max(0, verified stages minus blocked stages)); otherwise grade zero. Across admissible worlds report the minimum grade.",
            "If executive signed, grade min(4, verified stages); otherwise grade zero.",
        ),
        Policy(
            "recovery-progress",
            "score",
            ("restored_sites", "overdue_tasks", "verified"),
            "If verified, grade max(0, min(4, restored sites minus overdue tasks)); otherwise grade zero. Across admissible worlds report the minimum grade.",
            "If verified, grade min(4, restored sites); otherwise grade zero.",
        ),
        Policy(
            "resilience-grid",
            "score",
            ("independent_feeds", "tested_islands", "verified"),
            "If verified, grade min(4, independent feeds plus tested islands); otherwise grade zero. Across admissible worlds report the minimum grade.",
            "If verified, grade min(4, independent feeds); otherwise grade zero.",
        ),
    )
}


def validate(policy: Policy, facts: dict[str, Any]) -> None:
    if set(facts) != set(policy.fields):
        raise ValueError("Policy field schema differs from source facts")
    if policy.kind == "choice":
        a, b, allowed = (facts[field] for field in policy.fields)
        if (
            not isinstance(a, dict)
            or not isinstance(b, dict)
            or not isinstance(allowed, list)
            or not 3 <= len(a) <= 5
            or set(a) != set(b)
            or not set(allowed) <= set(a)
            or not allowed
            or len(allowed) != len(set(allowed))
            or any(not isinstance(k, str) or not k.isalpha() or k == "hold" for k in a)
            or any(
                type(v) is not int or not 0 <= v <= 200
                for v in [*a.values(), *b.values()]
            )
        ):
            raise ValueError("Invalid choice source register")
        if policy.id == "relief-supplier" and min(a.values()) < 1:
            raise ValueError("Supplier costs must be positive")
        if policy.id == "cold-chain-dispatch" and any(
            v not in (0, 1) for v in b.values()
        ):
            raise ValueError("Seal status must be zero or one")
        return
    if any(type(v) not in (int, bool) for v in facts.values()):
        raise ValueError("Invalid scalar source value")
    if any(type(v) is int and not 0 <= v <= 200 for v in facts.values()):
        raise ValueError("Scalar source value outside domain")
    for key in (
        "seal_intact",
        "bond_current",
        "signed",
        "reviewer_signed",
        "executive_signed",
        "verified",
    ):
        if key in facts and type(facts[key]) is not bool:
            raise ValueError("Signature or status source must be boolean")
    if policy.id in {
        "incident-severity",
        "inspection-deficit",
        "readiness-evidence",
        "recovery-progress",
        "resilience-grid",
    }:
        if any(type(v) is int and v > 8 for v in facts.values()):
            raise ValueError("Score source outside grade domain")


def evaluate(
    policy: Policy, facts: dict[str, Any], *, archived: bool = False
) -> str | bool | int:
    validate(policy, facts)
    p = policy.id
    if p == "relief-supplier":
        keys = facts["certified"]
        return (
            min(keys, key=lambda k: (facts["costs"][k], k))
            if archived
            else min(
                keys,
                key=lambda k: (-Fraction(facts["capacity"][k], facts["costs"][k]), k),
            )
        )
    if p == "cold-chain-dispatch":
        keys = [k for k in facts["authorized"] if archived or facts["seal_ok"][k] == 1]
        return (
            min(keys, key=lambda k: (facts["drive_minutes"][k], k)) if keys else "hold"
        )
    if p == "grant-panel":
        keys = [
            k for k in facts["eligible"] if archived or facts["matching_funds"][k] >= 50
        ]
        return (
            (
                min(keys, key=lambda k: (-facts["matching_funds"][k], k))
                if archived
                else min(keys, key=lambda k: (-facts["merit"][k], k))
            )
            if keys
            else "hold"
        )
    if p == "backup-routing":
        keys = facts["available"]
        return min(
            keys,
            key=lambda k: (
                (facts["restore_hours"][k], facts["outage_risk"][k], k)
                if archived
                else (facts["outage_risk"][k], facts["restore_hours"][k], k)
            ),
        )
    if p == "sterile-release":
        return (
            facts["signoffs"] >= (1 if archived else 2)
            and facts["open_defects"] <= (1 if archived else 0)
            and facts["seal_intact"]
        )
    if p == "safe-exit":
        return facts["inspected_exits"] >= math.ceil(facts["occupied_levels"] / 2) and (
            archived or facts["certified_capacity"] >= 12 * facts["occupied_levels"]
        )
    if p == "supplier-renewal":
        return (
            facts["delivery_pct"] >= (90 if archived else 96)
            and facts["open_complaints"] <= 2
            and facts["bond_current"]
        )
    if p == "incident-severity":
        return (
            min(
                4, max(0, facts["exposure"] - (0 if archived else facts["containment"]))
            )
            if facts["signed"]
            else 0
        )
    if p == "inspection-deficit":
        return (
            max(0, 4 - (1 if archived else 2) * facts["critical"] - facts["major"])
            if facts["reviewer_signed"]
            else 0
        )
    if p == "readiness-evidence":
        return (
            min(
                4,
                max(
                    0,
                    facts["verified_stages"]
                    - (0 if archived else facts["blocked_stages"]),
                ),
            )
            if facts["executive_signed"]
            else 0
        )
    if p == "recovery-progress":
        return (
            min(
                4,
                max(
                    0,
                    facts["restored_sites"]
                    - (0 if archived else facts["overdue_tasks"]),
                ),
            )
            if facts["verified"]
            else 0
        )
    if p == "resilience-grid":
        return (
            min(
                4,
                facts["independent_feeds"]
                + (0 if archived else facts["tested_islands"]),
            )
            if facts["verified"]
            else 0
        )
    raise AssertionError(p)


def reference(
    policy: Policy, facts: dict[str, Any], *, archived: bool = False
) -> str | bool | int:
    """Independent, simple reference path for arithmetic/selection checks."""
    validate(policy, facts)
    if policy.kind == "choice":
        keys = facts[policy.fields[2]]
        if policy.id == "cold-chain-dispatch" and not archived:
            keys = [k for k in keys if facts["seal_ok"][k]]
        if policy.id == "grant-panel" and not archived:
            keys = [k for k in keys if facts["matching_funds"][k] >= 50]
        if not keys:
            return "hold"
        ordered = sorted(keys)

        def preference(k: str) -> tuple[Any, ...]:
            if policy.id == "relief-supplier":
                return (
                    (facts["costs"][k], k)
                    if archived
                    else (-facts["capacity"][k] / facts["costs"][k], k)
                )
            if policy.id == "cold-chain-dispatch":
                return facts["drive_minutes"][k], k
            if policy.id == "grant-panel":
                return -facts["matching_funds" if archived else "merit"][k], k
            return (
                (facts["restore_hours"][k], facts["outage_risk"][k], k)
                if archived
                else (facts["outage_risk"][k], facts["restore_hours"][k], k)
            )

        return sorted(ordered, key=preference)[0]
    p = policy.id
    f = facts
    if p == "sterile-release":
        return bool(
            f["seal_intact"]
            and f["signoffs"] >= (1 if archived else 2)
            and f["open_defects"] <= (1 if archived else 0)
        )
    if p == "safe-exit":
        return bool(
            2 * f["inspected_exits"] >= f["occupied_levels"]
            and (archived or f["certified_capacity"] // 12 >= f["occupied_levels"])
        )
    if p == "supplier-renewal":
        return bool(
            f["bond_current"]
            and f["open_complaints"] < 3
            and f["delivery_pct"] >= (90 if archived else 96)
        )
    if p == "incident-severity":
        return (
            min(4, max(0, f["exposure"] - (0 if archived else f["containment"])))
            if f["signed"]
            else 0
        )
    if p == "inspection-deficit":
        return (
            max(0, 4 - f["major"] - f["critical"] * (1 if archived else 2))
            if f["reviewer_signed"]
            else 0
        )
    if p == "readiness-evidence":
        return (
            sorted(
                (0, f["verified_stages"] - (0 if archived else f["blocked_stages"]), 4)
            )[1]
            if f["executive_signed"]
            else 0
        )
    if p == "recovery-progress":
        return (
            sorted(
                (0, f["restored_sites"] - (0 if archived else f["overdue_tasks"]), 4)
            )[1]
            if f["verified"]
            else 0
        )
    if p == "resilience-grid":
        return (
            min(4, f["independent_feeds"] + (0 if archived else f["tested_islands"]))
            if f["verified"]
            else 0
        )
    raise AssertionError(p)


def aggregate(policy: Policy, outputs: list[str | bool | int]) -> str | bool | int:
    if not outputs:
        raise ValueError("No admissible completions")
    if policy.kind == "choice":
        return outputs[0] if all(value == outputs[0] for value in outputs) else "hold"
    if policy.kind == "noul":
        return all(value is True for value in outputs)
    return min(outputs)
