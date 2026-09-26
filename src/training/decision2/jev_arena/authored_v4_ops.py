"""Sixty formal decision operations for the authored v4 candidate.

The operation registry is intentionally semantic: different domains and
document layouts never increment the template count. Every operation has a
distinct policy and fact dependency; behavioral fingerprints are tested.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import copy
import itertools
import random
from typing import Any

DOMAINS = ("access", "archive", "delivery", "finance", "incident", "release")
TAGS = ("safety", "legal", "capacity", "quality", "privacy", "timing")
OPTIONS = ("A", "B", "C", "D")
WORKFLOW_ORDERS = tuple(
    ("intake", first, second, "approve", "close")
    for first in ("triage", "review", "verify")
    for second in ("review", "verify", "legal", "quality", "security", "capacity")
    if first != second
)
DOMAIN_BY_KIND = {
    "choice": (
        "delivery",
        "finance",
        "finance",
        "access",
        "finance",
        "delivery",
        "delivery",
        "access",
        "release",
        "archive",
        "release",
        "delivery",
        "incident",
        "finance",
        "incident",
        "delivery",
        "access",
        "archive",
        "access",
        "delivery",
    ),
    "noul": (
        "access",
        "incident",
        "incident",
        "access",
        "access",
        "archive",
        "release",
        "delivery",
        "archive",
        "release",
        "release",
        "release",
        "archive",
        "finance",
        "archive",
        "access",
        "access",
        "delivery",
        "release",
        "access",
    ),
    "score": (
        "archive",
        "finance",
        "release",
        "incident",
        "finance",
        "delivery",
        "archive",
        "incident",
        "finance",
        "release",
        "archive",
        "access",
        "delivery",
        "finance",
        "delivery",
        "finance",
        "release",
        "release",
        "finance",
        "incident",
    ),
}


@dataclass(frozen=True)
class Operation:
    id: str
    kind: str
    domain: str
    policy: str
    context_fields: tuple[str, ...]
    option_fields: tuple[str, ...] = ()


def _ops(kind: str, rows: list[tuple[str, str, str, str]]) -> list[Operation]:
    result = []
    for index, (name, policy, context, candidate) in enumerate(rows):
        result.append(
            Operation(
                f"{kind}{index + 1:02d}-{name}",
                kind,
                DOMAIN_BY_KIND[kind][index],
                policy,
                tuple(context.split()) if context else (),
                tuple(candidate.split()) if candidate else (),
            )
        )
    return result


CHOICE = _ops(
    "choice",
    [
        (
            "first-eligible",
            "Follow the stated priority order and select the first available, approved provider.",
            "order",
            "available approved",
        ),
        (
            "cheapest-feasible",
            "Among providers with enough capacity and risk at most two, choose the lowest cost.",
            "demand",
            "capacity risk cost",
        ),
        (
            "highest-benefit-under-budget",
            "Among providers within the budget, choose the largest benefit.",
            "budget",
            "cost benefit",
        ),
        (
            "approval-priority",
            "Among approved providers, choose the smallest priority number, then the largest benefit.",
            "",
            "approved priority benefit",
        ),
        (
            "pareto-risk",
            "Discard a provider if another has no higher cost and no lower benefit, with at least one strict improvement; among survivors choose the lowest risk.",
            "",
            "cost benefit risk",
        ),
        (
            "earliest-feasible-slot",
            "Choose the available interval beginning no earlier than the request day and ending by the deadline; take the earliest start.",
            "request_day deadline",
            "available start end",
        ),
        (
            "minimal-adequate-capacity",
            "Among available providers meeting demand, choose the smallest adequate capacity.",
            "demand",
            "available capacity",
        ),
        (
            "required-tag-coverage",
            "Choose the provider covering the most required tags; break ties by lower cost.",
            "required_tags",
            "tags cost",
        ),
        (
            "dependency-unlock",
            "Among approved providers with no unmet dependencies, choose the one unlocking the most downstream work.",
            "",
            "approved unmet downstream",
        ),
        (
            "authority-then-recency",
            "Among approved records, choose highest source authority, then latest timestamp.",
            "",
            "approved authority timestamp",
        ),
        (
            "compatible-version",
            "Among approved versions no newer than the required version, choose the newest.",
            "required_version",
            "approved version",
        ),
        (
            "fallback-chain",
            "Inspect the fallback order and select the first available provider; approval is irrelevant for this fallback.",
            "order",
            "available",
        ),
        (
            "minimax-loss",
            "Among approved plans choose the smallest stated worst-case loss.",
            "",
            "approved loss",
        ),
        (
            "weighted-utility",
            "Among approved plans maximize two times benefit minus cost minus risk.",
            "",
            "approved benefit cost risk",
        ),
        (
            "severity-urgency",
            "Among approved cases select the greatest product of severity and urgency.",
            "",
            "approved severity urgency",
        ),
        (
            "shortest-covering-window",
            "Choose the available interval that completely covers the requested window and has the shortest duration.",
            "request_window",
            "available start end",
        ),
        (
            "quorum-then-benefit",
            "Only providers meeting the vote quorum qualify; among them choose the greatest benefit.",
            "quorum",
            "votes benefit",
        ),
        (
            "canonical-record",
            "For duplicate records, prefer a confirmed record, then higher authority, then later timestamp.",
            "",
            "confirmed authority timestamp",
        ),
        (
            "borda-ballots",
            "Award three, two, one, and zero points per ballot rank; select the highest total across all ballots.",
            "ballots",
            "",
        ),
        (
            "fair-share-deviation",
            "Among available providers meeting minimum demand, choose capacity closest to the fair-share target.",
            "demand fair_share",
            "available capacity",
        ),
    ],
)


NOUL = _ops(
    "noul",
    [
        (
            "all-required",
            "Are all four checks named by indices verified?",
            "checks indices",
            "",
        ),
        (
            "any-trigger",
            "Is at least one of the four indexed triggers verified?",
            "checks indices",
            "",
        ),
        (
            "exactly-two",
            "Are exactly two of the four indexed checks verified?",
            "checks indices",
            "",
        ),
        (
            "threshold-quorum",
            "Do at least the specified number of the four indexed checks pass?",
            "checks indices quorum",
            "",
        ),
        (
            "conditional-obligation",
            "Evaluate (not first) OR second for the first two indexed checks. A false first check makes this test true.",
            "checks indices",
            "",
        ),
        (
            "paired-equivalence",
            "Do the checks at the first two listed indices have the same verified status?",
            "checks indices",
            "",
        ),
        (
            "event-order",
            "In dates, the first timestamp is authorization and the second is execution; did authorization occur strictly before execution?",
            "dates",
            "",
        ),
        (
            "window-containment",
            "The first interval in intervals is the approved window; is request_window fully contained within it?",
            "intervals request_window",
            "",
        ),
        (
            "freshness-cutoff",
            "The first timestamp in dates is the evidence date; is it on or after cutoff?",
            "dates cutoff",
            "",
        ),
        (
            "major-minor-version",
            "The first version is available and the second is required; do they share the major version with available minor at least required minor?",
            "versions",
            "",
        ),
        (
            "path-reachability",
            "Each edge pair [u,v] points from u to v. Can node zero reach node four by following these directed edges?",
            "edges",
            "",
        ),
        ("cycle-freedom", "Is the directed dependency graph acyclic?", "edges", ""),
        ("unique-identifiers", "Are all four submitted identifiers unique?", "ids", ""),
        (
            "budget-sum",
            "Is the sum of the four costs within the approved budget?",
            "costs budget",
            "",
        ),
        (
            "record-consistency",
            "Do repeated record keys agree on their values?",
            "records",
            "",
        ),
        (
            "scope-inclusion",
            "Are all requested capabilities within the authorized scope?",
            "required_tags authorized_tags",
            "",
        ),
        (
            "distinct-attestors",
            "Match signers and attested flags by list position; are there at least two distinct signers with verified attestations?",
            "signers attested",
            "",
        ),
        (
            "interval-nonoverlap",
            "Treat the four booked intervals as half-open [start, end) periods; are they pairwise non-overlapping?",
            "intervals",
            "",
        ),
        (
            "dependency-completion",
            "Are all named prerequisites completed, with none of those prerequisites revoked?",
            "required_tags completed_tags revoked_tags",
            "",
        ),
        (
            "exception-exemption",
            "Does the exemption apply, or otherwise are all four indexed checks verified?",
            "exempt checks indices",
            "",
        ),
    ],
)


SCORE = _ops(
    "score",
    [
        (
            "verified-count",
            "Give one point for each verified check at the four listed indices; range zero to four.",
            "checks indices",
            "",
        ),
        (
            "weighted-clamped-sum",
            "Multiply each value by its matching weight, sum, then clamp to the grade range zero through four.",
            "values weights",
            "",
        ),
        (
            "bottleneck-minimum",
            "The grade is the lowest of four subsystem grades.",
            "grades",
            "",
        ),
        (
            "worst-severity",
            "The grade is the highest reported severity among four events.",
            "grades",
            "",
        ),
        (
            "ratio-quarter-band",
            "The grade is floor(four times successes divided by trials), capped at four.",
            "successes trials",
            "",
        ),
        (
            "lateness-bucket",
            "Score lateness: zero if on time, otherwise ceil(days late divided by three), capped at four.",
            "deadline finish_day",
            "",
        ),
        (
            "recency-decay",
            "Evidence age is report_day minus observed_day; start at four and subtract one for every full five days of age, floored at zero.",
            "report_day observed_day",
            "",
        ),
        (
            "risk-product-band",
            "Multiply likelihood, impact, and case risk scale; divide by eight and round up, then cap at four.",
            "likelihood impact risk_scale",
            "",
        ),
        (
            "credit-minus-penalty",
            "Count verified credits at the four listed indices, subtract penalties, and clamp to zero through four.",
            "checks indices penalties",
            "",
        ),
        (
            "dependency-longest-path",
            "Each edge pair [u,v] points from u to v. Grade the longest simple directed path from node zero (never revisit a node), capped at four edges.",
            "edges",
            "",
        ),
        (
            "distinct-source-count",
            "Match signers and attested flags by list position; grade distinct verified signer names, capped at four.",
            "signers attested",
            "",
        ),
        (
            "coverage-fraction-band",
            "Grade floor(four times covered required tags divided by all required tags), capped at four.",
            "required_tags covered_tags",
            "",
        ),
        (
            "interval-overlap-days",
            "Use the first two half-open [start, end) intervals in intervals; grade their shared days, capped at four.",
            "intervals",
            "",
        ),
        (
            "forecast-error-band",
            "Give four only for an exact forecast. Otherwise subtract ceil(absolute error divided by three) from four, floored at zero.",
            "forecast actual",
            "",
        ),
        (
            "inventory-coverage",
            "Score floor(stock divided by daily use), capped at four.",
            "stock daily_use",
            "",
        ),
        (
            "peer-percentile",
            "Count peers strictly below the target value; cap at four.",
            "target_value peer_values",
            "",
        ),
        (
            "trend-improvement",
            "Grade positive change from first to last measurement divided by two, floored and capped at four.",
            "history",
            "",
        ),
        (
            "workflow-stage",
            "Grade the zero-based position of the current stage in the stated five-stage workflow order.",
            "workflow_order current_stage",
            "",
        ),
        (
            "budget-variance",
            "Begin at four and subtract one for every full five units over budget; no penalty under budget.",
            "budget actual_cost",
            "",
        ),
        (
            "workload-balance",
            "Start at four and subtract the range of four team loads, floored at zero.",
            "loads",
            "",
        ),
    ],
)


OPERATIONS = tuple(CHOICE + NOUL + SCORE)
BY_ID = {op.id: op for op in OPERATIONS}
assert len(OPERATIONS) == len(BY_ID) == 60
assert Counter(op.kind for op in OPERATIONS) == {"choice": 20, "noul": 20, "score": 20}


def _base_facts(rng: random.Random) -> dict[str, Any]:
    order = list(OPTIONS)
    rng.shuffle(order)
    tags = list(TAGS)
    rng.shuffle(tags)
    req = sorted(tags[: rng.randint(2, 4)])
    ballots = []
    for _ in range(5):
        ballot = list(OPTIONS)
        rng.shuffle(ballot)
        ballots.append(ballot)
    options = []
    for key in OPTIONS:
        start = rng.randint(1, 18)
        options.append(
            {
                "id": key,
                "cost": rng.randint(3, 20),
                "benefit": rng.randint(2, 20),
                "risk": rng.randint(1, 4),
                "capacity": rng.randint(3, 20),
                "priority": rng.randint(1, 4),
                "approved": rng.choice((True, False)),
                "available": rng.choice((True, False)),
                "version": rng.randint(1, 8),
                "tags": sorted(rng.sample(TAGS, rng.randint(1, 4))),
                "start": start,
                "end": start + rng.randint(2, 8),
                "votes": rng.randint(0, 5),
                "downstream": rng.randint(0, 12),
                "unmet": rng.randint(0, 2),
                "authority": rng.randint(1, 4),
                "timestamp": rng.randint(1, 30),
                "loss": rng.randint(1, 20),
                "severity": rng.randint(1, 4),
                "urgency": rng.randint(1, 4),
                "confirmed": rng.choice((True, False)),
            }
        )
    intervals = []
    for _ in range(4):
        start = rng.randint(0, 20)
        intervals.append([start, start + rng.randint(1, 7)])
    ids = [f"R{rng.randint(1, 5)}" for _ in range(4)]
    signers = [f"P{rng.randint(1, 5)}" for _ in range(4)]
    dates = [rng.randint(0, 30) for _ in range(4)]
    edges = [
        [a, b] for a in range(5) for b in range(5) if a != b and rng.random() < 0.17
    ]
    checks = [True] * rng.randint(0, 8)
    checks.extend([False] * (8 - len(checks)))
    rng.shuffle(checks)
    workflow_order = list(rng.choice(WORKFLOW_ORDERS))
    report_day = rng.randint(30, 60)
    base = {
        "options": options,
        "order": order,
        "budget": rng.randint(15, 40),
        "demand": rng.randint(5, 15),
        "request_day": rng.randint(1, 12),
        "deadline": rng.randint(16, 28),
        "required_tags": req,
        "fair_share": rng.randint(7, 18),
        "quorum": rng.randint(2, 4),
        "required_version": rng.randint(3, 8),
        "request_window": [rng.randint(3, 7), rng.randint(10, 15)],
        "ballots": ballots,
        "checks": checks,
        "values": [rng.randint(0, 2) for _ in range(4)],
        "indices": sorted(rng.sample(range(8), 4)),
        "weights": [rng.randint(1, 2) for _ in range(4)],
        "grades": [rng.randint(0, 4) for _ in range(4)],
        "successes": rng.randint(0, 12),
        "trials": rng.randint(13, 20),
        "finish_day": rng.randint(10, 36),
        "report_day": report_day,
        "observed_day": report_day - rng.randint(0, 25),
        "likelihood": rng.randint(1, 4),
        "impact": rng.randint(1, 4),
        "risk_scale": rng.randint(1, 5),
        "penalties": rng.randint(0, 4),
        "edges": edges,
        "signers": signers,
        "attested": [rng.choice((True, False)) for _ in range(4)],
        "covered_tags": sorted(rng.sample(req, rng.randint(0, len(req)))),
        "intervals": intervals,
        "forecast": rng.randint(10, 30),
        "actual": rng.randint(10, 30),
        "stock": rng.randint(5, 40),
        "daily_use": rng.randint(2, 10),
        "target_value": rng.randint(0, 20),
        "peer_values": [rng.randint(0, 20) for _ in range(5)],
        "history": [rng.randint(0, 20), rng.randint(0, 20)],
        "workflow_order": workflow_order,
        "current_stage": rng.choice(workflow_order),
        "actual_cost": rng.randint(15, 55),
        "loads": [rng.randint(0, 5) for _ in range(4)],
        "dates": dates,
        "cutoff": rng.randint(5, 25),
        "versions": [[rng.randint(1, 3), rng.randint(0, 8)] for _ in range(2)],
        "ids": ids,
        "costs": [rng.randint(3, 12) for _ in range(4)],
        "records": [{"key": ids[i], "value": rng.randint(0, 2)} for i in range(4)],
        "authorized_tags": sorted(rng.sample(TAGS, rng.randint(2, 6))),
        "completed_tags": sorted(rng.sample(TAGS, rng.randint(2, 6))),
        "revoked_tags": sorted(rng.sample(TAGS, rng.randint(0, 2))),
        "exempt": rng.choice((True, False)),
    }
    return base


def generate_facts(op: Operation, rng: random.Random) -> dict[str, Any]:
    base = _base_facts(rng)
    name = op.id.partition("-")[2]
    if name == "shortest-covering-window":
        left, right = base["request_window"]
        if rng.random() < 0.8:
            winner = rng.choice(base["options"])
            winner.update(
                available=True,
                start=left - rng.randint(0, 2),
                end=right + rng.randint(0, 2),
            )
    if name == "window-containment":
        left, right = base["request_window"]
        base["intervals"][0] = (
            [left - rng.randint(0, 3), right + rng.randint(0, 3)]
            if rng.randrange(2)
            else [left + 1, right - 1]
        )
    if name == "interval-nonoverlap":
        start = rng.randint(0, 5)
        base["intervals"] = [
            [start + 3 * i, start + 3 * i + rng.randint(1, 3)] for i in range(4)
        ]
        if rng.randrange(2):
            base["intervals"][rng.randrange(1, 4)][0] -= 2
    if name == "weighted-clamped-sum":
        goal = rng.randrange(1, 5)
        weights = [rng.randint(1, 2) for _ in range(4)]
        values = [0] * 4
        remaining = goal
        for index in rng.sample(range(4), 4):
            if weights[index] <= remaining and rng.random() < 0.8:
                values[index] = 1
                remaining -= weights[index]
        if remaining:
            index = next(index for index, value in enumerate(values) if value == 0)
            weights[index] = 1
            values[index] = remaining
        base["weights"], base["values"] = weights, values
    if name == "forecast-error-band":
        base["actual"] = base["forecast"] + rng.choice((0, 1, 2, 3, 4, 5, 6, 7, 9, 12))
    facts = {field: base[field] for field in op.context_fields}
    if op.kind == "choice" and op.option_fields:
        facts["options"] = [
            {"id": option["id"], **{field: option[field] for field in op.option_fields}}
            for option in base["options"]
        ]
    return facts


def _has_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_has_missing(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_missing(item) for item in value)
    return False


def _paths(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    if isinstance(value, dict):
        return list(
            itertools.chain.from_iterable(
                _paths(part, path + (key,))
                for key, part in value.items()
                if key not in ("id", "key")
            )
        )
    if isinstance(value, list):
        paths = []
        if (
            value
            and all(isinstance(part, str) for part in value)
            and path
            and path[0]
            in (
                "required_tags",
                "authorized_tags",
                "completed_tags",
                "revoked_tags",
                "covered_tags",
            )
        ):
            paths.append(path)
        if (
            path
            and path[0] in ("ballots", "edges", "workflow_order")
            and len(path) == 1
        ):
            paths.append(path)
        if not (path and path[0] in ("ballots", "edges", "indices", "workflow_order")):
            for index, part in enumerate(value):
                paths.extend(_paths(part, path + (index,)))
        return paths
    return [path] if isinstance(value, (bool, int, str)) else []


def _at(value: Any, path: tuple[Any, ...]) -> Any:
    for component in path:
        value = value[component]
    return value


def _set(value: Any, path: tuple[Any, ...], replacement: Any) -> None:
    holder = value
    for component in path[:-1]:
        holder = holder[component]
    holder[path[-1]] = replacement


def _alternatives(facts: dict[str, Any], path: tuple[Any, ...]) -> list[Any]:
    original = _at(facts, path)
    root = path[0]
    if isinstance(original, bool):
        return [not original]
    if isinstance(original, int):
        if root == "edges":
            return []
        values = (0, 1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50)
        return [value for value in values if value != original]
    if isinstance(original, str):
        if root == "current_stage":
            values = facts["workflow_order"]
        elif root in ("signers",):
            values = [f"P{i}" for i in range(1, 6)]
        elif root in ("ids",):
            values = [f"R{i}" for i in range(1, 6)]
        elif (
            root
            in (
                "required_tags",
                "authorized_tags",
                "completed_tags",
                "revoked_tags",
                "covered_tags",
            )
            or "tags" in path
        ):
            values = TAGS
        else:
            values = OPTIONS
        return [value for value in values if value != original]
    if isinstance(original, list) and root == "edges":
        candidates = []
        for edge in ([0, 4], [4, 0], [0, 1], [1, 4]):
            variant = [list(item) for item in original]
            if edge in variant:
                variant.remove(edge)
            else:
                variant.append(edge)
            candidates.append(variant)
        return candidates
    if isinstance(original, list) and root == "ballots":
        return [
            [
                list(reversed(ballot)) if index == changed else list(ballot)
                for index, ballot in enumerate(original)
            ]
            for changed in range(len(original))
        ]
    if isinstance(original, list) and root == "workflow_order":
        return [list(order) for order in WORKFLOW_ORDERS if list(order) != original]
    if isinstance(original, list):
        candidate = list(original)
        if candidate:
            candidate.pop()
        addable = next((item for item in TAGS if item not in original), None)
        return [candidate, sorted(original + [addable])] if addable else [candidate]
    return []


def possible_worlds(facts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Expand two explicitly admissible values for one unreported fact."""
    uncertainty = facts.get("admissible_uncertainty")
    if not isinstance(uncertainty, dict) or set(uncertainty) != {"path", "values"}:
        raise ValueError("Partial evidence requires one admissible uncertainty")
    path = tuple(uncertainty["path"])
    values = uncertainty["values"]
    if (
        not path
        or len(values) != 2
        or _at(facts, path) is not None
        or values[0] == values[1]
    ):
        raise ValueError("Malformed admissible uncertainty")
    worlds = []
    for candidate in values:
        world = copy.deepcopy(
            {key: val for key, val in facts.items() if key != "admissible_uncertainty"}
        )
        _set(world, path, candidate)
        worlds.append(world)
    return worlds[0], worlds[1]


def _valid_facts(facts: dict[str, Any]) -> bool:
    if (
        "request_window" in facts
        and facts["request_window"][0] >= facts["request_window"][1]
    ):
        return False
    if "intervals" in facts and any(a >= b for a, b in facts["intervals"]):
        return False
    if "options" in facts and any(
        row.get("start", 0) >= row.get("end", 1000) for row in facts["options"]
    ):
        return False
    if "order" in facts and set(facts["order"]) != set(OPTIONS):
        return False
    if "ballots" in facts and any(
        set(ballot) != set(OPTIONS) or len(ballot) != 4 for ballot in facts["ballots"]
    ):
        return False
    if "edges" in facts and any(
        a == b or a not in range(5) or b not in range(5) for a, b in facts["edges"]
    ):
        return False
    if (
        "current_stage" in facts
        and facts["current_stage"] not in facts["workflow_order"]
    ):
        return False
    if (
        "workflow_order" in facts
        and tuple(facts["workflow_order"]) not in WORKFLOW_ORDERS
    ):
        return False
    if "trials" in facts and facts["trials"] < 1:
        return False
    if "daily_use" in facts and facts["daily_use"] < 1:
        return False
    if "required_tags" in facts and not facts["required_tags"]:
        return False
    if "grades" in facts and any(value not in range(5) for value in facts["grades"]):
        return False
    return True


def inject_missing(
    op: Operation, facts: dict[str, Any], rng: random.Random, *, resolved: bool
) -> dict[str, Any] | None:
    """Construct a decision-relevant partial report with an auditable world pair."""
    paths = _paths(facts)
    rng.shuffle(paths)
    baseline = evaluate(op, facts)
    fallback = {"choice": "hold", "noul": False, "score": 0}[op.kind]
    for path in paths:
        original = _at(facts, path)
        alternatives = _alternatives(facts, path)
        rng.shuffle(alternatives)
        for alternative in alternatives:
            complete = copy.deepcopy(facts)
            _set(complete, path, alternative)
            if not _valid_facts(complete):
                continue
            try:
                changed = evaluate(op, complete)
            except (TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
                continue
            if resolved and (baseline != changed or baseline == fallback):
                continue
            if not resolved and baseline == changed:
                continue
            partial = copy.deepcopy(facts)
            _set(partial, path, None)
            partial["admissible_uncertainty"] = {
                "path": list(path),
                "values": [original, alternative],
            }
            return partial
    return None


def _pick(rows: list[dict[str, Any]], key: Any, *, reverse: bool = False) -> str:
    if not rows:
        return "hold"
    return sorted(rows, key=lambda row: (key(row), row["id"]), reverse=reverse)[0]["id"]


def _reach(edges: list[list[int]], start: int, goal: int) -> bool:
    stack, seen = [start], set()
    while stack:
        node = stack.pop()
        if node == goal:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(b for a, b in edges if a == node)
    return False


def _longest_from_zero(edges: list[list[int]]) -> int:
    # Longest simple path remains defined even when a decoy graph has a cycle.
    def visit(node: int, seen: frozenset[int]) -> int:
        return max(
            (1 + visit(b, seen | {b}) for a, b in edges if a == node and b not in seen),
            default=0,
        )

    return visit(0, frozenset({0}))


def evaluate(op: Operation, facts: dict[str, Any]) -> str | bool | int:
    """Direct oracle over the private typed fact specification."""
    if "admissible_uncertainty" in facts:
        answers = [evaluate(op, world) for world in possible_worlds(facts)]
        return (
            answers[0]
            if answers[0] == answers[1]
            else {"choice": "hold", "noul": False, "score": 0}[op.kind]
        )
    if _has_missing(facts):
        return "hold" if op.kind == "choice" else False if op.kind == "noul" else 0
    name = op.id.split("-", 1)[1]
    if op.kind == "choice":
        rows = facts.get("options", [])
        if name == "first-eligible":
            return next(
                (
                    key
                    for key in facts["order"]
                    if any(
                        row["id"] == key and row["available"] and row["approved"]
                        for row in rows
                    )
                ),
                "hold",
            )
        if name == "cheapest-feasible":
            return _pick(
                [
                    r
                    for r in rows
                    if r["capacity"] >= facts["demand"] and r["risk"] <= 2
                ],
                lambda r: r["cost"],
            )
        if name == "highest-benefit-under-budget":
            return _pick(
                [r for r in rows if r["cost"] <= facts["budget"]],
                lambda r: -r["benefit"],
            )
        if name == "approval-priority":
            return _pick(
                [r for r in rows if r["approved"]],
                lambda r: (r["priority"], -r["benefit"]),
            )
        if name == "pareto-risk":
            front = [
                r
                for r in rows
                if not any(
                    other["id"] != r["id"]
                    and other["cost"] <= r["cost"]
                    and other["benefit"] >= r["benefit"]
                    and (other["cost"] < r["cost"] or other["benefit"] > r["benefit"])
                    for other in rows
                )
            ]
            return _pick(front, lambda r: r["risk"])
        if name == "earliest-feasible-slot":
            return _pick(
                [
                    r
                    for r in rows
                    if r["available"]
                    and r["start"] >= facts["request_day"]
                    and r["end"] <= facts["deadline"]
                ],
                lambda r: r["start"],
            )
        if name == "minimal-adequate-capacity":
            return _pick(
                [
                    r
                    for r in rows
                    if r["available"] and r["capacity"] >= facts["demand"]
                ],
                lambda r: r["capacity"],
            )
        if name == "required-tag-coverage":
            required = set(facts["required_tags"])
            return _pick(rows, lambda r: (-len(required & set(r["tags"])), r["cost"]))
        if name == "dependency-unlock":
            return _pick(
                [r for r in rows if r["approved"] and r["unmet"] == 0],
                lambda r: -r["downstream"],
            )
        if name == "authority-then-recency":
            return _pick(
                [r for r in rows if r["approved"]],
                lambda r: (-r["authority"], -r["timestamp"]),
            )
        if name == "compatible-version":
            return _pick(
                [
                    r
                    for r in rows
                    if r["approved"] and r["version"] <= facts["required_version"]
                ],
                lambda r: -r["version"],
            )
        if name == "fallback-chain":
            return next(
                (
                    key
                    for key in facts["order"]
                    if any(row["id"] == key and row["available"] for row in rows)
                ),
                "hold",
            )
        if name == "minimax-loss":
            return _pick([r for r in rows if r["approved"]], lambda r: r["loss"])
        if name == "weighted-utility":
            return _pick(
                [r for r in rows if r["approved"]],
                lambda r: -(2 * r["benefit"] - r["cost"] - r["risk"]),
            )
        if name == "severity-urgency":
            return _pick(
                [r for r in rows if r["approved"]],
                lambda r: -(r["severity"] * r["urgency"]),
            )
        if name == "shortest-covering-window":
            left, right = facts["request_window"]
            return _pick(
                [
                    r
                    for r in rows
                    if r["available"] and r["start"] <= left and r["end"] >= right
                ],
                lambda r: r["end"] - r["start"],
            )
        if name == "quorum-then-benefit":
            return _pick(
                [r for r in rows if r["votes"] >= facts["quorum"]],
                lambda r: -r["benefit"],
            )
        if name == "canonical-record":
            return _pick(
                rows, lambda r: (-int(r["confirmed"]), -r["authority"], -r["timestamp"])
            )
        if name == "borda-ballots":
            points = {key: 0 for key in OPTIONS}
            for ballot in facts["ballots"]:
                for index, key in enumerate(ballot):
                    points[key] += 3 - index
            return sorted(OPTIONS, key=lambda key: (-points[key], key))[0]
        if name == "fair-share-deviation":
            return _pick(
                [
                    r
                    for r in rows
                    if r["available"] and r["capacity"] >= facts["demand"]
                ],
                lambda r: abs(r["capacity"] - facts["fair_share"]),
            )
    if op.kind == "noul":
        checks = facts.get("checks", [])
        chosen = (
            [checks[index] for index in facts["indices"]] if "indices" in facts else []
        )
        if name == "all-required":
            return all(chosen)
        if name == "any-trigger":
            return any(chosen)
        if name == "exactly-two":
            return sum(chosen) == 2
        if name == "threshold-quorum":
            return sum(chosen) >= facts["quorum"]
        if name == "conditional-obligation":
            return not chosen[0] or chosen[1]
        if name == "paired-equivalence":
            return chosen[0] == chosen[1]
        if name == "event-order":
            return facts["dates"][0] < facts["dates"][1]
        if name == "window-containment":
            a, b = facts["intervals"][0]
            c, d = facts["request_window"]
            return a <= c and d <= b
        if name == "freshness-cutoff":
            return facts["dates"][0] >= facts["cutoff"]
        if name == "major-minor-version":
            available, required = facts["versions"]
            return available[0] == required[0] and available[1] >= required[1]
        if name == "path-reachability":
            return _reach(facts["edges"], 0, 4)
        if name == "cycle-freedom":
            edges = facts["edges"]
            return not any(_reach(edges, b, a) for a, b in edges)
        if name == "unique-identifiers":
            return len(set(facts["ids"])) == len(facts["ids"])
        if name == "budget-sum":
            return sum(facts["costs"]) <= facts["budget"]
        if name == "record-consistency":
            grouped: dict[str, set[int]] = {}
            for row in facts["records"]:
                grouped.setdefault(row["key"], set()).add(row["value"])
            return all(len(values) == 1 for values in grouped.values())
        if name == "scope-inclusion":
            return set(facts["required_tags"]) <= set(facts["authorized_tags"])
        if name == "distinct-attestors":
            return (
                len(
                    {
                        name
                        for name, ok in zip(facts["signers"], facts["attested"])
                        if ok
                    }
                )
                >= 2
            )
        if name == "interval-nonoverlap":
            ranges = facts["intervals"]
            return all(
                a[1] <= b[0] or b[1] <= a[0]
                for i, a in enumerate(ranges)
                for b in ranges[i + 1 :]
            )
        if name == "dependency-completion":
            required = set(facts["required_tags"])
            return required <= set(
                facts["completed_tags"]
            ) and not required.intersection(facts["revoked_tags"])
        if name == "exception-exemption":
            return facts["exempt"] or all(chosen)
    if op.kind == "score":
        clamp = lambda value: max(0, min(4, int(value)))
        if name == "verified-count":
            return sum(facts["checks"][index] for index in facts["indices"])
        if name == "weighted-clamped-sum":
            return clamp(sum(a * b for a, b in zip(facts["values"], facts["weights"])))
        if name == "bottleneck-minimum":
            return min(facts["grades"])
        if name == "worst-severity":
            return max(facts["grades"])
        if name == "ratio-quarter-band":
            return clamp(4 * facts["successes"] // facts["trials"])
        if name == "lateness-bucket":
            return clamp((max(0, facts["finish_day"] - facts["deadline"]) + 2) // 3)
        if name == "recency-decay":
            return clamp(4 - (facts["report_day"] - facts["observed_day"]) // 5)
        if name == "risk-product-band":
            return clamp(
                (facts["likelihood"] * facts["impact"] * facts["risk_scale"] + 7) // 8
            )
        if name == "credit-minus-penalty":
            return clamp(
                sum(facts["checks"][index] for index in facts["indices"])
                - facts["penalties"]
            )
        if name == "dependency-longest-path":
            return clamp(_longest_from_zero(facts["edges"]))
        if name == "distinct-source-count":
            return clamp(
                len(
                    {
                        name
                        for name, ok in zip(facts["signers"], facts["attested"])
                        if ok
                    }
                )
            )
        if name == "coverage-fraction-band":
            return clamp(
                4
                * len(set(facts["required_tags"]) & set(facts["covered_tags"]))
                // len(facts["required_tags"])
            )
        if name == "interval-overlap-days":
            (a, b), (c, d) = facts["intervals"][:2]
            return clamp(max(0, min(b, d) - max(a, c)))
        if name == "forecast-error-band":
            return clamp(4 - (abs(facts["forecast"] - facts["actual"]) + 2) // 3)
        if name == "inventory-coverage":
            return clamp(facts["stock"] // facts["daily_use"])
        if name == "peer-percentile":
            return clamp(
                sum(value < facts["target_value"] for value in facts["peer_values"])
            )
        if name == "trend-improvement":
            return clamp(max(0, facts["history"][-1] - facts["history"][0]) // 2)
        if name == "workflow-stage":
            return facts["workflow_order"].index(facts["current_stage"])
        if name == "budget-variance":
            return clamp(4 - max(0, facts["actual_cost"] - facts["budget"]) // 5)
        if name == "workload-balance":
            return clamp(4 - (max(facts["loads"]) - min(facts["loads"])))
    raise ValueError(f"Unsupported authored v4 operation {op.id}")
