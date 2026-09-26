"""Independent rendered-fact oracle for authored v3 operations.

This module does not invoke the private-spec evaluator. It is deliberately
written with separate selection, graph, and numeric implementations so the
builder can detect disagreements after parsing the visible evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .authored_v3_ops import Operation, OPTIONS


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_missing(part) for part in value.values())
    if isinstance(value, list):
        return any(_missing(part) for part in value)
    return False


def _minimum(rows: list[dict[str, Any]], criterion: Any) -> str:
    if not rows:
        return "hold"
    scored = [(criterion(row), row["id"]) for row in rows]
    return min(scored)[1]


def _reachable(edges: list[list[int]], source: int) -> set[int]:
    reached = {source}
    while True:
        grown = reached | {to for start, to in edges if start in reached}
        if grown == reached:
            return reached
        reached = grown


def _depth(edges: list[list[int]], node: int, visited: set[int]) -> int:
    paths = [
        _depth(edges, to, visited | {to}) + 1
        for start, to in edges
        if start == node and to not in visited
    ]
    return max(paths, default=0)


def evaluate_rendered(op: Operation, facts: dict[str, Any]) -> str | bool | int:
    if _missing(facts):
        return {"choice": "hold", "noul": False, "score": 0}[op.kind]
    name = op.id.partition("-")[2]
    if op.kind == "choice":
        c = facts.get("options", [])
        if name == "first-eligible":
            matches = {row["id"] for row in c if row["available"] and row["approved"]}
            return next((key for key in facts["order"] if key in matches), "hold")
        if name == "cheapest-feasible":
            return _minimum(
                [
                    row
                    for row in c
                    if row["risk"] <= 2 and row["capacity"] >= facts["demand"]
                ],
                lambda row: row["cost"],
            )
        if name == "highest-benefit-under-budget":
            return _minimum(
                [row for row in c if row["cost"] <= facts["budget"]],
                lambda row: -row["benefit"],
            )
        if name == "approval-priority":
            return _minimum(
                [row for row in c if row["approved"]],
                lambda row: (row["priority"], -row["benefit"]),
            )
        if name == "pareto-risk":
            ordered = sorted(
                c, key=lambda row: (row["cost"], -row["benefit"], row["id"])
            )
            survivors = []
            for row in ordered:
                if not any(
                    other["cost"] <= row["cost"]
                    and other["benefit"] >= row["benefit"]
                    and (other["cost"], other["benefit"])
                    != (row["cost"], row["benefit"])
                    for other in ordered
                    if other["id"] != row["id"]
                ):
                    survivors.append(row)
            return _minimum(survivors, lambda row: row["risk"])
        if name == "earliest-feasible-slot":
            return _minimum(
                [
                    row
                    for row in c
                    if row["available"]
                    and facts["request_day"] <= row["start"]
                    and row["end"] <= facts["deadline"]
                ],
                lambda row: row["start"],
            )
        if name == "minimal-adequate-capacity":
            return _minimum(
                [
                    row
                    for row in c
                    if row["available"] and row["capacity"] >= facts["demand"]
                ],
                lambda row: row["capacity"],
            )
        if name == "required-tag-coverage":
            required = set(facts["required_tags"])
            return _minimum(
                c, lambda row: (-len(set(row["tags"]) & required), row["cost"])
            )
        if name == "dependency-unlock":
            return _minimum(
                [row for row in c if row["approved"] and row["unmet"] == 0],
                lambda row: -row["downstream"],
            )
        if name == "authority-then-recency":
            return _minimum(
                [row for row in c if row["approved"]],
                lambda row: (-row["authority"], -row["timestamp"]),
            )
        if name == "compatible-version":
            return _minimum(
                [
                    row
                    for row in c
                    if row["approved"] and row["version"] <= facts["required_version"]
                ],
                lambda row: -row["version"],
            )
        if name == "fallback-chain":
            available = {row["id"] for row in c if row["available"]}
            return next((key for key in facts["order"] if key in available), "hold")
        if name == "minimax-loss":
            return _minimum(
                [row for row in c if row["approved"]], lambda row: row["loss"]
            )
        if name == "weighted-utility":
            return _minimum(
                [row for row in c if row["approved"]],
                lambda row: row["cost"] + row["risk"] - 2 * row["benefit"],
            )
        if name == "severity-urgency":
            return _minimum(
                [row for row in c if row["approved"]],
                lambda row: -row["severity"] * row["urgency"],
            )
        if name == "shortest-covering-window":
            requested = facts["request_window"]
            return _minimum(
                [
                    row
                    for row in c
                    if row["available"]
                    and row["start"] <= requested[0]
                    and requested[1] <= row["end"]
                ],
                lambda row: row["end"] - row["start"],
            )
        if name == "quorum-then-benefit":
            return _minimum(
                [row for row in c if row["votes"] >= facts["quorum"]],
                lambda row: -row["benefit"],
            )
        if name == "canonical-record":
            return _minimum(
                c,
                lambda row: (
                    -int(row["confirmed"]),
                    -row["authority"],
                    -row["timestamp"],
                ),
            )
        if name == "borda-ballots":
            tally = Counter()
            for ballot in facts["ballots"]:
                tally.update({key: 3 - rank for rank, key in enumerate(ballot)})
            return min(OPTIONS, key=lambda key: (-tally[key], key))
        if name == "fair-share-deviation":
            return _minimum(
                [
                    row
                    for row in c
                    if row["available"] and row["capacity"] >= facts["demand"]
                ],
                lambda row: abs(row["capacity"] - facts["fair_share"]),
            )
    elif op.kind == "noul":
        flags = facts.get("checks", [])
        chosen = (
            [flags[index] for index in facts["indices"]] if "indices" in facts else []
        )
        if name == "all-required":
            return not (False in chosen)
        if name == "any-trigger":
            return True in chosen
        if name == "exactly-two":
            return Counter(chosen)[True] == 2
        if name == "threshold-quorum":
            return Counter(chosen)[True] >= facts["quorum"]
        if name == "conditional-obligation":
            return (chosen[0] is False) or (chosen[1] is True)
        if name == "paired-equivalence":
            return chosen[0] is chosen[1]
        if name == "event-order":
            return facts["dates"][1] - facts["dates"][0] > 0
        if name == "window-containment":
            approved, requested = facts["intervals"][0], facts["request_window"]
            return min(approved) <= min(requested) and max(approved) >= max(requested)
        if name == "freshness-cutoff":
            return facts["dates"][0] - facts["cutoff"] >= 0
        if name == "major-minor-version":
            available, required = facts["versions"]
            return (
                tuple(available)[0] == tuple(required)[0]
                and available[1] - required[1] >= 0
            )
        if name == "path-reachability":
            return 4 in _reachable(facts["edges"], 0)
        if name == "cycle-freedom":
            return all(
                start not in _reachable(facts["edges"], to)
                for start, to in facts["edges"]
            )
        if name == "unique-identifiers":
            return len(set(facts["ids"])) == 4
        if name == "budget-sum":
            return facts["budget"] - sum(facts["costs"]) >= 0
        if name == "record-consistency":
            grouped: dict[str, list[int]] = defaultdict(list)
            for item in facts["records"]:
                grouped[item["key"]].append(item["value"])
            return all(len(set(values)) < 2 for values in grouped.values())
        if name == "scope-inclusion":
            return len(set(facts["required_tags"]) - set(facts["authorized_tags"])) == 0
        if name == "distinct-attestors":
            verified = [
                person
                for person, signed in zip(facts["signers"], facts["attested"])
                if signed
            ]
            return len(set(verified)) > 1
        if name == "interval-nonoverlap":
            sequence = sorted(facts["intervals"])
            return all(
                sequence[index][1] <= sequence[index + 1][0] for index in range(3)
            )
        if name == "dependency-completion":
            return not set(facts["required_tags"]).difference(facts["completed_tags"])
        if name == "exception-exemption":
            return bool(facts["exempt"]) or not (False in chosen)
    elif op.kind == "score":
        bound = lambda n: min(4, max(0, n))
        if name == "verified-count":
            return sum(bool(facts["checks"][i]) for i in facts["indices"])
        if name == "weighted-clamped-sum":
            return bound(
                sum(facts["values"][i] * facts["weights"][i] for i in range(4))
            )
        if name == "bottleneck-minimum":
            return sorted(facts["grades"])[0]
        if name == "worst-severity":
            return sorted(facts["grades"])[-1]
        if name == "ratio-quarter-band":
            return bound((facts["successes"] * 4) // facts["trials"])
        if name == "lateness-bucket":
            return bound((max(0, facts["finish_day"] - facts["deadline"]) + 2) // 3)
        if name == "recency-decay":
            age = facts["report_day"] - facts["observed_day"]
            return bound(4 - (age // 5))
        if name == "risk-product-band":
            return bound(
                (facts["impact"] * facts["likelihood"] * facts["risk_scale"] + 7) // 8
            )
        if name == "credit-minus-penalty":
            return bound(
                sum(bool(facts["checks"][i]) for i in facts["indices"])
                - facts["penalties"]
            )
        if name == "dependency-longest-path":
            return bound(_depth(facts["edges"], 0, {0}))
        if name == "distinct-source-count":
            people = [
                name for name, ok in zip(facts["signers"], facts["attested"]) if ok
            ]
            return bound(len(set(people)))
        if name == "coverage-fraction-band":
            return bound(
                (4 * len(set(facts["covered_tags"]) & set(facts["required_tags"])))
                // len(facts["required_tags"])
            )
        if name == "interval-overlap-days":
            left, right = facts["intervals"][:2]
            return bound(max(0, min(left[1], right[1]) - max(left[0], right[0])))
        if name == "forecast-error-band":
            return bound(4 - (abs(facts["actual"] - facts["forecast"]) // 3))
        if name == "inventory-coverage":
            return bound(facts["stock"] // facts["daily_use"])
        if name == "peer-percentile":
            return bound(
                len([n for n in facts["peer_values"] if n < facts["target_value"]])
            )
        if name == "trend-improvement":
            return bound(max(0, facts["history"][1] - facts["history"][0]) // 2)
        if name == "workflow-stage":
            return next(
                index
                for index, stage in enumerate(facts["workflow_order"])
                if stage == facts["current_stage"]
            )
        if name == "budget-variance":
            return bound(4 - (max(0, facts["actual_cost"] - facts["budget"]) // 5))
        if name == "workload-balance":
            return bound(4 - (sorted(facts["loads"])[-1] - sorted(facts["loads"])[0]))
    raise ValueError(f"Unknown rendered operation: {op.id}")
