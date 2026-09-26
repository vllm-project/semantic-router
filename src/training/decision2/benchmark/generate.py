"""Generate deterministic, oracle-checked Choice/Noul/Score decisions.

The private JSONL contains gold labels. Only the separate prompt JSONL may be
sent to a model. Final-set entropy is read from a private file and never emitted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import string
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "typed-decision-bench/1"
SUITE_VERSION = "0.1.0"
QUESTION_ID = "decision"
DEV_FAMILIES = (
    "attribute_gate",
    "rule_precedence",
    "set_reconciliation",
    "transition_table",
)
FINAL_FAMILIES = (
    "constraint_competition",
    "exception_stack",
    "evidence_join",
    "resource_ledger",
)
FAMILIES = {"dev": DEV_FAMILIES, "final": FINAL_FAMILIES}


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    raw = value if isinstance(value, bytes) else encoded(value).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def codes(rng: random.Random, count: int, banned: set[str] | None = None) -> list[str]:
    used = set() if banned is None else set(banned)
    alphabet = string.ascii_uppercase + string.digits
    result = []
    while len(result) < count:
        candidate = "X" + "".join(rng.choices(alphabet, k=5))
        if candidate not in used:
            result.append(candidate)
            used.add(candidate)
    return result


def choice_question(
    instructions: str, labels: list[str], detail: str
) -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": dict.fromkeys(labels, detail),
    }


@dataclass
class Scenario:
    base: dict[str, Any]
    counterfactual: dict[str, Any]
    questions: dict[str, dict[str, Any]]
    symbols: list[str]
    unordered_fields: tuple[str, ...]
    semantic_maps: dict[str, dict[str, str]]
    edit: str


def attribute_gate(rng: random.Random) -> Scenario:
    rec0, rec1, rec2 = codes(rng, 3)
    records = [
        {"id": rec0, "active": True, "training_complete": True, "hold": False},
        {"id": rec1, "active": False, "training_complete": True, "hold": False},
        {"id": rec2, "active": True, "training_complete": False, "hold": True},
    ]
    rng.shuffle(records)
    labels = [rec0, rec1, rec2, "none"]
    rng.shuffle(labels)
    base = {"records": records}
    cf = copy.deepcopy(base)
    next(row for row in cf["records"] if row["id"] == rec0)["training_complete"] = False
    questions = {
        QUESTION_ID: choice_question(
            "Select the sole record that is active, training complete, and not on hold. "
            "If no record qualifies, select none. Record order has no priority.",
            labels,
            "Select this exact option when it meets the stated rule.",
        )
    }
    questions[QUESTION_ID]["criteria"]["none"] = "No record meets all conditions."
    return Scenario(
        base,
        cf,
        questions,
        [rec0, rec1, rec2],
        ("records",),
        {
            QUESTION_ID: {
                rec0: "record_0",
                rec1: "record_1",
                rec2: "record_2",
                "none": "none",
            }
        },
        "training_complete for the otherwise qualifying record changes from true to false",
    )


def rule_precedence(rng: random.Random) -> Scenario:
    field0, field1, rule0, rule1, rule2 = codes(rng, 5)
    winner_allow = rng.choice([True, False])
    lower = rng.randint(2, 8)
    higher = lower + 1
    rules = [
        {
            "id": rule0,
            "priority": higher,
            "when": {"field": field0, "equals": True},
            "effect": "allow" if winner_allow else "deny",
        },
        {
            "id": rule1,
            "priority": lower,
            "when": {"field": field0, "equals": True},
            "effect": "deny" if winner_allow else "allow",
        },
        {
            "id": rule2,
            "priority": higher + 10,
            "when": {"field": field1, "equals": True},
            "effect": "deny",
        },
    ]
    rng.shuffle(rules)
    base = {"facts": {field0: True, field1: False}, "rules": rules}
    cf = copy.deepcopy(base)
    next(rule for rule in cf["rules"] if rule["id"] == rule1)["priority"] = higher + 1
    questions = {
        QUESTION_ID: {
            "type": "noul",
            "instructions": "Does the highest-priority matching rule allow the action? A rule matches when the named fact equals its condition. Higher numeric priority wins; unmatched rules do not apply.",
            "criteria": {
                "true": "The winning matching rule allows.",
                "false": "The winning matching rule denies, or no rule matches.",
            },
        }
    }
    if rng.choice([True, False]):
        base, cf = cf, base
    return Scenario(
        base,
        cf,
        questions,
        [field0, field1, rule0, rule1, rule2],
        ("rules",),
        {},
        "one conflicting matching rule increases from below to above the other priority",
    )


def set_reconciliation(rng: random.Random) -> Scenario:
    a, b, c, unrelated = codes(rng, 4)
    requested = [a, b, c]
    rng.shuffle(requested)
    mode = rng.choice(["unverified", "blocked"])
    if mode == "unverified":
        base = {
            "requested": requested,
            "verified": [a, b, unrelated],
            "blocked": [unrelated],
        }
        cf = copy.deepcopy(base)
        cf["verified"].append(c)
        edit = "one requested identifier gains verification"
    else:
        base = {
            "requested": requested,
            "verified": [a, b, c, unrelated],
            "blocked": [c],
        }
        cf = copy.deepcopy(base)
        cf["blocked"].remove(c)
        edit = "one requested identifier is removed from the blocked set"
    questions = {
        QUESTION_ID: {
            "type": "score",
            "instructions": "Rate the requested identifiers only. A blocked requested identifier takes precedence over verification. Ignore identifiers not requested.",
            "criteria": [
                "At least one requested identifier is blocked.",
                "None is blocked, but at least one requested identifier is not verified.",
                "All requested identifiers are verified and none is blocked.",
            ],
        }
    }
    if rng.choice([True, False]):
        base, cf = cf, base
    return Scenario(
        base,
        cf,
        questions,
        [a, b, c, unrelated],
        ("requested", "verified", "blocked"),
        {},
        edit,
    )


def transition_table(rng: random.Random) -> Scenario:
    s0, s1, s2, event0, event1, row0, row1, row2 = codes(rng, 8)
    rows = [
        {"id": row0, "from": s0, "event": event0, "guard": False, "to": s1},
        {"id": row1, "from": s0, "event": event0, "guard": True, "to": s2},
        {"id": row2, "from": s1, "event": event1, "guard": True, "to": s0},
    ]
    rng.shuffle(rows)
    base = {"current": s0, "event": event0, "guard": False, "transitions": rows}
    cf = copy.deepcopy(base)
    cf["guard"] = True
    labels = [s0, s1, s2]
    rng.shuffle(labels)
    questions = {
        QUESTION_ID: choice_question(
            "Apply the unique transition matching current state, event, and guard exactly. "
            "If no row matches, remain in the current state. Transition row order has no priority.",
            labels,
            "This is the resulting state identifier.",
        )
    }
    if rng.choice([True, False]):
        base, cf = cf, base
    return Scenario(
        base,
        cf,
        questions,
        [s0, s1, s2, event0, event1, row0, row1, row2],
        ("transitions",),
        {QUESTION_ID: {s0: "state_0", s1: "state_1", s2: "state_2"}},
        "guard truth value changes",
    )


def constraint_competition(rng: random.Random) -> Scenario:
    best, cheap, noncompliant, slow = codes(rng, 4)
    capacity = rng.randint(4, 12)
    maximum_latency = rng.randint(4, 10)
    best_latency = rng.randint(2, maximum_latency)
    cheap_latency = rng.randint(1, best_latency - 1)
    best_cost = rng.randint(5, 15)
    cheap_cost = (
        best_cost if rng.choice([True, False]) else rng.randint(1, best_cost - 1)
    )
    offers = [
        {
            "id": best,
            "capacity": capacity,
            "latency": best_latency,
            "compliant": True,
            "cost": best_cost,
        },
        {
            "id": cheap,
            "capacity": capacity - 1,
            "latency": cheap_latency,
            "compliant": True,
            "cost": cheap_cost,
        },
        {
            "id": noncompliant,
            "capacity": capacity + 2,
            "latency": 1,
            "compliant": False,
            "cost": 1,
        },
        {
            "id": slow,
            "capacity": capacity + 2,
            "latency": maximum_latency + 1,
            "compliant": True,
            "cost": 1,
        },
    ]
    rng.shuffle(offers)
    base = {
        "requirements": {
            "minimum_capacity": capacity,
            "maximum_latency": maximum_latency,
            "compliant": True,
        },
        "offers": offers,
    }
    cf = copy.deepcopy(base)
    next(offer for offer in cf["offers"] if offer["id"] == cheap)["capacity"] = capacity
    labels = [best, cheap, noncompliant, slow, "none"]
    rng.shuffle(labels)
    questions = {
        QUESTION_ID: choice_question(
            "Select the compliant offer meeting both capacity and latency limits with the lowest cost. "
            "Break a cost tie by lower latency. If none qualifies, select none. Offer order has no priority.",
            labels,
            "Select this offer only if it wins after all constraints and tie breaks.",
        )
    }
    questions[QUESTION_ID]["criteria"][
        "none"
    ] = "No offer satisfies all hard constraints."
    if rng.choice([True, False]):
        base, cf = cf, base
    return Scenario(
        base,
        cf,
        questions,
        [best, cheap, noncompliant, slow],
        ("offers",),
        {
            QUESTION_ID: {
                best: "offer_0",
                cheap: "offer_1",
                noncompliant: "offer_2",
                slow: "offer_3",
                "none": "none",
            }
        },
        "the competing offer gains one capacity unit and crosses the minimum",
    )


def exception_stack(rng: random.Random) -> Scenario:
    flag_a, flag_b, allow_rule, deny_rule, low_rule, other_rule = codes(rng, 6)
    base_allow = rng.choice([True, False])
    dominant = "allow" if base_allow else "deny"
    opposite = "deny" if base_allow else "allow"
    clauses = [
        {
            "id": allow_rule,
            "priority": 5,
            "all": [{"fact": flag_a, "is": True}],
            "effect": dominant,
        },
        {
            "id": deny_rule,
            "priority": 6,
            "all": [{"fact": flag_b, "is": True}, {"fact": flag_a, "is": True}],
            "effect": opposite,
        },
        {
            "id": low_rule,
            "priority": 2,
            "all": [{"fact": flag_b, "is": False}],
            "effect": opposite,
        },
        {
            "id": other_rule,
            "priority": 9,
            "all": [{"fact": flag_a, "is": False}],
            "effect": opposite,
        },
    ]
    rng.shuffle(clauses)
    base = {"facts": {flag_a: True, flag_b: False}, "clauses": clauses}
    cf = copy.deepcopy(base)
    cf["facts"][flag_b] = True
    questions = {
        QUESTION_ID: {
            "type": "noul",
            "instructions": "Does the highest-priority matching clause allow the action? Every condition in 'all' must match its named fact, including false conditions. Higher priority wins; clause order has no priority.",
            "criteria": {
                "true": "The winning clause allows.",
                "false": "The winning clause denies, or none matches.",
            },
        }
    }
    if rng.choice([True, False]):
        base, cf = cf, base
    return Scenario(
        base,
        cf,
        questions,
        [flag_a, flag_b, allow_rule, deny_rule, low_rule, other_rule],
        ("clauses",),
        {},
        "one Boolean fact is negated, activating the higher-priority exception",
    )


def evidence_join(rng: random.Random) -> Scenario:
    entity, other_entity, item, other_item = codes(rng, 4)
    verdict = rng.choice(["pass", "fail"])
    missing = rng.choice(["attestation", "registry"])
    state = {
        "target": {"entity": entity, "item": item},
        "registry": [
            {"entity": entity, "active": missing != "registry"},
            {"entity": other_entity, "active": False},
        ],
        "attestations": [
            {
                "entity": entity,
                "item": other_item,
                "verdict": "fail" if verdict == "pass" else "pass",
            },
            {
                "entity": other_entity,
                "item": item,
                "verdict": "pass" if verdict == "pass" else "fail",
            },
        ],
    }
    cf = copy.deepcopy(state)
    if missing == "attestation":
        cf["attestations"].append({"entity": entity, "item": item, "verdict": verdict})
        edit = "a target attestation matching both join keys is added or removed"
    else:
        state["attestations"].append(
            {"entity": entity, "item": item, "verdict": verdict}
        )
        cf["attestations"].append({"entity": entity, "item": item, "verdict": verdict})
        cf["registry"][0]["active"] = True
        edit = "the target registry active flag is negated"
    questions = {
        QUESTION_ID: {
            "type": "choice",
            "instructions": "For the target entity-item pair, use an active registry entry and an attestation matching both keys. "
            "Choose insufficient_evidence if either is missing; otherwise use the matching attestation verdict. "
            "Other entities and items do not supply evidence.",
            "criteria": {
                "approved": "A matching attestation says pass and the entity has an active registry entry.",
                "rejected": "A matching attestation says fail and the entity has an active registry entry.",
                "insufficient_evidence": "No active registry entry or no attestation matching both target keys.",
            },
        },
        "determinate": {
            "type": "noul",
            "instructions": "Is there enough matching evidence to determine approved or rejected for the target pair? Require an active registry entry and an attestation matching both keys.",
            "criteria": {
                "true": "Both required matching records exist.",
                "false": "At least one required matching record is absent.",
            },
        },
    }
    if rng.choice([True, False]):
        state, cf = cf, state
    return Scenario(
        state,
        cf,
        questions,
        [entity, other_entity, item, other_item],
        ("registry", "attestations"),
        {
            QUESTION_ID: {
                "approved": "approved",
                "rejected": "rejected",
                "insufficient_evidence": "insufficient_evidence",
            }
        },
        edit,
    )


def replay_ledger(state: dict[str, Any]) -> int:
    amount = state["initial"]
    for event in sorted(state["events"], key=lambda item: item["tick"]):
        if event["posted"]:
            amount += event["units"] if event["kind"] == "add" else -event["units"]
            amount = max(0, min(state["capacity"], amount))
    return amount


def resource_ledger(rng: random.Random) -> Scenario:
    event_ids = codes(rng, 3)
    while True:
        initial = rng.randint(0, 4)
        events = [
            {
                "id": event_ids[0],
                "tick": 10,
                "kind": "add",
                "units": rng.randint(1, 3),
                "posted": True,
            },
            {
                "id": event_ids[1],
                "tick": 20,
                "kind": "remove",
                "units": rng.randint(1, 3),
                "posted": True,
            },
            {
                "id": event_ids[2],
                "tick": 30,
                "kind": rng.choice(["add", "remove"]),
                "units": rng.randint(1, 2),
                "posted": False,
            },
        ]
        rng.shuffle(events)
        base = {"initial": initial, "capacity": 4, "events": events}
        cf = copy.deepcopy(base)
        next(event for event in cf["events"] if event["id"] == event_ids[1])[
            "posted"
        ] = False
        if replay_ledger(base) != replay_ledger(cf):
            break
    questions = {
        QUESTION_ID: {
            "type": "score",
            "instructions": "Replay posted events by increasing tick from the initial amount. Add or remove units, clamping to zero and capacity after every event. Ignore unposted events and the input order of event rows. Rate the final amount.",
            "criteria": [
                f"Final amount is {n} unit{'s' if n != 1 else ''}." for n in range(5)
            ],
        }
    }
    if rng.choice([True, False]):
        base, cf = cf, base
    return Scenario(
        base,
        cf,
        questions,
        event_ids,
        ("events",),
        {},
        "the posted flag of one removal event is negated",
    )


BUILDERS: dict[str, Callable[[random.Random], Scenario]] = {
    "attribute_gate": attribute_gate,
    "rule_precedence": rule_precedence,
    "set_reconciliation": set_reconciliation,
    "transition_table": transition_table,
    "constraint_competition": constraint_competition,
    "exception_stack": exception_stack,
    "evidence_join": evidence_join,
    "resource_ledger": resource_ledger,
}


def oracle(family: str, state: dict[str, Any]) -> dict[str, Any]:
    if family == "attribute_gate":
        eligible = [
            r["id"]
            for r in state["records"]
            if r["active"] and r["training_complete"] and not r["hold"]
        ]
        return {QUESTION_ID: eligible[0] if len(eligible) == 1 else "none"}
    if family == "rule_precedence":
        matched = [
            r
            for r in state["rules"]
            if state["facts"][r["when"]["field"]] == r["when"]["equals"]
        ]
        top = max(matched, key=lambda r: r["priority"]) if matched else None
        return {QUESTION_ID: top is not None and top["effect"] == "allow"}
    if family == "set_reconciliation":
        requested = set(state["requested"])
        level = (
            0
            if requested & set(state["blocked"])
            else (2 if requested <= set(state["verified"]) else 1)
        )
        return {QUESTION_ID: level}
    if family == "transition_table":
        matched = [
            r
            for r in state["transitions"]
            if r["from"] == state["current"]
            and r["event"] == state["event"]
            and r["guard"] == state["guard"]
        ]
        if len(matched) > 1:
            raise ValueError("ambiguous transition")
        return {QUESTION_ID: matched[0]["to"] if matched else state["current"]}
    if family == "constraint_competition":
        req = state["requirements"]
        eligible = [
            o
            for o in state["offers"]
            if o["capacity"] >= req["minimum_capacity"]
            and o["latency"] <= req["maximum_latency"]
            and o["compliant"] == req["compliant"]
        ]
        winner = (
            min(eligible, key=lambda o: (o["cost"], o["latency"])) if eligible else None
        )
        return {QUESTION_ID: winner["id"] if winner else "none"}
    if family == "exception_stack":
        matched = [
            c
            for c in state["clauses"]
            if all(state["facts"][p["fact"]] == p["is"] for p in c["all"])
        ]
        top = max(matched, key=lambda c: c["priority"]) if matched else None
        return {QUESTION_ID: top is not None and top["effect"] == "allow"}
    if family == "evidence_join":
        target = state["target"]
        active = any(
            r["entity"] == target["entity"] and r["active"] for r in state["registry"]
        )
        matching = [
            a
            for a in state["attestations"]
            if a["entity"] == target["entity"] and a["item"] == target["item"]
        ]
        if len(matching) > 1:
            raise ValueError("multiple target attestations")
        known = active and len(matching) == 1
        decision = (
            "insufficient_evidence"
            if not known
            else ("approved" if matching[0]["verdict"] == "pass" else "rejected")
        )
        return {QUESTION_ID: decision, "determinate": known}
    if family == "resource_ledger":
        return {QUESTION_ID: replay_ledger(state)}
    raise ValueError(f"unknown family: {family}")


def rename(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            mapping.get(key, key): rename(item, mapping) for key, item in value.items()
        }
    if isinstance(value, list):
        return [rename(item, mapping) for item in value]
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


def reorder_surface(
    state: dict[str, Any], questions: dict[str, Any], fields: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = copy.deepcopy(state)
    questions = copy.deepcopy(questions)
    for key in fields:
        if isinstance(state[key], list):
            state[key].reverse()
        elif isinstance(state[key], dict):
            state[key] = dict(reversed(list(state[key].items())))
    for question in questions.values():
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            question["criteria"] = dict(reversed(list(criteria.items())))
    return state, questions


def make_gold(
    family: str,
    state: dict[str, Any],
    questions: dict[str, Any],
    semantic_maps: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    values = oracle(family, state)
    if values.keys() != questions.keys():
        raise ValueError(f"oracle/question mismatch in {family}")
    gold = {}
    for key, question in questions.items():
        qtype = question["type"]
        value = values[key]
        answer = {"type": qtype, "value": value}
        if qtype == "choice":
            mapping = semantic_maps[key]
            if value not in question["criteria"] or set(mapping) != set(
                question["criteria"]
            ):
                raise ValueError(f"invalid choice oracle in {family}")
            answer["semantic_value"] = mapping[value]
            answer["label_to_semantic"] = mapping
        else:
            answer["semantic_value"] = value
        if qtype == "score" and not 0 <= value < len(question["criteria"]):
            raise ValueError(f"invalid score oracle in {family}")
        gold[key] = answer
    return gold


def derived_rng(master: bytes, family: str, index: int) -> random.Random:
    material = (
        master + b"\0" + family.encode("ascii") + b"\0" + str(index).encode("ascii")
    )
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def generate(split: str, seed: bytes, groups_per_family: int) -> list[dict[str, Any]]:
    if split not in FAMILIES or groups_per_family < 1:
        raise ValueError(
            "split must be dev or final; groups_per_family must be positive"
        )
    code_digest = digest(Path(__file__).read_bytes())
    commitment = digest(seed)
    items = []
    for family in FAMILIES[split]:
        for instance in range(groups_per_family):
            rng = derived_rng(seed, family, instance)
            scenario = BUILDERS[family](rng)
            shuffled_state, shuffled_questions = reorder_surface(
                scenario.base, scenario.questions, scenario.unordered_fields
            )
            replacement = dict(
                zip(
                    scenario.symbols,
                    codes(rng, len(scenario.symbols), set(scenario.symbols)),
                )
            )
            renamed_map = {
                key: {
                    replacement.get(label, label): semantic
                    for label, semantic in mapping.items()
                }
                for key, mapping in scenario.semantic_maps.items()
            }
            variants = [
                ("base", scenario.base, scenario.questions, scenario.semantic_maps),
                (
                    "counterfactual",
                    scenario.counterfactual,
                    scenario.questions,
                    scenario.semantic_maps,
                ),
                ("order", shuffled_state, shuffled_questions, scenario.semantic_maps),
                (
                    "label",
                    rename(scenario.base, replacement),
                    rename(scenario.questions, replacement),
                    renamed_map,
                ),
            ]
            group_id = "g_" + digest([split, family, instance, commitment])[:20]
            rows = []
            for variant, state, questions, semantic_maps in variants:
                gold = make_gold(family, state, questions, semantic_maps)
                payload = {"state": state, "questions": questions}
                item_id = "td_" + digest([group_id, variant, payload])[:24]
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "id": item_id,
                        "split": split,
                        "family": family,
                        "group_id": group_id,
                        "pairs": [],
                        "state": state,
                        "questions": questions,
                        "gold": gold,
                        "provenance": {
                            "source": "programmatic_synthetic",
                            "suite_version": SUITE_VERSION,
                            "generator_family": family,
                            "generator_version": 1,
                            "python_version": platform.python_version(),
                            "rng": "python_random_MT19937",
                            "instance_index": instance,
                            "variant": variant,
                            "oracle": f"benchmark.generate.oracle:{family}",
                            "counterfactual_edit": scenario.edit,
                            "seed_commitment_sha256": commitment,
                            "generator_code_sha256": code_digest,
                            "payload_sha256": digest(payload),
                        },
                    }
                )
            base_semantic = {
                key: answer["semantic_value"] for key, answer in rows[0]["gold"].items()
            }
            cf_semantic = {
                key: answer["semantic_value"] for key, answer in rows[1]["gold"].items()
            }
            if base_semantic == cf_semantic:
                raise ValueError(f"counterfactual failed to change answer in {family}")
            for target in (2, 3):
                if {
                    key: answer["semantic_value"]
                    for key, answer in rows[target]["gold"].items()
                } != base_semantic:
                    raise ValueError(f"invariance violation in {family}")
            for index, relation in (
                (1, "counterfactual"),
                (2, "order_invariance"),
                (3, "label_invariance"),
            ):
                pair_id = "p_" + digest([group_id, relation])[:20]
                rows[0]["pairs"].append(
                    {"id": pair_id, "relation": relation, "role": "anchor"}
                )
                rows[index]["pairs"].append(
                    {"id": pair_id, "relation": relation, "role": "variant"}
                )
            items.extend(rows)
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("duplicate item IDs")
    return items


def prompt_record(item: dict[str, Any]) -> dict[str, Any]:
    return {"id": item["id"], "state": item["state"], "questions": item["questions"]}


def atomic_jsonl(
    path: Path, rows: list[dict[str, Any]], overwrite: bool, private: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            for row in rows:
                output.write(encoded(row) + "\n")
        if private:
            os.chmod(temp_name, 0o600)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=FAMILIES, required=True)
    parser.add_argument("--groups-per-family", type=int, default=20)
    parser.add_argument("--seed", help="Development seed; forbidden for final")
    parser.add_argument(
        "--seed-file", type=Path, help="Private 32+ byte entropy file for final"
    )
    parser.add_argument("--output", type=Path, required=True, help="Private gold JSONL")
    parser.add_argument(
        "--prompts-output", type=Path, required=True, help="Gold-free model input JSONL"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.resolve() == args.prompts_output.resolve():
        parser.error("private output and prompt output must be distinct paths")
    if args.split == "final":
        if args.seed is not None or args.seed_file is None:
            parser.error("final requires --seed-file and forbids --seed")
        seed = args.seed_file.read_bytes()
        if len(seed) < 32:
            parser.error(
                "final seed file must contain at least 32 bytes of private entropy"
            )
    else:
        if args.seed_file is not None:
            parser.error("dev uses --seed, not --seed-file")
        seed = (args.seed if args.seed is not None else "public-demo-v1").encode(
            "utf-8"
        )
    items = generate(args.split, seed, args.groups_per_family)
    for path in (args.output, args.prompts_output):
        if path.exists() and not args.overwrite:
            parser.error(f"{path} already exists; pass --overwrite to replace it")
    atomic_jsonl(args.output, items, args.overwrite, private=True)
    atomic_jsonl(
        args.prompts_output,
        [prompt_record(item) for item in items],
        args.overwrite,
        private=False,
    )
    print(
        encoded(
            {
                "split": args.split,
                "items": len(items),
                "groups": len(items) // 4,
                "families": list(FAMILIES[args.split]),
                "seed_commitment_sha256": digest(seed),
                "private_output": str(args.output),
                "prompts_output": str(args.prompts_output),
            }
        )
    )


if __name__ == "__main__":
    main()
