"""Build the private, DEV-only JevArena authored v10 editorial pilot.

Public code contains the semantic rules and proof contract. Original case text,
fact packs, salt, targets, and source-deletion witnesses remain private.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .authored_v5_dossier import compact, opaque

VERSION = "jevarena-authored-v10-dev12-editorial-pilot/1"
COUNTS = {"choice": 4, "noul": 4, "score": 4}
MECHANISMS = {
    "long_scope": 1,
    "long_consent": 1,
    "long_capacity": 1,
    "missing_source": 3,
    "rule_precedence": 3,
    "ordinary": 3,
}
SOURCE_START = re.compile(
    r"^SOURCE ([0-9a-f]{12}) \| SCOPE ([0-9a-f]{12}) \| FORM ([a-z]+)$"
)
SOURCE_END = re.compile(r"^END SOURCE ([0-9a-f]{12})$")
DATA_LINE = re.compile(r"^DATA ([a-z_]+): (.+)$")
NUMBER_WORD = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|first|second|third)\b",
    re.I,
)
RESULT_CUE = re.compile(
    r"\b(?:is|was|were|has been|have been)\s+"
    r"(?:signed|approved|cleared|certified|verified|revoked|closed|available|vetoed)\b",
    re.I,
)


@dataclass(frozen=True)
class Operation:
    id: str
    kind: str
    fields: tuple[str, ...]
    current: str
    archived: str


OPS = {
    op.id: op
    for op in (
        Operation(
            "floodgate-dispatch",
            "choice",
            ("capacity", "risk", "cleared", "demand"),
            "For the target flood-control file, consider only cleared gates with "
            "capacity at least demand and risk at most three. Choose the lowest "
            "risk, then the greatest capacity, then the alphabetically first gate. "
            "Hold if none qualifies.",
            "Earlier guidance allowed risk four; all other ordering is unchanged.",
        ),
        Operation(
            "art-loan-courier",
            "choice",
            ("transit_hours", "insured", "handling", "fragility"),
            "Consider insured couriers whose handling rating meets fragility. "
            "Choose the largest handling margin above fragility, then the "
            "shortest transit time, then the alphabetically first courier. "
            "Hold if none qualifies.",
            "Earlier guidance chose the shortest transit among insured couriers.",
        ),
        Operation(
            "radio-channel-assignment",
            "choice",
            ("interference", "reserved", "bandwidth", "minimum_bandwidth"),
            "Exclude reserved channels and channels below minimum bandwidth. "
            "Choose the remaining channel with the lowest interference, then "
            "greatest bandwidth, then alphabetically first channel. Hold if none.",
            "Earlier guidance did not exclude reserved channels.",
        ),
        Operation(
            "council-proposal-vote",
            "choice",
            ("votes", "vetoed", "audited"),
            "Among audited proposals with no active veto, choose the highest "
            "vote total, then the alphabetically first proposal. Hold if none.",
            "Earlier guidance counted audited proposals even under veto.",
        ),
        Operation(
            "forest-crossing-clearance",
            "noul",
            (
                "load",
                "rated_capacity",
                "closure_active",
                "emergency_override",
                "inspector_signed",
            ),
            "Clear the crossing when load does not exceed rated capacity and "
            "the closure is inactive. An active closure may instead be overridden "
            "only if the emergency override and inspector signature are both "
            "present and load exceeds rated capacity by no more than two.",
            "Earlier guidance did not accept an emergency override.",
        ),
        Operation(
            "patient-data-consent",
            "noul",
            (
                "consent_scopes",
                "requested_scopes",
                "revoked_scopes",
                "review_signed",
            ),
            "Release is permitted only when every requested scope has consent. "
            "No requested scope may have an active revocation, and a separate "
            "privacy review must be signed. A review does not replace consent.",
            "Earlier guidance did not check active revocations after consent.",
        ),
        Operation(
            "aquifer-alert-corroboration",
            "noul",
            ("readings", "validated_sensors", "threshold"),
            "Issue an alert only if at least two distinct validated sensors "
            "report a reading at or above the threshold.",
            "Earlier guidance required only one validated sensor at threshold.",
        ),
        Operation(
            "polling-ledger-reconciliation",
            "noul",
            ("issued", "cast", "spoiled", "seal_ok"),
            "Certify the ledger only when the seal is valid and cast plus "
            "spoiled ballots exactly equals issued ballots.",
            "Earlier guidance accepted a sealed ledger when cast plus spoiled "
            "did not exceed issued ballots.",
        ),
        Operation(
            "satellite-channel-readiness",
            "score",
            (
                "working_channels",
                "required_channels",
                "tested_channels",
                "ground_signed",
            ),
            "If ground control signed, grade the number of distinct required "
            "channels that are both working and tested, capped at four; otherwise "
            "grade zero.",
            "Earlier guidance counted required working channels without a test.",
        ),
        Operation(
            "hospital-surge-capacity",
            "score",
            ("staffed_beds", "oxygen_beds", "isolation_beds", "request_beds"),
            "Take the smallest of staffed, oxygen-ready and isolation-ready bed "
            "counts. Grade floor(four times that smallest count divided by "
            "requested beds), capped to zero through four.",
            "Earlier guidance used staffed beds alone.",
        ),
        Operation(
            "storm-barrier-stage",
            "score",
            ("surge_day", "ready_day", "drill_delay"),
            "Start at grade four. For each full day that ready_day plus "
            "drill_delay falls after surge_day, subtract one, flooring at zero.",
            "Earlier guidance did not include drill_delay.",
        ),
        Operation(
            "biosample-custody",
            "score",
            ("received", "verified", "unmatched", "review_signed"),
            "If review is unsigned or verified exceeds received, grade zero. "
            "Otherwise grade verified minus unmatched, clamped to zero through four.",
            "Earlier guidance did not subtract unmatched transfers.",
        ),
    )
}
assert len(OPS) == 12


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clamp(value: int) -> int:
    return max(0, min(4, value))


def candidates(facts: dict[str, Any]) -> set[str]:
    for value in facts.values():
        if isinstance(value, dict):
            return set(value)
    return set()


def validate(op: Operation, facts: dict[str, Any]) -> None:
    if set(facts) != set(op.fields):
        raise ValueError("Operation fields differ")
    name = op.id
    keys = candidates(facts)
    if op.kind == "choice":
        if not 3 <= len(keys) <= 5 or not all(
            re.fullmatch(r"[A-Za-z]+", k) for k in keys
        ):
            raise ValueError(
                "Choice candidates must be three to five named alternatives"
            )
        for field, value in facts.items():
            if isinstance(value, dict):
                if set(value) != keys or any(
                    type(v) is not int or v < 0 for v in value.values()
                ):
                    raise ValueError("Candidate metrics disagree")
            elif isinstance(value, list):
                if len(value) != len(set(value)) or not set(value) <= keys:
                    raise ValueError("Candidate subset differs")
            elif type(value) is not int or value < 0:
                raise ValueError("Invalid choice threshold")
    elif name == "forest-crossing-clearance":
        if any(
            type(facts[k]) is not int or facts[k] < 0
            for k in ("load", "rated_capacity")
        ) or any(
            type(facts[k]) is not bool
            for k in ("closure_active", "emergency_override", "inspector_signed")
        ):
            raise ValueError("Invalid crossing facts")
    elif name == "patient-data-consent":
        if (
            any(
                type(facts[k]) is not list or len(facts[k]) != len(set(facts[k]))
                for k in ("consent_scopes", "requested_scopes", "revoked_scopes")
            )
            or type(facts["review_signed"]) is not bool
        ):
            raise ValueError("Invalid consent facts")
    elif name == "aquifer-alert-corroboration":
        if (
            not isinstance(facts["readings"], dict)
            or not 3 <= len(facts["readings"]) <= 6
            or any(type(v) is not int or v < 0 for v in facts["readings"].values())
            or type(facts["validated_sensors"]) is not list
            or not set(facts["validated_sensors"]) <= set(facts["readings"])
            or type(facts["threshold"]) is not int
        ):
            raise ValueError("Invalid sensor facts")
    elif name == "polling-ledger-reconciliation":
        if (
            any(
                type(facts[k]) is not int or facts[k] < 0
                for k in ("issued", "cast", "spoiled")
            )
            or type(facts["seal_ok"]) is not bool
        ):
            raise ValueError("Invalid ledger facts")
    elif name == "satellite-channel-readiness":
        if (
            any(
                type(facts[k]) is not list or len(facts[k]) != len(set(facts[k]))
                for k in ("working_channels", "required_channels", "tested_channels")
            )
            or type(facts["ground_signed"]) is not bool
        ):
            raise ValueError("Invalid satellite facts")
    elif name == "hospital-surge-capacity":
        if (
            any(type(v) is not int or v < 0 for v in facts.values())
            or facts["request_beds"] == 0
        ):
            raise ValueError("Invalid bed facts")
    elif name == "storm-barrier-stage":
        if any(type(v) is not int or v < 0 for v in facts.values()):
            raise ValueError("Invalid barrier facts")
    elif name == "biosample-custody":
        if (
            any(
                type(facts[k]) is not int or facts[k] < 0
                for k in ("received", "verified", "unmatched")
            )
            or type(facts["review_signed"]) is not bool
        ):
            raise ValueError("Invalid custody facts")


def evaluate(op: Operation, facts: dict[str, Any], *, archived: bool = False) -> Any:
    """Direct specification oracle."""
    validate(op, facts)
    n = op.id
    if n == "floodgate-dispatch":
        valid = [
            k
            for k in facts["capacity"]
            if k in facts["cleared"]
            and facts["capacity"][k] >= facts["demand"]
            and facts["risk"][k] <= (4 if archived else 3)
        ]
        return (
            min(valid, key=lambda k: (facts["risk"][k], -facts["capacity"][k], k))
            if valid
            else "hold"
        )
    if n == "art-loan-courier":
        valid = [
            k
            for k in facts["transit_hours"]
            if k in facts["insured"]
            and (archived or facts["handling"][k] >= facts["fragility"])
        ]
        return (
            min(
                valid,
                key=lambda k: (
                    (facts["transit_hours"][k], k)
                    if archived
                    else (
                        -facts["handling"][k] + facts["fragility"],
                        facts["transit_hours"][k],
                        k,
                    )
                ),
            )
            if valid
            else "hold"
        )
    if n == "radio-channel-assignment":
        valid = [
            k
            for k in facts["interference"]
            if (archived or k not in facts["reserved"])
            and facts["bandwidth"][k] >= facts["minimum_bandwidth"]
        ]
        return (
            min(
                valid,
                key=lambda k: (facts["interference"][k], -facts["bandwidth"][k], k),
            )
            if valid
            else "hold"
        )
    if n == "council-proposal-vote":
        valid = [
            k
            for k in facts["votes"]
            if k in facts["audited"] and (archived or k not in facts["vetoed"])
        ]
        return min(valid, key=lambda k: (-facts["votes"][k], k)) if valid else "hold"
    if n == "forest-crossing-clearance":
        return (
            facts["load"] <= facts["rated_capacity"] and not facts["closure_active"]
        ) or (
            not archived
            and facts["closure_active"]
            and facts["emergency_override"]
            and facts["inspector_signed"]
            and facts["load"] <= facts["rated_capacity"] + 2
        )
    if n == "patient-data-consent":
        req = set(facts["requested_scopes"])
        return (
            req <= set(facts["consent_scopes"])
            and (archived or not (req & set(facts["revoked_scopes"])))
            and facts["review_signed"]
        )
    if n == "aquifer-alert-corroboration":
        above = sum(
            facts["readings"][k] >= facts["threshold"]
            for k in facts["validated_sensors"]
        )
        return above >= (1 if archived else 2)
    if n == "polling-ledger-reconciliation":
        total = facts["cast"] + facts["spoiled"]
        return facts["seal_ok"] and (
            total <= facts["issued"] if archived else total == facts["issued"]
        )
    if n == "satellite-channel-readiness":
        matched = set(facts["required_channels"]) & set(facts["working_channels"])
        if not archived:
            matched &= set(facts["tested_channels"])
        return min(4, len(matched)) if facts["ground_signed"] else 0
    if n == "hospital-surge-capacity":
        available = min(
            facts["staffed_beds"], facts["oxygen_beds"], facts["isolation_beds"]
        )
        return clamp(
            (4 * (facts["staffed_beds"] if archived else available))
            // facts["request_beds"]
        )
    if n == "storm-barrier-stage":
        return clamp(
            4
            - max(
                0,
                facts["ready_day"]
                + (0 if archived else facts["drill_delay"])
                - facts["surge_day"],
            )
        )
    if n == "biosample-custody":
        return (
            clamp(facts["verified"] - (0 if archived else facts["unmatched"]))
            if (facts["review_signed"] and facts["verified"] <= facts["received"])
            else 0
        )
    raise ValueError("Unknown operation")


def reference(op: Operation, facts: dict[str, Any], *, archived: bool = False) -> Any:
    """Independent, deliberately separate evaluation route for rendered facts."""
    validate(op, facts)
    n = op.id
    if n in {
        "floodgate-dispatch",
        "art-loan-courier",
        "radio-channel-assignment",
        "council-proposal-vote",
    }:
        options = []
        for name in sorted(candidates(facts)):
            if n == "floodgate-dispatch":
                if (
                    name not in facts["cleared"]
                    or facts["capacity"][name] < facts["demand"]
                    or facts["risk"][name] > (4 if archived else 3)
                ):
                    continue
                score = (facts["risk"][name], -facts["capacity"][name], name)
            elif n == "art-loan-courier":
                if name not in facts["insured"] or (
                    not archived and facts["handling"][name] < facts["fragility"]
                ):
                    continue
                score = (
                    (facts["transit_hours"][name], name)
                    if archived
                    else (
                        facts["fragility"] - facts["handling"][name],
                        facts["transit_hours"][name],
                        name,
                    )
                )
            elif n == "radio-channel-assignment":
                if (not archived and name in facts["reserved"]) or facts["bandwidth"][
                    name
                ] < facts["minimum_bandwidth"]:
                    continue
                score = (facts["interference"][name], -facts["bandwidth"][name], name)
            else:
                if name not in facts["audited"] or (
                    not archived and name in facts["vetoed"]
                ):
                    continue
                score = (-facts["votes"][name], name)
            options.append((score, name))
        return sorted(options)[0][1] if options else "hold"
    if n == "forest-crossing-clearance":
        capacity_ok = facts["load"] <= facts["rated_capacity"]
        ordinary = capacity_ok and not facts["closure_active"]
        emergency = (
            not archived
            and facts["closure_active"]
            and facts["emergency_override"]
            and facts["inspector_signed"]
            and facts["load"] - facts["rated_capacity"] <= 2
        )
        return bool(ordinary or emergency)
    if n == "patient-data-consent":
        consent = all(
            scope in facts["consent_scopes"] for scope in facts["requested_scopes"]
        )
        revocation = any(
            scope in facts["revoked_scopes"] for scope in facts["requested_scopes"]
        )
        return bool(consent and (archived or not revocation) and facts["review_signed"])
    if n == "aquifer-alert-corroboration":
        signals = [
            k
            for k, reading in facts["readings"].items()
            if k in facts["validated_sensors"] and reading >= facts["threshold"]
        ]
        return len(set(signals)) >= (1 if archived else 2)
    if n == "polling-ledger-reconciliation":
        if not facts["seal_ok"]:
            return False
        difference = facts["issued"] - facts["cast"] - facts["spoiled"]
        return difference >= 0 if archived else difference == 0
    if n == "satellite-channel-readiness":
        hits = sum(
            1
            for name in set(facts["required_channels"])
            if name in facts["working_channels"]
            and (archived or name in facts["tested_channels"])
        )
        return min(hits, 4) if facts["ground_signed"] else 0
    if n == "hospital-surge-capacity":
        counts = (
            [facts["staffed_beds"]]
            if archived
            else [facts[k] for k in ("staffed_beds", "oxygen_beds", "isolation_beds")]
        )
        feasible = sorted(counts)[0]
        return min(4, (feasible * 4) // facts["request_beds"])
    if n == "storm-barrier-stage":
        lateness = facts["ready_day"] - facts["surge_day"]
        if not archived:
            lateness += facts["drill_delay"]
        return 4 - min(4, max(0, lateness))
    if not facts["review_signed"] or facts["verified"] > facts["received"]:
        return 0
    remaining = (
        facts["verified"] if archived else facts["verified"] - facts["unmatched"]
    )
    return min(4, max(0, remaining))


def answer(
    op: Operation, worlds: list[dict[str, Any]], *, archived: bool = False
) -> Any:
    values = []
    for facts in worlds:
        left, right = evaluate(op, facts, archived=archived), reference(
            op, facts, archived=archived
        )
        if left != right:
            raise ValueError("Direct and independent oracles differ")
        values.append(left)
    if op.kind == "choice":
        return values[0] if len(set(values)) == 1 else "hold"
    if op.kind == "noul":
        return all(values)
    return min(values)


def source_blocks(state: str) -> list[str]:
    lines = state.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        header = SOURCE_START.fullmatch(lines[index])
        if header is None:
            index += 1
            continue
        start, source_id = index, header[1]
        index += 1
        while index < len(lines) and lines[index] != f"END SOURCE {source_id}":
            index += 1
        if index == len(lines):
            raise ValueError("Unclosed source block")
        blocks.append("\n".join(lines[start : index + 1]))
        index += 1
    return blocks


def parse_sources(state: str, target_scope: str) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    for block in source_blocks(state):
        rows = block.splitlines()
        header = SOURCE_START.fullmatch(rows[0])
        if header is None or SOURCE_END.fullmatch(rows[-1])[1] != header[1]:
            raise ValueError("Malformed source enclosure")
        matches = [m for row in rows[1:-1] for m in [DATA_LINE.fullmatch(row)] if m]
        if len(matches) != 1:
            raise ValueError("Every source must attest exactly one field")
        if header[2] != target_scope:
            continue
        field = matches[0][1]
        if field in facts:
            raise ValueError("Conflicting target field sources")
        facts[field] = json.loads(matches[0][2])
    return facts


def ordered_value(secret: bytes, slug: str, value: Any, names: set[str]) -> Any:
    priority = lambda key: opaque(secret, f"v10:{slug}:candidate:{key}", 64)
    if isinstance(value, dict) and set(value) == names:
        return {key: value[key] for key in sorted(value, key=priority)}
    if isinstance(value, list) and value and set(value) <= names:
        return sorted(value, key=priority)
    return value


def render_source(
    secret: bytes,
    slug: str,
    doc: dict[str, Any],
    scopes: dict[str, str],
    names: set[str],
) -> str:
    if doc["form"] not in {
        "brief",
        "ledger",
        "memo",
        "register",
        "report",
        "notice",
        "worksheet",
        "certificate",
    }:
        raise ValueError("Unsupported document form")
    prose = doc["text"].strip()
    if (
        len(prose.split()) < 48
        or re.search(r"\d", prose)
        or RESULT_CUE.search(prose)
        or re.search(r"(?m)^\s*(?:SOURCE |END SOURCE |DATA )", prose)
    ):
        raise ValueError("Source prose is thin or contains a result/injected data cue")
    ident = opaque(secret, f"v10:{slug}:source:{doc['slug']}", 12)
    scope = scopes[doc.get("scope", "target")]
    value = ordered_value(secret, slug, doc["value"], names)
    return (
        f"SOURCE {ident} | SCOPE {scope} | FORM {doc['form']}\n"
        f"{doc['title']}\n{prose}\n"
        f"DATA {doc['field']}: {json.dumps(value, ensure_ascii=False, separators=(',', ':'))}\n"
        f"END SOURCE {ident}"
    )


def delete_source(state: str, block: str) -> str:
    if state.count(block) != 1:
        raise ValueError("Source deletion must be unique")
    reduced = state.replace("\n\n" + block, "", 1)
    if reduced == state:
        raise ValueError("Source separator missing")
    if any(other not in reduced for other in source_blocks(state) if other != block):
        raise ValueError("Source deletion modified a surviving block")
    return reduced


def worlds_for(
    visible: dict[str, Any], domains: dict[str, list[Any]]
) -> list[dict[str, Any]]:
    if not domains:
        return [dict(visible)]
    fields = sorted(domains)
    return [
        dict(visible, **dict(zip(fields, values)))
        for values in itertools.product(*(domains[field] for field in fields))
    ]


def same_facts(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Candidate and scope lists are sets; displayed order is answer independent."""
    return set(left) == set(right) and all(
        (
            (set(left[field]) == set(right[field]))
            if isinstance(left[field], list) and isinstance(right[field], list)
            else left[field] == right[field]
        )
        for field in left
    )


def build_item(
    spec: dict[str, Any], secret: bytes
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    slug, op = spec["slug"], OPS[spec["operation_id"]]
    mechanism = spec["mechanism"]
    if (
        mechanism not in MECHANISMS
        or len(spec["scene"].split()) < 55
        or re.search(r"\d", spec["scene"])
        or NUMBER_WORD.search(spec["scene"])
        or RESULT_CUE.search(spec["scene"])
    ):
        raise ValueError("Scene is thin or contains result-bearing prose")
    facts = spec["facts"]
    validate(op, facts)
    missing = spec.get("missing_field")
    if (mechanism == "missing_source") != (missing is not None) or (
        missing is not None and missing not in op.fields
    ):
        raise ValueError("Missing-field challenge differs")
    docs = spec["documents"]
    if len({d["slug"] for d in docs}) != len(docs) or len(
        {d["field"] for d in docs if d.get("scope", "target") == "target"}
    ) != len([d for d in docs if d.get("scope", "target") == "target"]):
        raise ValueError("Duplicate source identity or target field")
    names = candidates(facts) if op.kind == "choice" else set()
    scopes = {
        "target": opaque(secret, f"v10:{slug}:scope:target", 12),
        "other": opaque(secret, f"v10:{slug}:scope:other", 12),
    }
    rendered = {d["slug"]: render_source(secret, slug, d, scopes, names) for d in docs}
    intro = spec["scene"].strip()
    procedure = (
        f"Target scope: {scopes['target']}. Use DATA attestations only from SOURCE "
        "blocks with that exact scope. Narrative describes methods, not extra "
        "attestations. Apply the current rule to the target file.\n"
        f"CURRENT RULE: {op.current}"
    )
    if mechanism == "rule_precedence":
        procedure += f"\nSUPERSEDED RULE: {op.archived} Do not apply this version."
    if missing is not None:
        admissible = spec["admissible_values"]
        if len(admissible) < 2 or len(
            {json.dumps(v, sort_keys=True) for v in admissible}
        ) != len(admissible):
            raise ValueError("Missing field lacks distinct admissible values")
        procedure += f"\nUNREPORTED FIELD {missing}; admissible values: {json.dumps(admissible, ensure_ascii=False)}. "
        procedure += "Choice holds if winners differ; Noul certifies only if every completion passes; Score reports the lowest grade."
    state = (
        intro
        + "\n\n"
        + procedure
        + "\n\n"
        + "\n\n".join(rendered[d["slug"]] for d in docs)
    )
    visible = parse_sources(state, scopes["target"])
    expected = {field: value for field, value in facts.items() if field != missing}
    if not same_facts(visible, expected):
        raise ValueError("Rendered facts differ from private source specification")
    if mechanism.startswith("long_"):
        if not 650 <= len(state.split()) <= 1200 or len(spec["essential"]) < 4:
            raise ValueError("Long case lacks useful length or essential sources")
    domains = {missing: spec["admissible_values"]} if missing is not None else {}
    original_worlds = worlds_for(visible, domains)
    current_answer = answer(op, original_worlds)
    archived_answer = answer(op, original_worlds, archived=True)
    if mechanism == "rule_precedence" and current_answer == archived_answer:
        raise ValueError("Rule-precedence case does not change the answer")
    essential = spec["essential"]
    if set(essential) - set(rendered) or any(
        field not in op.fields or field == missing for field in essential.values()
    ):
        raise ValueError("Invalid essential-source declaration")
    ablations = []
    sensitivity = {}
    for doc_slug, field in essential.items():
        block = rendered[doc_slug]
        header = SOURCE_START.fullmatch(block.splitlines()[0])
        if header is None or header[2] != scopes["target"]:
            raise ValueError("Essential source must be target scoped")
        reduced = delete_source(state, block)
        remaining = parse_sources(reduced, scopes["target"])
        if field in remaining or not same_facts(
            remaining, {k: v for k, v in visible.items() if k != field}
        ):
            raise ValueError("Source deletion failed to remove exactly one fact")
        possible = spec["domains"][field]
        if len(possible) < 2 or not any(v == facts[field] for v in possible):
            raise ValueError("Domain witnesses must include the original value")
        missing_domains = {**domains, field: possible}
        completions = worlds_for(remaining, missing_domains)
        outcomes = [answer(op, [world]) for world in completions]
        if len({json.dumps(v, sort_keys=True) for v in outcomes}) < 2:
            raise ValueError("Deleted source leaves a single provable outcome")
        sensitivity[doc_slug] = {
            "completions": len(completions),
            "distinct_outputs": len(set(map(str, outcomes))),
        }
        ablations.append(
            {
                "parent_id": opaque(secret, f"v10:{slug}:item", 16),
                "omitted_source": header[1],
                "state": reduced,
                "review_instruction": "Without guessing the omitted source, can the original decision still be proven from the remaining case file? Describe any residual clue.",
            }
        )
    if op.kind == "choice":
        labels = sorted(
            [*names, "hold"],
            key=lambda k: opaque(secret, f"v10:{slug}:candidate:{k}", 64),
        )
        criteria: Any = {key: f"Select {key}" for key in labels}
    elif op.kind == "noul":
        labels = sorted(
            ("true", "false"),
            key=lambda k: opaque(secret, f"v10:{slug}:boolean:{k}", 64),
        )
        criteria = {
            key: ("Certified" if key == "true" else "Not certified") for key in labels
        }
    else:
        criteria = [f"Grade {level}" for level in range(5)]
    item_id = opaque(secret, f"v10:{slug}:item", 16)
    question = {
        "decision": {
            "type": op.kind,
            "instructions": "Apply the current rule to the target scope.",
            "criteria": criteria,
        }
    }
    prompt = {"id": item_id, "state": state, "questions": question}
    target = {
        "id": item_id,
        "kind": op.kind,
        "answer": {op.kind: current_answer},
        "source_group": opaque(secret, f"v10:{slug}:group", 16),
    }
    proof = {
        "id": item_id,
        "slug": slug,
        "operation_id": op.id,
        "mechanism": mechanism,
        "visible_facts": visible,
        "answer": current_answer,
        "archived_answer": archived_answer,
        "essential_sources": essential,
        "sensitivity": sensitivity,
        "visible_words": len(state.split()),
    }
    for row in ablations:
        row["questions"] = question
    return prompt, target, proof, ablations


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(compact(row) for row in rows))
    path.chmod(0o600)


def build(spec_path: Path, salt_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("A frozen candidate cannot be overwritten")
    specs = json.loads(spec_path.read_text())
    secret = salt_path.read_bytes()
    if (
        len(secret) != 32
        or len(specs) != 12
        or len({row["slug"] for row in specs}) != 12
        or len({row["operation_id"] for row in specs}) != 12
    ):
        raise ValueError(
            "V10 requires twelve distinct private cases and one 256-bit salt"
        )
    prompts, targets, proofs, ablations = [], [], [], []
    for spec in specs:
        prompt, target, proof, rows = build_item(spec, secret)
        prompts.append(prompt)
        targets.append(target)
        proofs.append(proof)
        ablations.extend(rows)
    if (
        Counter(t["kind"] for t in targets) != COUNTS
        or Counter(p["mechanism"] for p in proofs) != MECHANISMS
    ):
        raise ValueError("Type or mechanism quotas differ")
    bool_answers = Counter(t["answer"]["noul"] for t in targets if t["kind"] == "noul")
    scores = {t["answer"]["score"] for t in targets if t["kind"] == "score"}
    choice_answers = {t["answer"]["choice"] for t in targets if t["kind"] == "choice"}
    positions = Counter()
    for prompt, target in zip(prompts, targets):
        if target["kind"] == "choice":
            labels = list(prompt["questions"]["decision"]["criteria"])
            positions[labels.index(target["answer"]["choice"]) + 1] += 1
    balanced = (
        bool_answers == {True: 2, False: 2}
        and len(scores) >= 3
        and "hold" in choice_answers
        and len(choice_answers) >= 3
        and len(positions) == 4
    )
    if not balanced:
        raise ValueError("Preregistered output or displayed-position balance failed")
    order = sorted(
        range(12), key=lambda i: opaque(secret, f"v10:row:{prompts[i]['id']}", 64)
    )
    output.mkdir(parents=True)
    private = output / "private"
    private.mkdir(mode=0o700)
    write_jsonl(output / "prompts.jsonl", [prompts[i] for i in order])
    write_jsonl(output / "ablations.gold-free.jsonl", ablations)
    write_jsonl(private / "targets.jsonl", [targets[i] for i in order])
    write_jsonl(private / "proof_traces.jsonl", [proofs[i] for i in order])
    receipt = {
        "version": VERSION,
        "status": "AUTOMATED_PROOF_ONLY",
        "release_qualified": False,
        "human_review_passed": False,
        "accepted": 12,
        "type_counts": dict(COUNTS),
        "mechanism_counts": dict(MECHANISMS),
        "boolean_labels": {str(k): v for k, v in bool_answers.items()},
        "score_levels": sorted(scores),
        "choice_positions_one_based": dict(positions),
        "ablation_rows": len(ablations),
        "long_lengths": sorted(
            p["visible_words"] for p in proofs if p["mechanism"].startswith("long_")
        ),
        "spec_sha256": digest(spec_path.read_bytes()),
        "salt_commitment_sha256": digest(secret),
        "builder_sha256": digest(Path(__file__).read_bytes()),
        "prompts_sha256": digest((output / "prompts.jsonl").read_bytes()),
        "ablations_sha256": digest((output / "ablations.gold-free.jsonl").read_bytes()),
        "targets_sha256": digest((private / "targets.jsonl").read_bytes()),
        "proof_sha256": digest((private / "proof_traces.jsonl").read_bytes()),
    }
    (private / "audit.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    )
    (private / "audit.json").chmod(0o600)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--private-salt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.specs, args.private_salt, args.output_dir), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
