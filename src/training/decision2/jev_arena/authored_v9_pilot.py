"""Build a private, gold-separated authored v9 DEV editorial pilot.

The scenarios and salt are private input. This source defines visible evidence
rendering, exact mechanical semantics, and ablation proof; it cannot certify
editorial quality or promote a release benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .authored_v5_dossier import compact, opaque
from .authored_v7_policies import Policy
from .authored_v8_pilot import POLICIES as V8_POLICIES
from .authored_v8_pilot import evaluate as v8_evaluate
from .authored_v8_pilot import reference as v8_reference
from .authored_v8_pilot import validate as v8_validate

VERSION = "jevarena-authored-v9-dev12-editorial-pilot/1"
KIND_COUNTS = {"choice": 4, "noul": 4, "score": 4}
MECHANISM_COUNTS = {
    "long_amendment": 1,
    "long_alias": 1,
    "long_timeline": 1,
    "missing_source": 3,
    "rule_precedence": 3,
    "ordinary": 3,
}
NEW_POLICIES = {
    row.id: row
    for row in (
        Policy(
            "substation-restart",
            "noul",
            ("relay_tests", "isolated_faults", "operator_signed"),
            "Authorize restart only when at least three relay tests pass, at least two fault segments are isolated, and the licensed operator signs. Authorize true across admissible worlds only if every world passes.",
            "Authorize restart with at least two relay tests and one isolated segment after any crew signature.",
        ),
        Policy(
            "harbor-entry",
            "noul",
            ("trained_crew", "residual_risk", "permit_current"),
            "Permit harbor entry only with at least two trained crew, audited residual risk no greater than three, and a current permit. Permit true across admissible worlds only if every world passes.",
            "Permit harbor entry only with at least three trained crew, residual risk no greater than two, and a current permit.",
        ),
        Policy(
            "port-berth-arrival",
            "choice",
            ("arrival_minutes", "draft_meters", "licensed"),
            "Among licensed vessels with draft no greater than eight meters, assign the berth to the earliest arrival; break ties by vessel ID. Hold if none qualifies or admissible winners differ.",
            "Among licensed vessels, assign the berth to the earliest arrival regardless of draft; break ties by vessel ID.",
        ),
        Policy(
            "training-coverage",
            "score",
            ("completed_drills", "failed_drills", "supervisor_signed"),
            "If supervisor signed, grade max(0, min(4, completed drills minus failed drills)); otherwise grade zero. Across admissible worlds report the lowest grade.",
            "If supervisor signed, grade min(4, completed drills); otherwise grade zero.",
        ),
        Policy(
            "triage-throughput",
            "score",
            ("cleared_bays", "blocked_bays", "chief_signed"),
            "If chief signed, grade max(0, min(4, cleared bays minus blocked bays)); otherwise grade zero. Across admissible worlds report the lowest grade.",
            "If chief signed, grade min(4, cleared bays); otherwise grade zero.",
        ),
    )
}
POLICIES = {**V8_POLICIES, **NEW_POLICIES}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def order_value(secret: bytes, slug: str, value: Any) -> Any:
    """Use one answer-independent candidate order in evidence and criteria."""
    if isinstance(value, dict):
        return dict(
            sorted(
                value.items(),
                key=lambda pair: opaque(secret, f"v9:{slug}:option:{pair[0]}", 64),
            )
        )
    if isinstance(value, list):
        return sorted(
            value,
            key=lambda key: opaque(secret, f"v9:{slug}:option:{key}", 64),
        )
    return value


def validate(policy: Policy, facts: dict[str, Any]) -> None:
    if policy.id not in NEW_POLICIES:
        v8_validate(policy, facts)
        return
    if set(facts) != set(policy.fields):
        raise ValueError("Policy fields disagree")
    if policy.id == "port-berth-arrival":
        arrival = facts["arrival_minutes"]
        draft = facts["draft_meters"]
        licensed = facts["licensed"]
        if (
            not isinstance(arrival, dict)
            or not isinstance(draft, dict)
            or not isinstance(licensed, list)
            or not 3 <= len(arrival) <= 5
            or set(arrival) != set(draft)
            or not set(licensed) <= set(arrival)
            or any(
                type(v) is not int or not 0 <= v <= 200
                for v in [*arrival.values(), *draft.values()]
            )
        ):
            raise ValueError("Invalid berth arrival facts")
        return
    for field, value in facts.items():
        if field in (
            "operator_signed",
            "permit_current",
            "supervisor_signed",
            "chief_signed",
        ):
            if type(value) is not bool:
                raise ValueError("Signature or permit must be boolean")
        elif type(value) is not int or not 0 <= value <= 20:
            raise ValueError("Operational count outside domain")


def evaluate(policy: Policy, facts: dict[str, Any], *, archived: bool = False) -> Any:
    validate(policy, facts)
    if policy.id not in NEW_POLICIES:
        return v8_evaluate(policy, facts, archived=archived)
    if policy.id == "substation-restart":
        return (
            facts["relay_tests"] >= (2 if archived else 3)
            and facts["isolated_faults"] >= (1 if archived else 2)
            and facts["operator_signed"]
        )
    if policy.id == "harbor-entry":
        return (
            facts["trained_crew"] >= (3 if archived else 2)
            and facts["residual_risk"] <= (2 if archived else 3)
            and facts["permit_current"]
        )
    if policy.id == "port-berth-arrival":
        candidates = [
            name
            for name in facts["licensed"]
            if archived or facts["draft_meters"][name] <= 8
        ]
        return (
            min(candidates, key=lambda name: (facts["arrival_minutes"][name], name))
            if candidates
            else "hold"
        )
    if policy.id == "training-coverage":
        return (
            min(
                4,
                max(
                    0,
                    facts["completed_drills"]
                    - (0 if archived else facts["failed_drills"]),
                ),
            )
            if facts["supervisor_signed"]
            else 0
        )
    return (
        min(
            4,
            max(0, facts["cleared_bays"] - (0 if archived else facts["blocked_bays"])),
        )
        if facts["chief_signed"]
        else 0
    )


def reference(policy: Policy, facts: dict[str, Any], *, archived: bool = False) -> Any:
    if policy.id not in NEW_POLICIES:
        return v8_reference(policy, facts, archived=archived)
    validate(policy, facts)
    if policy.id == "port-berth-arrival":
        eligible = [
            name
            for name in facts["licensed"]
            if archived or facts["draft_meters"][name] <= 8
        ]
        return (
            sorted(eligible, key=lambda name: (facts["arrival_minutes"][name], name))[0]
            if eligible
            else "hold"
        )
    if policy.id == "training-coverage":
        if not facts["supervisor_signed"]:
            return 0
        return max(
            0,
            min(
                4,
                facts["completed_drills"] - (0 if archived else facts["failed_drills"]),
            ),
        )
    if policy.id == "triage-throughput":
        if not facts["chief_signed"]:
            return 0
        return max(
            0,
            min(4, facts["cleared_bays"] - (0 if archived else facts["blocked_bays"])),
        )
    threshold = {
        "substation-restart": ((2, 1) if archived else (3, 2)),
        "harbor-entry": ((3, 2) if archived else (2, 3)),
    }[policy.id]
    if policy.id == "substation-restart":
        tests = facts["relay_tests"] >= threshold[0]
        isolation = facts["isolated_faults"] >= threshold[1]
        return all([tests, isolation, facts["operator_signed"]])
    crew = facts["trained_crew"] >= threshold[0]
    risk = facts["residual_risk"] <= threshold[1]
    return all([crew, risk, facts["permit_current"]])


DOC_START = re.compile(r"^BEGIN ([0-9a-f]{12}) \| ([a-z]+) \| FILE ([0-9a-f]{12})$")
DOC_END = re.compile(r"^END ([0-9a-f]{12})$")
FACT = re.compile(r"^ATTESTED ([a-z_]+) = (.+)$")
REVISION = re.compile(r"^REVISION ([A-Z])$")
ACCEPT = re.compile(r"^ACCEPT ([a-z_]+) REVISION ([A-Z])$")
ALIAS = re.compile(r"^ALIAS FILE ([0-9a-f]{12}) MAPS TO ([0-9a-f]{12})$")
EVENT = re.compile(r"^EVENT ([a-z_]+) DAY (\d+) VALUE (.+)$")
CUTOFF = re.compile(r"^EFFECTIVE CUTOFF DAY (\d+)$")
NUMBER_WORDS = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|first|second|third)\b",
    re.I,
)


def document_blocks(state: str) -> list[str]:
    lines = state.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        if DOC_START.fullmatch(lines[index]) is None:
            index += 1
            continue
        start = index
        ident = DOC_START.fullmatch(lines[index])[1]
        index += 1
        while index < len(lines) and lines[index] != f"END {ident}":
            index += 1
        if index == len(lines):
            raise ValueError("Unclosed signed document")
        blocks.append("\n".join(lines[start : index + 1]))
        index += 1
    return blocks


def parse_documents(state: str) -> list[dict[str, Any]]:
    result = []
    for block in document_blocks(state):
        lines = block.splitlines()
        header = DOC_START.fullmatch(lines[0])
        if header is None or DOC_END.fullmatch(lines[-1])[1] != header[1]:
            raise ValueError("Malformed document enclosure")
        matches = {
            "fact": [m for line in lines for m in [FACT.fullmatch(line)] if m],
            "revision": [m for line in lines for m in [REVISION.fullmatch(line)] if m],
            "accept": [m for line in lines for m in [ACCEPT.fullmatch(line)] if m],
            "alias": [m for line in lines for m in [ALIAS.fullmatch(line)] if m],
            "events": [m for line in lines for m in [EVENT.fullmatch(line)] if m],
            "cutoff": [m for line in lines for m in [CUTOFF.fullmatch(line)] if m],
        }
        result.append(
            {
                "id": header[1],
                "format": header[2],
                "file": header[3],
                "fact": (
                    (matches["fact"][0][1], json.loads(matches["fact"][0][2]))
                    if len(matches["fact"]) == 1
                    else None
                ),
                "revision": (
                    matches["revision"][0][1] if len(matches["revision"]) == 1 else None
                ),
                "accept": (
                    (matches["accept"][0][1], matches["accept"][0][2])
                    if len(matches["accept"]) == 1
                    else None
                ),
                "alias": (
                    (matches["alias"][0][1], matches["alias"][0][2])
                    if len(matches["alias"]) == 1
                    else None
                ),
                "events": [
                    (m[1], int(m[2]), json.loads(m[3])) for m in matches["events"]
                ],
                "cutoff": (
                    int(matches["cutoff"][0][1])
                    if len(matches["cutoff"]) == 1
                    else None
                ),
            }
        )
    return result


def derive_facts(
    rows: list[dict[str, Any]], target: str, mechanism: str
) -> dict[str, Any]:
    facts = {}
    aliases = {
        alias: dest
        for row in rows
        if (mapping := row["alias"])
        for alias, dest in [mapping]
    }
    selectors = {
        field: rev
        for row in rows
        if (choice := row["accept"])
        for field, rev in [choice]
    }
    cutoffs = [
        row["cutoff"]
        for row in rows
        if row["cutoff"] is not None and row["file"] == target
    ]
    if len(cutoffs) > 1:
        raise ValueError("Conflicting effective cutoffs")
    for row in rows:
        if row["file"] != target and aliases.get(row["file"]) != target:
            continue
        if row["fact"] is not None:
            field, value = row["fact"]
            if row["revision"] is not None and selectors.get(field) != row["revision"]:
                continue
            if field in facts:
                raise ValueError("Conflicting attested values")
            facts[field] = value
        if row["events"] and mechanism == "long_timeline" and cutoffs:
            cutoff = cutoffs[0]
            active = [
                (day, value) for field, day, value in row["events"] if day <= cutoff
            ]
            if active:
                field = row["events"][0][0]
                if field in facts:
                    raise ValueError("Timeline field collides with attestation")
                facts[field] = max(active, key=lambda item: item[0])[1]
    return facts


def render_document(
    secret: bytes, slug: str, spec: dict[str, Any], ids: dict[str, str], index: int
) -> str:
    doc_id = opaque(secret, f"v9:{slug}:document:{spec['slug']}", 12)
    if spec["format"] not in {
        "dispatch",
        "invoice",
        "log",
        "minutes",
        "letter",
        "memo",
        "ticket",
    }:
        raise ValueError("Unsupported evidence form")
    text = spec["text"].strip()
    if len(text.split()) < 45 or any(
        phrase in text.lower()
        for phrase in ("correct answer", "therefore choose", "therefore grade")
    ):
        raise ValueError("Thin evidence or explicit answer cue")
    if re.search(
        r"(?m)^\s*(BEGIN |END |ATTESTED |REVISION |ACCEPT |ALIAS FILE |EVENT |EFFECTIVE CUTOFF)",
        text,
    ):
        raise ValueError("Authored prose spoofs machine evidence syntax")
    role = spec["role"]
    case = ids[role]
    lines = [f"BEGIN {doc_id} | {spec['format']} | FILE {case}", spec["title"], text]
    kind = spec["kind"]
    if kind in {"fact", "revision_fact"}:
        if kind == "revision_fact":
            lines.append(f"REVISION {spec['revision']}")
        lines.append(
            f"ATTESTED {spec['field']} = {json.dumps(order_value(secret, slug, spec['value']), ensure_ascii=False)}"
        )
    elif kind == "amendment":
        lines.append(f"ACCEPT {spec['field']} REVISION {spec['revision']}")
    elif kind == "alias":
        lines.append(f"ALIAS FILE {ids['alias']} MAPS TO {ids['target']}")
    elif kind == "timeline":
        for event in spec["events"]:
            lines.append(
                f"EVENT {spec['field']} DAY {event['day']} VALUE {json.dumps(event['value'])}"
            )
    elif kind == "cutoff":
        lines.append(f"EFFECTIVE CUTOFF DAY {spec['day']}")
    else:
        raise ValueError("Unsupported evidence kind")
    lines.append(f"END {doc_id}")
    return "\n".join(lines)


def aggregate(policy: Policy, outputs: list[Any]) -> Any:
    if policy.kind == "choice":
        return outputs[0] if len(set(outputs)) == 1 else "hold"
    if policy.kind == "noul":
        return all(outputs)
    return min(outputs)


def same_facts(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return set(a) == set(b) and all(
        (
            (set(a[field]) == set(b[field]))
            if isinstance(a[field], list) and isinstance(b[field], list)
            else a[field] == b[field]
        )
        for field in a
    )


def remove_document(state: str, doc_id: str) -> str:
    blocks = document_blocks(state)
    target = [block for block in blocks if block.startswith(f"BEGIN {doc_id} |")]
    if len(target) != 1:
        raise ValueError("Source ablation must remove exactly one document")
    return state.replace(target[0], "", 1).replace("\n\n\n\n", "\n\n")


def build_item(
    spec: dict[str, Any], secret: bytes
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    slug = spec["slug"]
    policy = POLICIES[spec["policy_id"]]
    mechanism = spec["mechanism"]
    if mechanism not in MECHANISM_COUNTS or len(spec["scene"].split()) < 55:
        raise ValueError("Unplanned mechanism or thin introduction")
    if re.search(r"\d", spec["scene"]) or NUMBER_WORDS.search(spec["scene"]):
        raise ValueError("Introduction contains numeric answer cues")
    if re.search(
        r"\b(?:below|attached|following|appears here|enclosed)\b", spec["scene"], re.I
    ):
        raise ValueError("Introduction inventories evidence and may break ablation")
    ids = {
        name: opaque(secret, f"v9:{slug}:file:{name}", 12)
        for name in ("target", "neighbor", "alias")
    }
    docs = spec["documents"]
    if len(docs) != len({doc["slug"] for doc in docs}):
        raise ValueError("Duplicate document slug")
    if mechanism == "long_amendment" and len(docs) != 5:
        raise ValueError("Amendment dossier needs five documents")
    if mechanism == "long_alias" and len(docs) != 4:
        raise ValueError("Alias dossier needs four documents")
    if mechanism == "long_timeline" and len(docs) not in (4, 5):
        raise ValueError("Timeline dossier needs four or five documents")
    rendered = {
        doc["slug"]: render_document(secret, slug, doc, ids, index)
        for index, doc in enumerate(docs)
    }
    order = list(rendered)
    random.Random(int(opaque(secret, f"v9:{slug}:doc-order", 16), 16)).shuffle(order)
    policy_text = (
        f"File {ids['target']} needs one decision under the current signed rule. "
        "An archived rule is shown solely to resolve version confusion. Evidence "
        "with another file identifier does not transfer unless a signed alias map "
        "explicitly links it. A revision attestation counts only when a signed "
        "amendment accepts that revision; a timeline takes its latest event at or "
        "before a signed effective cutoff. A missing required field has only the "
        "listed admissible completions. Choice holds when winners differ across "
        "worlds, Noul needs every world true, and Score takes the minimum grade.\n"
        f"CURRENT RULE: {policy.current}\nARCHIVED RULE: {policy.archived}"
    )
    envelope = ""
    missing = spec.get("missing_field")
    if mechanism == "missing_source":
        if missing not in policy.fields:
            raise ValueError("Missing field outside policy")
        envelope = f"\nMISSING FIELD {missing}; admissible values: {json.dumps(spec['admissible_values'], ensure_ascii=False)}."
    elif missing is not None:
        raise ValueError("Unexpected missing field")
    state = f"{spec['scene'].strip()}\n\n{policy_text}{envelope}\n\n" + "\n\n".join(
        rendered[key] for key in order
    )
    if mechanism.startswith("long_") and not 700 <= len(state.split()) <= 1800:
        raise ValueError("Long dossier outside planned context range")
    parsed = parse_documents(state)
    visible = derive_facts(parsed, ids["target"], mechanism)
    expected = {
        field: value for field, value in spec["facts"].items() if field != missing
    }
    if not same_facts(visible, expected):
        raise ValueError("Rendered evidence differs from authored fact key")
    if missing:
        values = spec["admissible_values"]
        if len(values) < 2 or len(
            {json.dumps(value, sort_keys=True) for value in values}
        ) != len(values):
            raise ValueError("Missing-source envelope lacks distinct completions")
        worlds = [{**visible, missing: value} for value in values]
    else:
        worlds = [visible]
    outputs = []
    for facts in worlds:
        validate(policy, facts)
        left = evaluate(policy, facts)
        if left != reference(policy, facts):
            raise ValueError("Decision and independent reference disagree")
        outputs.append(left)
    answer = aggregate(policy, outputs)
    archived = aggregate(
        policy, [reference(policy, facts, archived=True) for facts in worlds]
    )
    if mechanism == "rule_precedence" and answer == archived:
        raise ValueError("Archived/current conflict missing")
    essential = spec["essential"]
    if any(
        doc_slug not in rendered or field not in policy.fields
        for doc_slug, field in essential.items()
    ):
        raise ValueError("Essential-source map invalid")
    ablations = []
    for doc_slug, field in essential.items():
        reduced_state = remove_document(
            state, DOC_START.fullmatch(rendered[doc_slug].splitlines()[0])[1]
        )
        reduced = derive_facts(parse_documents(reduced_state), ids["target"], mechanism)
        if field in reduced or not same_facts(
            reduced, {k: v for k, v in visible.items() if k != field}
        ):
            raise ValueError("Source ablation did not remove exactly its target field")
        alternative = spec["counterfactuals"][doc_slug]
        changed_worlds = [{**facts, field: alternative} for facts in worlds]
        for facts in changed_worlds:
            validate(policy, facts)
        changed = aggregate(
            policy, [reference(policy, facts) for facts in changed_worlds]
        )
        if changed == answer:
            raise ValueError("Purportedly essential source leaves answer invariant")
        ablations.append(
            {
                "parent_id": opaque(secret, f"v9:{slug}:item", 16),
                "omitted_source": DOC_START.fullmatch(
                    rendered[doc_slug].splitlines()[0]
                )[1],
                "omitted_field": field,
                "state": reduced_state,
                "review_instruction": (
                    "Do not make a forced decision. From the remaining text alone, "
                    "is the original decision still provable? If not, explain the "
                    "missing evidence without guessing its value."
                ),
            }
        )
    if mechanism.startswith("long_") and len(essential) < 4:
        raise ValueError(
            "Long dossier needs at least four causally essential documents"
        )
    if policy.kind == "choice":
        candidates = next(
            value for value in spec["facts"].values() if isinstance(value, dict)
        )
        labels = sorted(
            [*candidates, "hold"],
            key=lambda key: opaque(secret, f"v9:{slug}:option:{key}", 64),
        )
        criteria: Any = {key: f"Return {key}" for key in labels}
    elif policy.kind == "noul":
        criteria = {"true": "Certified", "false": "Not certified"}
    else:
        criteria = [f"Grade {grade}" for grade in range(5)]
    item_id = opaque(secret, f"v9:{slug}:item", 16)
    prompt = {
        "id": item_id,
        "state": state,
        "questions": {
            "decision": {
                "type": policy.kind,
                "instructions": "Apply the current rule to the target file.",
                "criteria": criteria,
            }
        },
    }
    target = {
        "id": item_id,
        "kind": policy.kind,
        "answer": {policy.kind: answer},
        "source_group": opaque(secret, f"v9:{slug}:source-group", 16),
    }
    proof = {
        "id": item_id,
        "slug": slug,
        "policy_id": policy.id,
        "mechanism": mechanism,
        "visible_facts": visible,
        "world_outputs": outputs,
        "answer": answer,
        "archived_answer": archived,
        "essential_fields": essential,
        "visible_words": len(state.split()),
        "ablation_count": len(ablations),
    }
    for row in ablations:
        row["questions"] = prompt["questions"]
    return prompt, target, proof, ablations


def build(spec_path: Path, salt_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Frozen v9 packet may not be overwritten")
    specs = json.loads(spec_path.read_text())
    secret = salt_path.read_bytes()
    if (
        len(secret) < 32
        or not 1 <= len(specs) <= 12
        or len({s["slug"] for s in specs}) != len(specs)
    ):
        raise ValueError("v9 needs distinct private scenarios and one private salt")
    prompts, targets, proofs, ablations = [], [], [], []
    for spec in specs:
        prompt, target, proof, rows = build_item(spec, secret)
        prompts.append(prompt)
        targets.append(target)
        proofs.append(proof)
        ablations.extend(rows)
    if len(specs) == 12:
        if (
            Counter(t["kind"] for t in targets) != KIND_COUNTS
            or Counter(p["mechanism"] for p in proofs) != MECHANISM_COUNTS
        ):
            raise ValueError("Full pilot misses preregistered type or mechanism cells")
    order = sorted(
        range(len(prompts)),
        key=lambda i: opaque(secret, f"v9:row:{prompts[i]['id']}", 64),
    )
    output.mkdir(parents=True)
    private = output / "private"
    private.mkdir(mode=0o700)
    for path, rows in (
        (output / "prompts.jsonl", [prompts[i] for i in order]),
        (output / "ablations.gold-free.jsonl", ablations),
        (private / "targets.jsonl", [targets[i] for i in order]),
        (private / "proof_traces.jsonl", [proofs[i] for i in order]),
    ):
        path.write_bytes(b"".join(compact(row) for row in rows))
        path.chmod(0o600)
    choice_positions = Counter()
    for prompt, target in zip(prompts, targets):
        if target["kind"] == "choice":
            labels = list(prompt["questions"]["decision"]["criteria"])
            choice_positions[labels.index(target["answer"]["choice"]) + 1] += 1
    position_gate = len(choice_positions) >= 3 and max(choice_positions.values()) <= 2
    receipt = {
        "version": VERSION,
        "status": "AUTOMATED_PROOF_ONLY" if position_gate else "BLOCKED_POSITION_GATE",
        "release_qualified": False,
        "blind_review_passed": False,
        "accepted": len(prompts),
        "type_counts": dict(Counter(t["kind"] for t in targets)),
        "mechanism_counts": dict(Counter(p["mechanism"] for p in proofs)),
        "choice_positions_one_based": dict(choice_positions),
        "position_gate": position_gate,
        "long_lengths": sorted(
            p["visible_words"] for p in proofs if p["mechanism"].startswith("long_")
        ),
        "ablation_rows": len(ablations),
        "spec_sha256": sha(spec_path.read_bytes()),
        "salt_commitment_sha256": sha(secret),
        "prompts_sha256": sha((output / "prompts.jsonl").read_bytes()),
        "blind_ablation_sha256": sha(
            (output / "ablations.gold-free.jsonl").read_bytes()
        ),
        "targets_sha256": sha((private / "targets.jsonl").read_bytes()),
        "proof_sha256": sha((private / "proof_traces.jsonl").read_bytes()),
    }
    (private / "audit.json").write_bytes(compact(receipt))
    (private / "audit.json").chmod(0o600)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--private-salt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.specs, args.private_salt, args.output_dir)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "status",
                    "accepted",
                    "prompts_sha256",
                    "blind_ablation_sha256",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
