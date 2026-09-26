"""Gold-separated v7 semantic DEV pilot; no release qualification.

The caller supplies private human-authored source prose, facts and a private
salt. Frozen gold-free prompts, keys, and proof are written to separate files.
No builder assertion substitutes for independent gold-blind review.
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
from .authored_v7_policies import (
    POLICIES,
    Policy,
    aggregate,
    evaluate,
    reference,
    validate,
)

VERSION = "jevarena-authored-v7-dev12-semantic-pilot/2"
TYPE_COUNTS = {"choice": 4, "noul": 3, "score": 5}
CHALLENGE_COUNTS = {
    "long_join": 3,
    "missing_source": 4,
    "rule_precedence": 2,
    "ordinary": 3,
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def answer(policy: Policy, output: str | bool | int) -> dict[str, Any]:
    return {"type": policy.kind, policy.kind: output}


def ordered_value(secret: bytes, slug: str, value: Any) -> Any:
    """Candidate order is derived from names and private salt, never values."""
    if isinstance(value, dict):
        return dict(
            sorted(
                value.items(),
                key=lambda row: opaque(secret, f"v7:{slug}:candidate:{row[0]}", 64),
            )
        )
    if isinstance(value, list):
        return sorted(
            value, key=lambda key: opaque(secret, f"v7:{slug}:candidate:{key}", 64)
        )
    return value


def source(
    secret: bytes,
    slug: str,
    case_id: str,
    field: str,
    value: Any,
    date: int,
    prose: str,
    serial: int,
) -> dict[str, Any]:
    if len(prose.split()) < 48 or "SIGNED FIELD" in prose or "SOURCE " in prose:
        raise ValueError("Source prose too thin or spoofs source syntax")
    if any(
        token in prose.lower()
        for token in ("therefore the answer", "the correct answer", "the grade is")
    ):
        raise ValueError("Source prose explicitly leaks the answer")
    doc_id = opaque(secret, f"v7:{slug}:source:{serial}", 12)
    display = ordered_value(secret, slug, value)
    return {
        "doc_id": doc_id,
        "case_id": case_id,
        "date": date,
        "field": field,
        "value": value,
        "body": (
            f"SOURCE {doc_id} | CASE {case_id} | SIGNED DAY {date}\n"
            f"{prose.strip()}\nSIGNED FIELD {field} = {json.dumps(display, sort_keys=False)}"
        ),
    }


SOURCE_RE = re.compile(
    r"^SOURCE ([0-9a-f]{12}) \| CASE ([0-9a-f]{12}) \| SIGNED DAY (\d+)$"
)
FIELD_RE = re.compile(r"^SIGNED FIELD ([a-z_]+) = (.+)$")


def parse_sources(state: str) -> list[dict[str, Any]]:
    rows = []
    lines = state.splitlines()
    for index, line in enumerate(lines):
        m = SOURCE_RE.fullmatch(line)
        if not m:
            continue
        next_header = next(
            (j for j in range(index + 1, len(lines)) if SOURCE_RE.fullmatch(lines[j])),
            len(lines),
        )
        matches = [FIELD_RE.fullmatch(row) for row in lines[index + 1 : next_header]]
        matches = [row for row in matches if row]
        if len(matches) != 1:
            raise ValueError("Visible source does not contain one signed field")
        rows.append(
            {
                "doc_id": m[1],
                "case_id": m[2],
                "date": int(m[3]),
                "field": matches[0][1],
                "value": json.loads(matches[0][2]),
            }
        )
    return rows


def selected_facts(
    rows: list[dict[str, Any]], case_id: str
) -> tuple[dict[str, Any], dict[str, str]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["case_id"] != case_id:
            continue
        field = row["field"]
        if field not in selected or row["date"] > selected[field]["date"]:
            selected[field] = row
        elif row["date"] == selected[field]["date"]:
            raise ValueError("Same-day signed source conflict")
    return (
        {key: row["value"] for key, row in selected.items()},
        {key: row["doc_id"] for key, row in selected.items()},
    )


def semantically_equal_facts(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    return all(
        (
            set(left[key]) == set(right[key])
            if isinstance(left[key], list) and isinstance(right[key], list)
            else left[key] == right[key]
        )
        for key in left
    )


def worlds(
    policy: Policy,
    facts: dict[str, Any],
    missing_field: str | None,
    admissible: list[Any] | None,
) -> list[dict[str, Any]]:
    if missing_field is None:
        validate(policy, facts)
        return [facts]
    if missing_field in facts or not admissible or len(admissible) < 2:
        raise ValueError("Missing source must be absent with at least two completions")
    if any(row == admissible[0] for row in admissible[1:]):
        raise ValueError("Missing-source completion duplicates")
    completions = [{**facts, missing_field: value} for value in admissible]
    for row in completions:
        validate(policy, row)
    return completions


def question(
    policy: Policy, facts: dict[str, Any], case_id: str, secret: bytes, slug: str
) -> dict[str, Any]:
    if policy.kind == "choice":
        keys = list(next(value for value in facts.values() if isinstance(value, dict)))
        # The same answer-independent candidate order is used in the source
        # registers and output criteria. R2's separate criterion hash created
        # a first-position shortcut despite clean source-register positions.
        criteria = {
            key: f"Select {key}"
            for key in sorted(
                keys, key=lambda key: opaque(secret, f"v7:{slug}:candidate:{key}", 64)
            )
        }
        criteria["hold"] = "Hold when no option qualifies or admissible worlds differ"
    elif policy.kind == "noul":
        criteria = {
            "true": "Certified in every admissible world",
            "false": "Cannot certify in every admissible world",
        }
    else:
        criteria = [f"Grade {number}" for number in range(5)]
    return {
        "type": policy.kind,
        "instructions": f"For decision file {case_id}, apply the current signed policy to the target case and return one decision.",
        "criteria": criteria,
    }


def build_item(
    spec: dict[str, Any], secret: bytes
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    policy = POLICIES[spec["policy_id"]]
    slug = spec["slug"]
    challenge = spec["challenge"]
    if challenge not in CHALLENGE_COUNTS:
        raise ValueError("Unknown v7 challenge")
    if len(spec["scene"].split()) < 55:
        raise ValueError("Scenario prose is too thin")
    facts = spec["facts"]
    validate(policy, facts)
    case_id = opaque(secret, f"v7:{slug}:target", 12)
    related_id = opaque(secret, f"v7:{slug}:related", 12)
    if case_id == related_id:
        raise ValueError("Target and related cases collided")
    missing = spec.get("missing_field")
    admissible = spec.get("admissible_values")
    if (challenge == "missing_source") != (missing is not None):
        raise ValueError("Missing-source challenge and absence disagree")
    if missing is not None and missing not in policy.fields:
        raise ValueError("Missing field is outside policy")
    documents = []
    prose = spec["source_prose"]
    if set(prose) != set(policy.fields):
        raise ValueError("Each source needs independently authored prose")
    for serial, field in enumerate(policy.fields):
        if field == missing:
            continue
        documents.append(
            source(
                secret,
                slug,
                case_id,
                field,
                facts[field],
                20 + serial,
                prose[field],
                serial,
            )
        )
    related_expected = None
    intervention_proof = {}
    if challenge == "long_join":
        if len(spec["scene"].split()) < 130:
            raise ValueError("Long case needs substantive scenario setup")
        stale_field = spec["stale_field"]
        if (
            stale_field not in policy.fields
            or spec["stale_value"] == facts[stale_field]
        ):
            raise ValueError("Stale source must conflict on a target field")
        stale = {**facts, stale_field: spec["stale_value"]}
        validate(policy, stale)
        if evaluate(policy, stale) == evaluate(policy, facts):
            raise ValueError("Stale target source is not a competing decision")
        documents.append(
            source(
                secret,
                slug,
                case_id,
                stale_field,
                spec["stale_value"],
                9,
                spec["stale_prose"],
                10,
            )
        )
        related = spec["related_facts"]
        validate(policy, related)
        related_expected = evaluate(policy, related)
        if related_expected == evaluate(policy, facts):
            raise ValueError("Related case must compete with target output")
        if set(spec["related_prose"]) != set(policy.fields):
            raise ValueError("Related case needs unique source prose")
        for serial, field in enumerate(policy.fields):
            documents.append(
                source(
                    secret,
                    slug,
                    related_id,
                    field,
                    related[field],
                    20 + serial,
                    spec["related_prose"][field],
                    20 + serial,
                )
            )
        if set(spec["counterfactuals"]) != set(policy.fields):
            raise ValueError("Every long source must be causally consequential")
        for field, alternate in spec["counterfactuals"].items():
            intervention = {**facts, field: alternate}
            validate(policy, intervention)
            changed = evaluate(policy, intervention)
            if changed == evaluate(policy, facts) or changed != reference(
                policy, intervention
            ):
                raise ValueError(
                    "Domain-valid source value intervention did not change output"
                )
            intervention_proof[field] = {
                "intervention_value": alternate,
                "changed_output": changed,
            }
    random.Random(int(opaque(secret, f"v7:{slug}:document-order", 16), 16)).shuffle(
        documents
    )
    governance = (
        f"DECISION FILE {case_id}. The current signed policy below governs this target case. "
        "A superseded archived rule follows for provenance only. Select facts by exact case ID "
        "and, for the same field, the latest signed day. A separate case cannot supply a "
        "missing target field. If a required source is absent, use every completion explicitly "
        "allowed by the signed evidence envelope; do not assume the unobserved value. "
        "Choice holds on differing winners; Noul certifies true only if every admissible "
        "world passes; Score reports the lowest admissible grade.\n"
        f"Current signed rule: {policy.current}\nArchived rule: {policy.archived}"
    )
    envelope = ""
    if missing is not None:
        domain = [ordered_value(secret, slug, value) for value in admissible]
        envelope = (
            f"\n\nEVIDENCE ENVELOPE FOR MISSING FIELD {missing}: The signed field source is absent "
            "from the target packet. Independent surviving bounds permit exactly "
            f"these completions: {json.dumps(domain)}. They are possibilities, not "
            "observed signed values; apply the current rule to all of them."
        )
    state = governance + "\n\n" + spec["scene"].strip() + envelope
    state += "\n\n" + "\n\n".join(doc["body"] for doc in documents)
    parsed = parse_sources(state)
    visible, selected = selected_facts(parsed, case_id)
    expected_visible = {key: value for key, value in facts.items() if key != missing}
    if not semantically_equal_facts(visible, expected_visible):
        raise ValueError("Visible parser and authored target facts disagree")
    completions = worlds(policy, visible, missing, admissible)
    outputs = [evaluate(policy, world) for world in completions]
    if any(
        evaluate(policy, world) != reference(policy, world) for world in completions
    ):
        raise ValueError("Independent decision oracle disagrees")
    result = aggregate(policy, outputs)
    archived = [reference(policy, world, archived=True) for world in completions]
    if challenge == "rule_precedence" and aggregate(policy, archived) == result:
        raise ValueError("Rule precedence has no current/archive conflict")
    if challenge == "long_join":
        related_visible, _ = selected_facts(parsed, related_id)
        if (
            not semantically_equal_facts(related_visible, spec["related_facts"])
            or evaluate(policy, related_visible) != related_expected
        ):
            raise ValueError("Related-case identity proof failed")
        if len(set(selected.values())) != 3:
            raise ValueError("Long target fields do not require three sources")
        if not 750 <= len(state.split()) <= 2200:
            raise ValueError("Long file outside pilot diagnostic length range")
    item_id = opaque(secret, f"v7:{slug}:item", 16)
    prompt = {
        "id": item_id,
        "state": state,
        "questions": {"decision": question(policy, facts, case_id, secret, slug)},
    }
    target = {
        "id": item_id,
        "kind": policy.kind,
        "answer": answer(policy, result),
        "source_group": opaque(secret, f"v7:{slug}:source-group", 16),
    }
    trace = {
        "id": item_id,
        "policy_id": policy.id,
        "kind": policy.kind,
        "challenge": challenge,
        "visible_words": len(state.split()),
        "selected_sources": selected,
        "visible_facts": visible,
        "world_outputs": outputs,
        "archived_outputs": archived,
        "answer": result,
        "missing_field": missing,
        "missing_source_count": (
            sum(row["field"] == missing and row["case_id"] == case_id for row in parsed)
            if missing
            else None
        ),
        "causal_value_interventions": intervention_proof,
        "related_output": related_expected,
    }
    return prompt, target, trace


def build(specs_path: Path, salt_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    specs = json.loads(specs_path.read_text())
    secret = salt_path.read_bytes()
    if (
        len(secret) < 32
        or len(specs) != 12
        or len({row["slug"] for row in specs}) != 12
    ):
        raise ValueError(
            "v7 pilot needs twelve distinct authored scenarios and a private salt"
        )
    prompts, targets, traces, failures = [], [], [], []
    for spec in specs:
        try:
            prompt, target, trace = build_item(spec, secret)
            prompts.append(prompt)
            targets.append(target)
            traces.append(trace)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                {"slug_sha256": sha(spec.get("slug", "").encode()), "reason": str(exc)}
            )
    order = sorted(
        range(len(prompts)),
        key=lambda i: opaque(secret, f"v7:row:{prompts[i]['id']}", 64),
    )
    prompts, targets, traces = (
        [rows[i] for i in order] for rows in (prompts, targets, traces)
    )
    type_counts = Counter(row["kind"] for row in targets)
    challenge_counts = Counter(row["challenge"] for row in traces)
    score_support = sorted(
        row["answer"]["score"] for row in targets if row["kind"] == "score"
    )
    long_proof = all(
        len(row["causal_value_interventions"]) == 3
        for row in traces
        if row["challenge"] == "long_join"
    )
    missing_proof = all(
        row["missing_source_count"] == 0 and len(row["world_outputs"]) >= 2
        for row in traces
        if row["challenge"] == "missing_source"
    )
    gate = (
        not failures
        and type_counts == TYPE_COUNTS
        and challenge_counts == CHALLENGE_COUNTS
        and len({row["policy_id"] for row in traces}) == 12
        and score_support == [0, 1, 2, 3, 4]
        and long_proof
        and missing_proof
    )
    output.mkdir(parents=True)
    private = output / "private"
    private.mkdir(mode=0o700)
    for path, rows in (
        (output / "prompts.jsonl", prompts),
        (private / "targets.jsonl", targets),
        (private / "proof_traces.jsonl", traces),
    ):
        path.write_bytes(b"".join(compact(row) for row in rows))
        path.chmod(0o600)
    receipt = {
        "version": VERSION,
        "status": "AUTOMATED_PROOF_ONLY" if gate else "BLOCKED",
        "release_qualified": False,
        "blind_review_passed": False,
        "accepted": len(prompts),
        "rejected": len(failures),
        "failures": failures,
        "type_counts": dict(type_counts),
        "challenge_counts": dict(challenge_counts),
        "score_support": score_support,
        "long_proof_complete": long_proof,
        "missing_proof_complete": missing_proof,
        "spec_sha256": sha(specs_path.read_bytes()),
        "salt_commitment_sha256": sha(secret),
        "prompts_sha256": sha((output / "prompts.jsonl").read_bytes()),
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
                key: receipt[key]
                for key in ("status", "accepted", "rejected", "prompts_sha256")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
