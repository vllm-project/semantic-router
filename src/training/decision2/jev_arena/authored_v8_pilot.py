"""Build a gold-separated authored DEV pilot; never qualify a release panel.

Private scenario specifications and salt live on the authorized experiment host.
This module contains only the public rendering and mechanical proof contract.
Independent gold-blind editorial review is mandatory after packet freeze.
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
from .authored_v7_policies import POLICIES as V7_POLICIES
from .authored_v7_policies import Policy
from .authored_v7_policies import evaluate as v7_evaluate
from .authored_v7_policies import reference as v7_reference
from .authored_v7_policies import validate as v7_validate

VERSION = "jevarena-authored-v8-dev12-editorial-pilot/1"
KIND_COUNTS = {"choice": 4, "noul": 4, "score": 4}
CHALLENGE_COUNTS = {
    "long_join": 3,
    "missing_source": 3,
    "rule_precedence": 3,
    "ordinary": 3,
}
ARCHIVE_POLICY = Policy(
    "archive-readiness",
    "noul",
    ("verified_copies", "independent_sites", "recovery_minutes"),
    "Certify restoration readiness only when at least three verified copies are held at two or more independent sites and the tested recovery time is at most 45 minutes. Across admissible completions certify true only if all worlds pass.",
    "Certify restoration readiness with two verified copies at one site and a recovery time at most 60 minutes.",
)
POLICIES = {**V7_POLICIES, ARCHIVE_POLICY.id: ARCHIVE_POLICY}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate(policy: Policy, facts: dict[str, Any]) -> None:
    if policy.id != ARCHIVE_POLICY.id:
        v7_validate(policy, facts)
        return
    if set(facts) != set(policy.fields) or any(
        type(value) is not int or not 0 <= value <= 200 for value in facts.values()
    ):
        raise ValueError("Invalid archive readiness fields")


def evaluate(policy: Policy, facts: dict[str, Any], *, archived: bool = False) -> Any:
    validate(policy, facts)
    if policy.id != ARCHIVE_POLICY.id:
        return v7_evaluate(policy, facts, archived=archived)
    if archived:
        return (
            facts["verified_copies"] >= 2
            and facts["independent_sites"] >= 1
            and facts["recovery_minutes"] <= 60
        )
    return (
        facts["verified_copies"] >= 3
        and facts["independent_sites"] >= 2
        and facts["recovery_minutes"] <= 45
    )


def reference(policy: Policy, facts: dict[str, Any], *, archived: bool = False) -> Any:
    if policy.id != ARCHIVE_POLICY.id:
        return v7_reference(policy, facts, archived=archived)
    validate(policy, facts)
    requirements = (2, 1, 60) if archived else (3, 2, 45)
    return all(
        (
            facts["verified_copies"] >= requirements[0],
            facts["independent_sites"] >= requirements[1],
            facts["recovery_minutes"] <= requirements[2],
        )
    )


def order_value(secret: bytes, slug: str, value: Any) -> Any:
    if isinstance(value, dict):
        return dict(
            sorted(
                value.items(),
                key=lambda kv: opaque(secret, f"v8:{slug}:candidate:{kv[0]}", 64),
            )
        )
    if isinstance(value, list):
        return sorted(
            value,
            key=lambda key: opaque(secret, f"v8:{slug}:candidate:{key}", 64),
        )
    return value


HEADER = re.compile(
    r"^DOCUMENT ([0-9a-f]{12}) \| FORMAT ([a-z]+) \| FILE ([0-9a-f]{12}) "
    r"\| DAY (\d+) \| STATUS (CURRENT|VOID)$"
)
FIELD = re.compile(r"^ATTESTED ([a-z_]+) = (.+)$")


def parse_documents(state: str) -> list[dict[str, Any]]:
    """Read only explicitly current signed evidence; VOID stays void if later docs vanish."""
    lines = state.splitlines()
    result = []
    for index, line in enumerate(lines):
        header = HEADER.fullmatch(line)
        if header is None:
            continue
        end = next(
            (j for j in range(index + 1, len(lines)) if HEADER.fullmatch(lines[j])),
            len(lines),
        )
        fields = [FIELD.fullmatch(row) for row in lines[index + 1 : end]]
        fields = [row for row in fields if row is not None]
        if len(fields) != 1:
            raise ValueError("Every signed document needs one attested field")
        result.append(
            {
                "doc_id": header[1],
                "format": header[2],
                "case_id": header[3],
                "day": int(header[4]),
                "status": header[5],
                "field": fields[0][1],
                "value": json.loads(fields[0][2]),
            }
        )
    return result


def selected(
    rows: list[dict[str, Any]], case_id: str
) -> tuple[dict[str, Any], dict[str, str]]:
    chosen: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["case_id"] != case_id or row["status"] != "CURRENT":
            continue
        old = chosen.get(row["field"])
        if old is None or row["day"] > old["day"]:
            chosen[row["field"]] = row
        elif row["day"] == old["day"]:
            raise ValueError("Same-day current source conflict")
    return (
        {field: row["value"] for field, row in chosen.items()},
        {field: row["doc_id"] for field, row in chosen.items()},
    )


def render_doc(
    secret: bytes, slug: str, spec: dict[str, Any], case_ids: dict[str, str], index: int
) -> str:
    required = {"role", "day", "status", "field", "value", "format", "title", "text"}
    if set(spec) != required:
        raise ValueError("Document schema differs from v8 contract")
    if spec["role"] not in case_ids or spec["status"] not in {"CURRENT", "VOID"}:
        raise ValueError("Invalid signed document scope or status")
    if spec["format"] not in {"memo", "email", "ledger", "letter", "ticket", "minutes"}:
        raise ValueError("Invalid document form")
    text = spec["text"].strip()
    if len(text.split()) < 35 or any(
        phrase in text.lower()
        for phrase in ("correct answer", "therefore choose", "therefore grade")
    ):
        raise ValueError("Thin document or answer cue")
    doc_id = opaque(secret, f"v8:{slug}:document:{index}", 12)
    return (
        f"DOCUMENT {doc_id} | FORMAT {spec['format']} | FILE {case_ids[spec['role']]} "
        f"| DAY {spec['day']} | STATUS {spec['status']}\n"
        f"{spec['title']}\n{text}\n"
        f"ATTESTED {spec['field']} = {json.dumps(order_value(secret, slug, spec['value']), ensure_ascii=False)}"
    )


def worlds(
    policy: Policy,
    visible: dict[str, Any],
    missing: str | None,
    possibilities: list[Any] | None,
) -> list[dict[str, Any]]:
    if missing is None:
        validate(policy, visible)
        return [visible]
    if missing in visible or not possibilities or len(possibilities) < 2:
        raise ValueError("Absent source needs at least two admissible completions")
    if len({json.dumps(v, sort_keys=True) for v in possibilities}) != len(
        possibilities
    ):
        raise ValueError("Duplicate admissible completion")
    result = [{**visible, missing: value} for value in possibilities]
    for facts in result:
        validate(policy, facts)
    return result


def aggregate(policy: Policy, outputs: list[Any]) -> Any:
    if policy.kind == "choice":
        return outputs[0] if len(set(outputs)) == 1 else "hold"
    if policy.kind == "noul":
        return all(outputs)
    return min(outputs)


def build_item(
    spec: dict[str, Any], secret: bytes
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    policy = POLICIES[spec["policy_id"]]
    slug = spec["slug"]
    challenge = spec["challenge"]
    if challenge not in CHALLENGE_COUNTS or len(spec["scene"].split()) < 45:
        raise ValueError("Unknown challenge or thin scene")
    case_ids = {
        role: opaque(secret, f"v8:{slug}:case:{role}", 12)
        for role in ("target", "neighbor")
    }
    docs = spec["documents"]
    if len(docs) < (7 if challenge == "long_join" else 2):
        raise ValueError("Too few independently signed documents")
    document_text = [
        render_doc(secret, slug, doc, case_ids, index) for index, doc in enumerate(docs)
    ]
    random.Random(int(opaque(secret, f"v8:{slug}:document-order", 16), 16)).shuffle(
        document_text
    )
    missing = spec.get("missing_field")
    if (challenge == "missing_source") != (missing is not None):
        raise ValueError("Challenge/absence disagreement")
    if missing is not None and missing not in policy.fields:
        raise ValueError("Absent field outside policy")
    envelope = ""
    if missing:
        completions = [order_value(secret, slug, v) for v in spec["admissible_values"]]
        envelope = (
            f"\nEvidence envelope: target field {missing} has no current signed document. "
            f"Exactly these values remain admissible: {json.dumps(completions, ensure_ascii=False)}."
        )
    rule = (
        f"Decision request for file {case_ids['target']}. The following current rule governs; "
        "the archived rule is superseded. CURRENT documents count only for their exact "
        "file ID. VOID documents remain void even if a newer document is removed. "
        "Among multiple CURRENT attestations for one field, use the latest day. "
        "For missing fields, use only the explicit evidence envelope: Choice holds "
        "when winners differ, Noul certifies true only if all completions pass, "
        "and Score takes the minimum grade.\n"
        f"Current rule: {policy.current}\nArchived rule: {policy.archived}"
    )
    state = f"{spec['scene'].strip()}\n\n{rule}{envelope}\n\n" + "\n\n".join(
        document_text
    )
    parsed = parse_documents(state)
    visible, source_ids = selected(parsed, case_ids["target"])
    expected = {k: v for k, v in spec["facts"].items() if k != missing}
    if set(visible) != set(expected) or any(
        (
            (set(visible[key]) != set(expected[key]))
            if isinstance(visible[key], list) and isinstance(expected[key], list)
            else visible[key] != expected[key]
        )
        for key in expected
    ):
        raise ValueError("Rendered target evidence does not match authored facts")
    completions = worlds(policy, visible, missing, spec.get("admissible_values"))
    outputs = [evaluate(policy, facts) for facts in completions]
    if any(a != reference(policy, facts) for a, facts in zip(outputs, completions)):
        raise ValueError("Independent oracle disagreement")
    answer = aggregate(policy, outputs)
    archived = aggregate(
        policy, [reference(policy, f, archived=True) for f in completions]
    )
    if challenge == "rule_precedence" and archived == answer:
        raise ValueError("Archived/current rule difference is not consequential")
    if challenge == "missing_source" and len(set(outputs)) == 1:
        raise ValueError("Missing-source worlds must change their decisions")
    if challenge == "long_join":
        if len(state.split()) < 750 or len(state.split()) > 2200:
            raise ValueError("Long dossier outside planned size")
        if {d["format"] for d in docs}.__len__() < 3:
            raise ValueError("Long dossier needs three document forms")
        if not any(d["role"] == "target" and d["status"] == "VOID" for d in docs):
            raise ValueError("Long dossier needs an independently voided target source")
        if not any(d["role"] == "neighbor" for d in docs):
            raise ValueError("Long dossier needs an adjacent-file source")
        if set(spec["counterfactuals"]) != set(policy.fields):
            raise ValueError("Long dossier lacks all field interventions")
        for field, alternative in spec["counterfactuals"].items():
            changed = {**visible, field: alternative}
            validate(policy, changed)
            if evaluate(policy, changed) == answer:
                raise ValueError(f"Inert target field in long dossier: {field}")
            reduced = [row for row in parsed if row["doc_id"] != source_ids[field]]
            remaining, _ = selected(reduced, case_ids["target"])
            if field in remaining:
                raise ValueError("Deleted source can be replaced by a stale source")
    item_id = opaque(secret, f"v8:{slug}:item", 16)
    if policy.kind == "choice":
        options = list(next(v for v in spec["facts"].values() if isinstance(v, dict)))
        options = sorted(
            options,
            key=lambda key: opaque(secret, f"v8:{slug}:candidate:{key}", 64),
        )
        criteria: Any = {key: f"Return {key}" for key in options}
        criteria["hold"] = "Return hold when no candidate qualifies or worlds disagree"
    elif policy.kind == "noul":
        criteria = {"true": "Certified across all worlds", "false": "Not certified"}
    else:
        criteria = [f"Grade {grade}" for grade in range(5)]
    prompt = {
        "id": item_id,
        "state": state,
        "questions": {
            "decision": {
                "type": policy.kind,
                "instructions": f"Apply the current rule to file {case_ids['target']}.",
                "criteria": criteria,
            }
        },
    }
    target = {
        "id": item_id,
        "kind": policy.kind,
        "answer": {policy.kind: answer},
        "source_group": opaque(secret, f"v8:{slug}:source-group", 16),
    }
    proof = {
        "id": item_id,
        "slug": slug,
        "policy_id": policy.id,
        "challenge": challenge,
        "case_id": case_ids["target"],
        "visible_facts": visible,
        "selected_sources": source_ids,
        "world_outputs": outputs,
        "current_answer": answer,
        "archived_answer": archived,
        "visible_words": len(state.split()),
    }
    return prompt, target, proof


def build(spec_path: Path, salt_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Pilot packet is immutable")
    specs = json.loads(spec_path.read_text())
    secret = salt_path.read_bytes()
    if len(secret) < 32 or len(specs) != 12 or len({s["slug"] for s in specs}) != 12:
        raise ValueError(
            "v8 needs twelve distinct private scenarios and a private salt"
        )
    prompts, targets, proofs = [], [], []
    for spec in specs:
        prompt, target, proof = build_item(spec, secret)
        prompts.append(prompt)
        targets.append(target)
        proofs.append(proof)
    kinds = Counter(t["kind"] for t in targets)
    challenges = Counter(p["challenge"] for p in proofs)
    if kinds != KIND_COUNTS or challenges != CHALLENGE_COUNTS:
        raise ValueError("Unbalanced v8 editorial pilot")
    order = sorted(
        range(12), key=lambda i: opaque(secret, f"v8:row:{prompts[i]['id']}", 64)
    )
    output.mkdir(parents=True)
    private = output / "private"
    private.mkdir(mode=0o700)
    for path, rows in (
        (output / "prompts.jsonl", [prompts[i] for i in order]),
        (private / "targets.jsonl", [targets[i] for i in order]),
        (private / "proof_traces.jsonl", [proofs[i] for i in order]),
    ):
        path.write_bytes(b"".join(compact(row) for row in rows))
        path.chmod(0o600)
    receipt = {
        "version": VERSION,
        "status": "AUTOMATED_PROOF_ONLY",
        "release_qualified": False,
        "blind_review_passed": False,
        "accepted": 12,
        "kinds": dict(kinds),
        "challenges": dict(challenges),
        "spec_sha256": sha(spec_path.read_bytes()),
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
    print(json.dumps({k: receipt[k] for k in ("status", "accepted", "prompts_sha256")}))


if __name__ == "__main__":
    main()
