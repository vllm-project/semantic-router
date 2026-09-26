"""Exploratory DEV60 assembly with private semantic proof traces.

This is a feasibility panel, never a release set. The 20 hand-authored dossier
cases are supplied separately; this module builds 40 smaller, distinct source
scenarios across policy precedence, partial evidence and near-case confusion.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .authored_v5_dossier import (
    BULLET_RE,
    DOCUMENT_RE,
    FIELD_LABELS,
    PROSE_RE,
    TABLE_RE,
    _document,
    _parse_fact,
    _parse_value,
    compact,
    opaque,
    sha_bytes,
)
from .authored_v5_reference import (
    evaluate_reference,
    governing_reference,
    swap_priority,
)
from .authored_v5_registry import (
    PAIRS,
    PolicyPair,
    evaluate_archived,
    evaluate_current,
    prove_conflict,
    validate_facts,
)

VERSION = "jevarena-authored-v5-dev60-feasibility/1"
UNCERTAIN_RE = re.compile(
    r"^The certified (?P<label>[a-z -]+) remains disputed: (?P<first>.+) versus (?P<second>.+)\. Both signed source worlds remain admissible\.$"
)
STYLES = ("prose", "bullets", "table")


def _uncertain_line(field: str, first: Any, second: Any) -> str:
    from .authored_v5_dossier import _value_text

    return (
        f"The certified {FIELD_LABELS[field]} remains disputed: "
        f"{_value_text(first)} versus {_value_text(second)}. "
        "Both signed source worlds remain admissible."
    )


def _read_visible(state: str, pair: PolicyPair) -> list[dict[str, Any]]:
    documents = []
    reversed_labels = {FIELD_LABELS[field]: field for field in pair.fields}
    for section in state.split("\n\n"):
        lines = section.splitlines()
        header = DOCUMENT_RE.fullmatch(lines[0]) if lines else None
        if header is None:
            continue
        if len(lines) != 4:
            raise ValueError("Short case document does not have four visible lines")
        disputed = UNCERTAIN_RE.fullmatch(lines[2])
        if disputed:
            field = reversed_labels.get(disputed["label"])
            if field is None:
                raise ValueError("Disputed field is outside policy schema")
            value = [
                _parse_value(field, disputed["first"]),
                _parse_value(field, disputed["second"]),
            ]
        else:
            if not any(
                pattern.fullmatch(lines[2])
                for pattern in (PROSE_RE, BULLET_RE, TABLE_RE)
            ):
                raise ValueError("Short case has an unparsable visible fact")
            field, value = _parse_fact(lines[2], pair)
        documents.append(
            {
                "doc_id": header["doc"],
                "case_id": header["case"],
                "filed_day": int(header["day"]),
                "field": field,
                "value": value,
            }
        )
    if len({document["doc_id"] for document in documents}) != len(documents):
        raise ValueError("Duplicate short-case source document")
    return documents


def _resolve_worlds(
    documents: list[dict[str, Any]],
    case_id: str,
    pair: PolicyPair,
    *,
    disputed: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    selected = {}
    for document in documents:
        if document["case_id"] != case_id:
            continue
        field = document["field"]
        previous = selected.get(field)
        if previous is None or document["filed_day"] > previous["filed_day"]:
            selected[field] = document
        elif document["filed_day"] == previous["filed_day"]:
            raise ValueError("Ambiguous same-day short-case sources")
    if set(selected) != set(pair.fields):
        raise ValueError("Short case is missing a required visible fact")
    first = {}
    second = {}
    for field in pair.fields:
        value = selected[field]["value"]
        if field == disputed:
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("Expected exactly two visible possible worlds")
            first[field], second[field] = value
        else:
            first[field] = second[field] = value
    return first, second, {field: selected[field]["doc_id"] for field in pair.fields}


def _question(
    pair: PolicyPair, facts: dict[str, Any], target_id: str
) -> dict[str, Any]:
    if pair.kind == "choice":
        options = {key: f"Registered option {key}" for key in facts[pair.fields[0]]}
        options["hold"] = "Hold because the signed evidence is unresolved"
    elif pair.kind == "noul":
        options = {
            "false": "The current decision condition does not hold",
            "true": "The current decision condition holds",
        }
    else:
        options = [f"Grade {grade}" for grade in range(5)]
    return {
        "type": pair.kind,
        "instructions": f"For decision file {target_id}, apply the current signed policy using the admissible source record.",
        "criteria": options,
    }


def _answer(pair: PolicyPair, value: str | bool | int) -> dict[str, Any]:
    if pair.kind == "choice":
        return {"type": "choice", "choice": value}
    if pair.kind == "noul":
        return {"type": "noul", "noul": value}
    return {"type": "score", "score": value}


def build_short(
    spec: dict[str, Any], secret: bytes
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pair = PAIRS[spec["pair_id"]]
    kind = spec["challenge"]
    if kind not in {"rule_precedence", "partial_evidence", "near_distractor"}:
        raise ValueError("Unsupported exploratory challenge")
    facts = spec["facts"]
    validate_facts(pair, facts)
    if len(spec["scene"].split()) < 35:
        raise ValueError("Private scene is too thin for a readable development case")
    target_id = opaque(secret, f"{spec['slug']}:target", 12)
    decoy_id = target_id[:-1] + ("0" if target_id[-1] != "0" else "1")
    style = STYLES[int(opaque(secret, f"{spec['slug']}:style", 2), 16) % len(STYLES)]
    disputed_field = spec.get("disputed_field") if kind == "partial_evidence" else None
    if kind == "partial_evidence":
        if (
            disputed_field not in pair.fields
            or spec["alternative_value"] == facts[disputed_field]
        ):
            raise ValueError("Two admissible worlds must differ in a material field")
        alternative = {**facts, disputed_field: spec["alternative_value"]}
        validate_facts(pair, alternative)
        first_out = evaluate_current(pair, facts)
        second_out = evaluate_current(pair, alternative)
        resolved = first_out == second_out
        if resolved != (spec["outcome_class"] == "invariant"):
            raise ValueError("Declared partial-evidence outcome is wrong")
        if resolved and first_out == pair.fallback:
            raise ValueError("Invariant partial evidence is a trivial fallback")
        answer = first_out if resolved else pair.fallback
    else:
        alternative = None
        first_out = evaluate_current(pair, facts)
        second_out = None
        resolved = None
        answer = first_out
    if kind == "rule_precedence":
        prove_conflict(pair, facts)
    if kind == "near_distractor":
        related = spec["related_facts"]
        validate_facts(pair, related)
        if evaluate_current(pair, related) == answer:
            raise ValueError("Similar-ID case does not change the answer")
    documents = []
    for index, field in enumerate(pair.fields):
        doc = _document(
            secret=secret,
            slug=spec["slug"],
            serial=index,
            case_id=target_id,
            day=20 + index,
            signer=("Mara Ellis", "Noah Chen", "Amina Patel")[index],
            genre=("intake memorandum", "signed register", "oversight letter")[index],
            intro=f"{spec['field_notes'][index]} The signed entry belongs to the named target file and carries its own accountable author.",
            field=field,
            value=facts[field],
            style=style,
            closing="The office retained the source document with its date so later reviewers can distinguish it from neighboring files and older drafts.",
        )
        if field == disputed_field:
            lines = doc["body"].splitlines()
            lines[2] = _uncertain_line(field, facts[field], alternative[field])
            doc["body"] = "\n".join(lines)
        documents.append(doc)
    if kind == "near_distractor":
        for index, field in enumerate(pair.fields):
            documents.append(
                _document(
                    secret=secret,
                    slug=spec["slug"],
                    serial=20 + index,
                    case_id=decoy_id,
                    day=21 + index,
                    signer=("Rosa Bell", "Leah Morgan", "Jonah Reed")[index],
                    genre="neighboring case register",
                    intro=f"The adjacent case concerns the same process but a different owner. {spec['related_note']}",
                    field=field,
                    value=spec["related_facts"][field],
                    style=style,
                    closing="A close identifier does not make these signed figures applicable to the target decision file.",
                )
            )
    random.Random(int(opaque(secret, f"{spec['slug']}:order", 16), 16)).shuffle(
        documents
    )
    governance = (
        f"Decision file {target_id}. A signed current {pair.domain} rule superseded an archived rule; "
        "apply the current version to this file. Each signed record below has its own case scope and filing day. "
        "Related identifiers do not transfer facts across cases.\n"
        f"Current signed rule: {pair.current_text}\nArchived rule: {pair.archived_text}"
    )
    state = (
        governance
        + "\n\n"
        + spec["scene"]
        + "\n\n"
        + "\n\n".join(doc["body"] for doc in documents)
    )
    parsed = _read_visible(state, pair)
    first, second, sources = _resolve_worlds(
        parsed, target_id, pair, disputed=disputed_field
    )
    if first != facts or second != (alternative if alternative is not None else facts):
        raise ValueError("Visible independent parser differs from authored worlds")
    if evaluate_current(pair, first) != first_out or (
        alternative is not None and evaluate_current(pair, second) != second_out
    ):
        raise ValueError("Visible-world evaluator differs from structured proof")
    if governing_reference(state, pair.id, first) != first_out or evaluate_reference(
        pair.id, second
    ) != (second_out if alternative is not None else first_out):
        raise ValueError("Independent visible-world semantic oracle disagrees")
    if kind == "rule_precedence":
        archived = evaluate_archived(pair, first)
        if (
            archived == first_out
            or evaluate_reference(pair.id, first, archived=True) != archived
            or governing_reference(swap_priority(state, pair.id), pair.id, first)
            != archived
        ):
            raise ValueError("Archived policy priority perturbation did not flip")
    selected_ids = set(sources.values())
    if len(selected_ids) != 3:
        raise ValueError("Three separate target fact documents are required")
    for document in parsed:
        if document["doc_id"] in selected_ids:
            others = [row for row in parsed if row["doc_id"] != document["doc_id"]]
            if (
                len({row["field"] for row in others if row["case_id"] == target_id})
                == 3
            ):
                raise ValueError("A purported required document is redundant")
    if kind == "near_distractor":
        without_decoy = [row for row in parsed if row["case_id"] != decoy_id]
        remaining, _, _ = _resolve_worlds(without_decoy, target_id, pair, disputed=None)
        decoy, _, _ = _resolve_worlds(parsed, decoy_id, pair, disputed=None)
        if (
            remaining != facts
            or decoy != spec["related_facts"]
            or evaluate_current(pair, decoy) == answer
        ):
            raise ValueError("Near-case perturbation invariant failed")
    item_id = opaque(secret, f"{spec['slug']}:item", 16)
    prompt = {
        "id": item_id,
        "state": state,
        "questions": {"decision": _question(pair, facts, target_id)},
    }
    target = {
        "id": item_id,
        "kind": pair.kind,
        "answer": _answer(pair, answer),
        "source_group": opaque(secret, f"{spec['slug']}:group", 16),
    }
    trace = {
        "id": item_id,
        "pair_id": pair.id,
        "kind": pair.kind,
        "challenge": kind,
        "style": style,
        "first_world": first,
        "second_world": second if kind == "partial_evidence" else None,
        "first_output": first_out,
        "second_output": second_out,
        "archived_output": evaluate_archived(pair, first),
        "priority_swap_output": (
            evaluate_archived(pair, first) if kind == "rule_precedence" else None
        ),
        "answer": answer,
        "selected_sources": sources,
        "decoy_output": (
            evaluate_current(pair, spec["related_facts"])
            if kind == "near_distractor"
            else None
        ),
        "partial_invariant_nonfallback": (
            resolved if kind == "partial_evidence" else None
        ),
    }
    return prompt, target, trace


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def build(
    *, dossier_dir: Path, short_specs: Path, private_salt: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    secret = private_salt.read_bytes()
    specs = json.loads(short_specs.read_text(encoding="utf-8"))
    if (
        len(secret) < 32
        or len(specs) != 40
        or len({row["slug"] for row in specs}) != 40
    ):
        raise ValueError("DEV60 needs forty distinct short source scenarios")
    long_prompts = _read_jsonl(dossier_dir / "prompts.jsonl")
    long_targets = _read_jsonl(dossier_dir / "private/targets.jsonl")
    long_traces = _read_jsonl(dossier_dir / "private/proof_traces.jsonl")
    if not (len(long_prompts) == len(long_targets) == len(long_traces) == 20):
        raise ValueError("Long pilot packet is not complete")
    if any(
        prompt["id"] != target["id"] or prompt["id"] != trace["id"]
        for prompt, target, trace in zip(long_prompts, long_targets, long_traces)
    ):
        raise ValueError("Long pilot prompt, key, and proof rows are not aligned")
    prompts, targets, traces = list(long_prompts), list(long_targets), list(long_traces)
    failures = []
    for spec in specs:
        try:
            prompt, target, trace = build_short(spec, secret)
            prompts.append(prompt)
            targets.append(target)
            traces.append(trace)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                {
                    "slug_sha256": sha_bytes(spec.get("slug", "").encode()),
                    "reason": str(exc),
                }
            )
    if len({row["id"] for row in prompts}) != len(prompts):
        raise ValueError("Opaque item ID collision")
    order = sorted(
        range(len(prompts)),
        key=lambda index: opaque(secret, f"dev60-order:{prompts[index]['id']}", 64),
    )
    prompts = [prompts[index] for index in order]
    targets = [targets[index] for index in order]
    traces = [traces[index] for index in order]
    by_type = Counter(target["kind"] for target in targets)
    by_challenge = Counter(trace.get("challenge", "long_context") for trace in traces)
    noul_by_challenge = {}
    for challenge in by_challenge:
        selected = [
            target
            for target, trace in zip(targets, traces)
            if target["kind"] == "noul"
            and trace.get("challenge", "long_context") == challenge
        ]
        noul_by_challenge[challenge] = {
            "true": sum(target["answer"]["noul"] is True for target in selected),
            "false": sum(target["answer"]["noul"] is False for target in selected),
        }
    partial = [
        trace for trace in traces if trace.get("challenge") == "partial_evidence"
    ]
    partial_by_type = {
        kind: {
            "invariant_nonfallback": sum(
                trace["partial_invariant_nonfallback"] is True
                for trace in partial
                if trace["kind"] == kind
            ),
            "unresolved": sum(
                trace["partial_invariant_nonfallback"] is False
                for trace in partial
                if trace["kind"] == kind
            ),
        }
        for kind in ("choice", "noul", "score")
    }
    gate = (
        not failures
        and len(prompts) == 60
        and by_type == {"choice": 20, "noul": 20, "score": 20}
        and by_challenge
        == {
            "long_context": 20,
            "rule_precedence": 12,
            "partial_evidence": 14,
            "near_distractor": 14,
        }
        and all(
            0.4 <= counts["true"] / (counts["true"] + counts["false"]) <= 0.6
            for counts in noul_by_challenge.values()
        )
        and all(
            counts["invariant_nonfallback"] >= 1 and counts["unresolved"] >= 1
            for counts in partial_by_type.values()
        )
    )
    output_dir.mkdir(parents=True)
    private = output_dir / "private"
    private.mkdir(mode=0o700)
    for path, rows in (
        (output_dir / "prompts.jsonl", prompts),
        (private / "targets.jsonl", targets),
        (private / "proof_traces.jsonl", traces),
    ):
        path.write_bytes(b"".join(compact(row) for row in rows))
        path.chmod(0o600)
    audit = {
        "build_version": VERSION,
        "status": "AUTOMATED_PROOF_ONLY" if gate else "BLOCKED",
        "human_editor_approved": False,
        "spec_sha256": sha_bytes(short_specs.read_bytes()),
        "dossier_prompts_sha256": sha_bytes(
            (dossier_dir / "prompts.jsonl").read_bytes()
        ),
        "prompts_sha256": sha_bytes((output_dir / "prompts.jsonl").read_bytes()),
        "targets_sha256": sha_bytes((private / "targets.jsonl").read_bytes()),
        "proof_traces_sha256": sha_bytes((private / "proof_traces.jsonl").read_bytes()),
        "accepted": len(prompts),
        "rejected": len(failures),
        "failures": failures,
        "by_type": dict(by_type),
        "by_challenge": dict(by_challenge),
        "noul_by_challenge": noul_by_challenge,
        "partial_by_type": partial_by_type,
        "style_by_challenge": {
            challenge: dict(
                Counter(
                    trace["style"]
                    for trace in traces
                    if trace.get("challenge", "long_context") == challenge
                )
            )
            for challenge in by_challenge
        },
    }
    (private / "audit.json").write_bytes(compact(audit))
    (private / "audit.json").chmod(0o600)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier-dir", type=Path, required=True)
    parser.add_argument("--short-specs", type=Path, required=True)
    parser.add_argument("--private-salt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        dossier_dir=args.dossier_dir,
        short_specs=args.short_specs,
        private_salt=args.private_salt,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "accepted",
                    "rejected",
                    "prompts_sha256",
                    "proof_traces_sha256",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
