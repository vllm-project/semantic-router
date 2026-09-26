"""Gold-separated semantic proof builder for v5 hand-authored dossiers.

This feasibility builder reads private authored scenario facts at runtime. It
does not embed candidate facts, keys or labels in the public implementation.
The independent text parser deliberately does not read those private facts.
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

BUILD_VERSION = "jevarena-authored-v5-semantic-proof/1"
DOCUMENT_RE = re.compile(
    r"^Record (?P<doc>[0-9a-f]{16}) \| case (?P<case>[0-9a-f]{12}) \| filed day (?P<day>\d+) \| signed by (?P<signer>[A-Za-z ]+) \| (?P<genre>[A-Za-z ]+)\.$"
)
PROSE_RE = re.compile(r"^The certified (?P<label>[a-z -]+) is (?P<value>.+)\.$")
BULLET_RE = re.compile(r"^- Certified (?P<label>[a-z -]+): (?P<value>.+)$")
TABLE_RE = re.compile(r"^\| Certified (?P<label>[a-z -]+) \| (?P<value>.+) \|$")
FIELD_LABELS = {
    "costs": "total cost register",
    "benefits": "service benefit register",
    "qualified_ids": "qualified bid list",
    "lead_days": "delivery lead days",
    "reliability": "reliability scores",
    "approved_ids": "approved plan list",
    "community_impact": "community impact scores",
    "cofund_pct": "committed co funding percentages",
    "eligible_ids": "eligible proposal list",
    "exposure_reduced": "measured exposure reductions",
    "start_hours": "response start hours",
    "cleared_ids": "cleared response list",
    "signed_votes": "signed vote count",
    "quorum": "required quorum",
    "committee_size": "committee seat count",
    "remaining_stock": "remaining stock units",
    "reserve_min": "minimum reserve units",
    "demand_next_day": "signed next day demand units",
    "expense": "claimed expense units",
    "approved_budget": "approved budget units",
    "receipt_verified": "receipt verification status",
    "severity": "incident severity level",
    "mitigation_signed": "mitigation sign off status",
    "incident_signed": "incident intake sign off status",
    "passed": "passed control count",
    "reviewed": "reviewed control count",
    "audit_signed": "audit sign off status",
    "late_days": "late day count",
    "deadline_signed": "deadline sign off status",
    "closure_signed": "closure sign off status",
    "likelihood": "likelihood level",
    "impact": "impact level",
    "assessment_signed": "assessment sign off status",
    "passed_checks": "passed inspection count",
    "critical_failures": "critical failure count",
    "review_signed": "review sign off status",
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def opaque(secret: bytes, label: str, length: int) -> str:
    return sha_bytes(secret + label.encode())[:length]


def _value_text(value: Any) -> str:
    if type(value) is bool:
        return "yes" if value else "no"
    if type(value) is int:
        return str(value)
    if isinstance(value, dict):
        return "; ".join(f"{key}: {number}" for key, number in value.items())
    if isinstance(value, list):
        return ", ".join(value)
    raise ValueError("Unsupported fact representation")


DICT_FIELDS = {
    "costs",
    "benefits",
    "lead_days",
    "reliability",
    "community_impact",
    "cofund_pct",
    "exposure_reduced",
    "start_hours",
}
LIST_FIELDS = {"qualified_ids", "approved_ids", "eligible_ids", "cleared_ids"}
BOOL_FIELDS = {
    "receipt_verified",
    "mitigation_signed",
    "incident_signed",
    "audit_signed",
    "deadline_signed",
    "closure_signed",
    "assessment_signed",
    "review_signed",
}


def _parse_value(field: str, text: str) -> Any:
    if field in BOOL_FIELDS:
        if text not in {"yes", "no"}:
            raise ValueError(f"Invalid visible Boolean for {field}")
        return text == "yes"
    if field in DICT_FIELDS:
        pieces = text.split("; ")
        parsed = {}
        for piece in pieces:
            matched = re.fullmatch(r"([A-Za-z][A-Za-z0-9-]*): (\d+)", piece)
            if matched is None or matched[1] in parsed:
                raise ValueError(f"Invalid visible register for {field}")
            parsed[matched[1]] = int(matched[2])
        return parsed
    if field in LIST_FIELDS:
        parsed = text.split(", ")
        if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", key) for key in parsed):
            raise ValueError(f"Invalid visible list for {field}")
        return parsed
    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"Invalid visible number for {field}")
    return int(text)


def _render_fact(field: str, value: Any, style: str) -> str:
    label = FIELD_LABELS[field]
    payload = _value_text(value)
    if style == "prose":
        return f"The certified {label} is {payload}."
    if style == "bullets":
        return f"- Certified {label}: {payload}"
    if style == "table":
        return f"| Certified {label} | {payload} |"
    raise ValueError("Unknown document style")


def _parse_fact(line: str, pair: PolicyPair) -> tuple[str, Any]:
    matched = next(
        (
            pattern.fullmatch(line)
            for pattern in (PROSE_RE, BULLET_RE, TABLE_RE)
            if pattern.fullmatch(line)
        ),
        None,
    )
    if matched is None:
        raise ValueError("Visible document has no parseable certified fact")
    reversed_labels = {FIELD_LABELS[field]: field for field in pair.fields}
    label = matched["label"]
    if label not in reversed_labels:
        raise ValueError("Visible field label does not belong to policy schema")
    field = reversed_labels[label]
    return field, _parse_value(field, matched["value"])


def _document(
    *,
    secret: bytes,
    slug: str,
    serial: int,
    case_id: str,
    day: int,
    signer: str,
    genre: str,
    intro: str,
    field: str,
    value: Any,
    style: str,
    closing: str,
) -> dict[str, Any]:
    if "\n" in intro or "\n" in closing or "\n" in signer or "\n" in genre:
        raise ValueError("Private narratives must be single paragraphs")
    doc_id = opaque(secret, f"{slug}:document:{serial}", 16)
    header = f"Record {doc_id} | case {case_id} | filed day {day} | signed by {signer} | {genre}."
    body = "\n".join((header, intro, _render_fact(field, value, style), closing))
    return {
        "doc_id": doc_id,
        "case_id": case_id,
        "filed_day": day,
        "field": field,
        "value": value,
        "body": body,
    }


def _visible_documents(state: str, pair: PolicyPair) -> list[dict[str, Any]]:
    sections = state.split("\n\n")
    documents = []
    for section in sections:
        lines = section.splitlines()
        matched = DOCUMENT_RE.fullmatch(lines[0]) if lines else None
        if matched is None:
            continue
        if len(lines) != 4 or not lines[1] or not lines[3]:
            raise ValueError("Visible document lacks its narrative or fact")
        field, value = _parse_fact(lines[2], pair)
        documents.append(
            {
                "doc_id": matched["doc"],
                "case_id": matched["case"],
                "filed_day": int(matched["day"]),
                "signer": matched["signer"],
                "genre": matched["genre"],
                "field": field,
                "value": value,
                "body": section,
            }
        )
    if len(documents) < 7 or len({item["doc_id"] for item in documents}) != len(
        documents
    ):
        raise ValueError("Dossier lacks distinct, distributed source documents")
    return documents


def _resolve(
    documents: list[dict[str, Any]], case_id: str, pair: PolicyPair
) -> tuple[dict[str, Any], dict[str, str]]:
    chosen: dict[str, dict[str, Any]] = {}
    for document in documents:
        if document["case_id"] != case_id:
            continue
        field = document["field"]
        if field not in pair.fields:
            raise ValueError("Document field is outside policy schema")
        old = chosen.get(field)
        if old is None or document["filed_day"] > old["filed_day"]:
            chosen[field] = document
        elif document["filed_day"] == old["filed_day"]:
            raise ValueError("Ambiguous same-day signed target facts")
    return (
        {field: chosen[field]["value"] for field in pair.fields if field in chosen},
        {field: chosen[field]["doc_id"] for field in pair.fields if field in chosen},
    )


def _output_or_unresolved(pair: PolicyPair, facts: dict[str, Any]) -> str | bool | int:
    if set(facts) != set(pair.fields):
        return "UNRESOLVED"
    return evaluate_current(pair, facts)


def build_one(
    spec: dict[str, Any], secret: bytes
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pair = PAIRS[spec["pair_id"]]
    facts = spec["facts"]
    validate_facts(pair, facts)
    pair_proof = prove_conflict(pair, facts)
    stale_field = spec["stale_field"]
    if stale_field not in pair.fields or spec["stale_value"] == facts[stale_field]:
        raise ValueError("Stale target fact is not a real conflict")
    stale_facts = {**facts, stale_field: spec["stale_value"]}
    validate_facts(pair, stale_facts)
    decoy_facts = spec["related_facts"]
    validate_facts(pair, decoy_facts)
    if evaluate_current(pair, decoy_facts) == pair_proof["current_output"]:
        raise ValueError("Related-case identity swap would not change the answer")
    slug = spec["slug"]
    target_id = opaque(secret, f"{slug}:target", 12)
    related_id = target_id[:-1] + ("0" if target_id[-1] != "0" else "1")
    if target_id == related_id:
        raise AssertionError("Related case must be different")
    style = ("prose", "bullets", "table")[
        int(opaque(secret, f"{slug}:style", 2), 16) % 3
    ]
    current_docs = []
    for position, field in enumerate(pair.fields):
        context = spec["field_contexts"][field]
        current_docs.append(
            _document(
                secret=secret,
                slug=slug,
                serial=position,
                case_id=target_id,
                day=20 + position,
                signer=context["signer"],
                genre=context["genre"],
                intro=context["intro"],
                field=field,
                value=facts[field],
                style=style,
                closing=context["closing"],
            )
        )
    stale_context = spec["stale_context"]
    stale_doc = _document(
        secret=secret,
        slug=slug,
        serial=10,
        case_id=target_id,
        day=8,
        signer=stale_context["signer"],
        genre=stale_context["genre"],
        intro=stale_context["intro"],
        field=stale_field,
        value=spec["stale_value"],
        style=style,
        closing=stale_context["closing"],
    )
    related_docs = []
    related_context = spec["related_context"]
    for position, field in enumerate(pair.fields):
        related_docs.append(
            _document(
                secret=secret,
                slug=slug,
                serial=20 + position,
                case_id=related_id,
                day=21 + position,
                signer=related_context["signer"],
                genre=related_context["genre"],
                intro=related_context["intro"],
                field=field,
                value=decoy_facts[field],
                style=style,
                closing=related_context["closing"],
            )
        )
    documents = current_docs + [stale_doc] + related_docs
    random.Random(int(opaque(secret, f"{slug}:position", 16), 16)).shuffle(documents)
    governance = (
        f"Decision file {target_id}. The governance office signed the current {pair.domain} rule on day 45; "
        f"an archived rule signed on day 11 remains in the record for historical comparison. "
        "Apply the current signed rule to this file only. Later signed case-specific records supersede older records "
        "for the same field; similarly named case files are separate.\n"
        f"Current signed rule: {pair.current_text}\nArchived rule: {pair.archived_text}"
    )
    for section in ("background", "timeline"):
        if not isinstance(spec.get(section), str) or len(spec[section].split()) < 35:
            raise ValueError(f"Hand-authored {section} is too thin")
    state = (
        governance
        + "\n\n"
        + spec["background"]
        + "\n\n"
        + spec["timeline"]
        + "\n\n"
        + "\n\n".join(document["body"] for document in documents)
    )
    if len(state.split()) < 450:
        raise ValueError("Natural dossier does not reach the pilot length gate")
    parsed = _visible_documents(state, pair)
    rendered_facts, sources = _resolve(parsed, target_id, pair)
    if (
        rendered_facts != facts
        or evaluate_current(pair, rendered_facts) != pair_proof["current_output"]
        or governing_reference(state, pair.id, rendered_facts)
        != pair_proof["current_output"]
    ):
        raise ValueError("Independent visible-text interpreter disagrees")
    if (
        evaluate_archived(pair, rendered_facts) != pair_proof["archived_output"]
        or evaluate_reference(pair.id, rendered_facts, archived=True)
        != pair_proof["archived_output"]
        or governing_reference(swap_priority(state, pair.id), pair.id, rendered_facts)
        != pair_proof["archived_output"]
    ):
        raise ValueError("Archived rule is not executable from visible evidence")
    selected_docs = [item for item in parsed if item["doc_id"] in set(sources.values())]
    if (
        len(selected_docs) != len(pair.fields)
        or len({item["doc_id"] for item in selected_docs}) < 3
    ):
        raise ValueError("Fewer than three necessary target documents")
    ablations = {}
    for document in selected_docs:
        remaining = [item for item in parsed if item["doc_id"] != document["doc_id"]]
        ablated_facts, _ = _resolve(remaining, target_id, pair)
        after = _output_or_unresolved(pair, ablated_facts)
        if after == pair_proof["current_output"]:
            raise ValueError("A required target document is causally redundant")
        ablations[document["doc_id"]] = after
    unrelated = [item for item in parsed if item["case_id"] != related_id]
    without_decoy, _ = _resolve(unrelated, target_id, pair)
    if (
        without_decoy != facts
        or evaluate_current(pair, without_decoy) != pair_proof["current_output"]
    ):
        raise ValueError("Removing a distractor changed the target answer")
    related_visible, _ = _resolve(parsed, related_id, pair)
    if (
        related_visible != decoy_facts
        or evaluate_current(pair, related_visible) == pair_proof["current_output"]
    ):
        raise ValueError("Identity swap failed on visible text")
    answer = pair_proof["current_output"]
    if pair.kind == "choice":
        criteria = {
            key: f"Select registered option {key}" for key in facts[pair.fields[0]]
        }
        criteria["hold"] = "Hold because the governing evidence is unresolved"
        response = {"type": "choice", "choice": answer}
    elif pair.kind == "noul":
        criteria = {
            "false": "The current decision condition does not hold",
            "true": "The current decision condition holds",
        }
        response = {"type": "noul", "noul": answer}
    else:
        criteria = [f"Grade {level}" for level in range(5)]
        response = {"type": "score", "score": answer}
    item_id = opaque(secret, f"{slug}:item", 16)
    question = {
        "type": pair.kind,
        "instructions": f"For decision file {target_id}, apply the current signed rule and return its decision.",
        "criteria": criteria,
    }
    prompt = {"id": item_id, "state": state, "questions": {"decision": question}}
    target = {
        "id": item_id,
        "kind": pair.kind,
        "answer": response,
        "source_group": opaque(secret, f"{slug}:group", 16),
    }
    trace = {
        "id": item_id,
        "pair_id": pair.id,
        "source_group": target["source_group"],
        "case_id": target_id,
        "related_case_id": related_id,
        "style": style,
        "selected_facts": rendered_facts,
        "selected_sources": sources,
        "current_output": answer,
        "archived_output": pair_proof["archived_output"],
        "priority_swap_changes_output": True,
        "priority_swap_output": pair_proof["archived_output"],
        "required_document_ablations": ablations,
        "decoy_drop_invariant": True,
        "identity_swap_output": evaluate_current(pair, related_visible),
        "target_doc_count": len(current_docs) + 1,
        "related_doc_count": len(related_docs),
    }
    return prompt, target, trace


def build(specs_path: Path, secret_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    secret = secret_path.read_bytes()
    if len(secret) < 32 or not isinstance(specs, list) or len(specs) != 20:
        raise ValueError(
            "V5 feasibility requires twenty private authored dossiers and a private salt"
        )
    if len({spec["slug"] for spec in specs}) != 20:
        raise ValueError("Duplicate authored dossier slug")
    prompts, targets, traces = [], [], []
    failures = []
    for spec in specs:
        try:
            prompt, target, trace = build_one(spec, secret)
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
    output_dir.mkdir(parents=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(mode=0o700)
    for path, rows in (
        (output_dir / "prompts.jsonl", prompts),
        (private_dir / "targets.jsonl", targets),
        (private_dir / "proof_traces.jsonl", traces),
    ):
        path.write_bytes(b"".join(compact(row) for row in rows))
        path.chmod(0o600)
    audit = {
        "build_version": BUILD_VERSION,
        "status": (
            "BLOCKED" if failures or len(prompts) != 20 else "AUTOMATED_PROOF_ONLY"
        ),
        "human_editor_approved": False,
        "spec_sha256": sha_bytes(specs_path.read_bytes()),
        "salt_commitment_sha256": sha_bytes(secret),
        "prompts_sha256": sha_bytes((output_dir / "prompts.jsonl").read_bytes()),
        "targets_sha256": sha_bytes((private_dir / "targets.jsonl").read_bytes()),
        "proof_traces_sha256": sha_bytes(
            (private_dir / "proof_traces.jsonl").read_bytes()
        ),
        "accepted": len(prompts),
        "rejected": len(failures),
        "failures": failures,
        "by_type": dict(Counter(PAIRS[trace["pair_id"]].kind for trace in traces)),
        "by_pair": dict(Counter(trace["pair_id"] for trace in traces)),
        "styles": dict(Counter(trace["style"] for trace in traces)),
        "same_schema_executable_and_conflicting": all(
            trace["priority_swap_changes_output"] for trace in traces
        ),
        "three_target_documents_necessary": all(
            len(trace["required_document_ablations"]) >= 3 for trace in traces
        ),
        "decoy_drop_invariant": all(trace["decoy_drop_invariant"] for trace in traces),
    }
    (private_dir / "audit.json").write_bytes(compact(audit))
    (private_dir / "audit.json").chmod(0o600)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--private-salt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = build(args.specs, args.private_salt, args.output_dir)
    print(
        json.dumps(
            {
                key: audit[key]
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
