"""Gold-separated semantic proof for a deliberately small v6 DEV pilot.

Only private authored specifications are read at runtime. The builder writes
gold-free prompts separately from private keys and proof traces. An automated
proof never qualifies an item for the release benchmark or substitutes for a
blind editorial review.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .authored_v5_dev import _read_visible, _resolve_worlds, _uncertain_line
from .authored_v5_dossier import (
    _document,
    _resolve,
    _visible_documents,
    compact,
    opaque,
    sha_bytes,
)
from .authored_v6_policies import (
    POLICIES,
    Policy,
    aggregate,
    evaluate,
    governing,
    reference,
    swap_priority,
    validate,
)

BUILD_VERSION = "jevarena-authored-v6-dev18-semantic-pilot/2"
STYLES = ("prose", "bullets", "table")


def _select_style(secret: bytes, slug: str) -> str:
    return STYLES[int(opaque(secret, f"v6:{slug}:style", 2), 16) % len(STYLES)]


def _question(
    policy: Policy, facts: dict[str, Any], case_id: str, secret: bytes
) -> dict[str, Any]:
    if policy.kind == "choice":
        criteria = {key: f"Registered option {key}" for key in facts[policy.fields[0]]}
        criteria["hold"] = "Hold if no option is eligible or admissible worlds disagree"
        criteria = dict(
            sorted(
                criteria.items(),
                key=lambda row: opaque(secret, f"v6:{case_id}:option:{row[0]}", 64),
            )
        )
    elif policy.kind == "noul":
        criteria = {
            "false": "The decision cannot be certified true",
            "true": "The decision is certified true",
        }
    else:
        criteria = [f"Grade {grade}" for grade in range(5)]
    return {
        "type": policy.kind,
        "instructions": (
            f"For decision file {case_id}, apply the current signed rule to the admissible "
            "source world or worlds and return the resulting decision."
        ),
        "criteria": criteria,
    }


def _answer(policy: Policy, value: str | bool | int) -> dict[str, Any]:
    key = {"choice": "choice", "noul": "noul", "score": "score"}[policy.kind]
    return {"type": policy.kind, key: value}


def _governance(policy: Policy, case_id: str) -> str:
    return (
        f"Decision file {case_id}. The governance office signed a current {policy.domain} "
        "rule after archiving the previous version. Apply the current signed rule to this "
        "file only. Later signed case-specific facts supersede earlier facts for the same "
        "field; similarly numbered cases are separate. An explicitly disputed signed "
        "field retains all admissible source worlds until reconciled.\n"
        f"Current signed rule: {policy.current_text}\n"
        f"Archived rule: {policy.archived_text}"
    )


def _check_oracles(
    policy: Policy, facts: dict[str, Any]
) -> tuple[str | bool | int, str | bool | int]:
    current = evaluate(policy, facts)
    archived = evaluate(policy, facts, archived=True)
    if current != reference(policy, facts) or archived != reference(
        policy, facts, archived=True
    ):
        raise ValueError("Primary and reference policy oracles disagree")
    return current, archived


def _check_counterfactuals(
    policy: Policy, facts: dict[str, Any], authored: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Actual answer sensitivity, not a parser's missing-field result."""
    if set(authored) != set(policy.fields):
        raise ValueError("Every target field needs an authored value intervention")
    original = evaluate(policy, facts)
    receipts = {}
    for field in policy.fields:
        row = authored[field]
        if (
            not isinstance(row.get("rationale"), str)
            or len(row["rationale"].split()) < 12
        ):
            raise ValueError("Counterfactual needs a concrete editorial rationale")
        if row["value"] == facts[field]:
            raise ValueError("Counterfactual did not alter its source field")
        alternate = {**facts, field: row["value"]}
        validate(policy, alternate)
        changed = evaluate(policy, alternate)
        if changed != reference(policy, alternate) or changed == original:
            raise ValueError(
                "Target document is not causally necessary for the current decision"
            )
        receipts[field] = {
            "original_output": original,
            "intervention_output": changed,
            "intervention_value": row["value"],
            "rationale": row["rationale"],
        }
    return receipts


def _long_document(
    secret: bytes,
    slug: str,
    serial: int,
    case_id: str,
    day: int,
    field: str,
    value: Any,
    style: str,
    context: dict[str, str],
) -> dict[str, Any]:
    if len(context["intro"].split()) < 25 or len(context["closing"].split()) < 25:
        raise ValueError("Long dossier source has insufficient substantive prose")
    return _document(
        secret=secret,
        slug=f"v6:{slug}",
        serial=serial,
        case_id=case_id,
        day=day,
        signer=context["signer"],
        genre=context["genre"],
        intro=context["intro"],
        field=field,
        value=value,
        style=style,
        closing=context["closing"],
    )


def build_long(
    spec: dict[str, Any], secret: bytes
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    policy = POLICIES[spec["policy_id"]]
    facts = spec["facts"]
    validate(policy, facts)
    current, archived = _check_oracles(policy, facts)
    if current == archived:
        raise ValueError("Long dossier lacks executable policy-version conflict")
    interventions = _check_counterfactuals(policy, facts, spec["counterfactuals"])
    stale_field = spec["stale_field"]
    if stale_field not in policy.fields or spec["stale_value"] == facts[stale_field]:
        raise ValueError("No genuine stale target evidence")
    stale = {**facts, stale_field: spec["stale_value"]}
    validate(policy, stale)
    if evaluate(policy, stale) == current:
        raise ValueError("Stale signed value does not compete with current answer")
    related = spec["related_facts"]
    validate(policy, related)
    if evaluate(policy, related) == current:
        raise ValueError("Related case does not compete with current answer")
    for section in spec["narrative_sections"]:
        if len(section.split()) < 70:
            raise ValueError("Long dossier has a thin narrative section")
    if len(spec["narrative_sections"]) < 6:
        raise ValueError("Long dossier needs varied, task-bearing narrative sections")
    slug = spec["slug"]
    case_id = opaque(secret, f"v6:{slug}:target", 12)
    related_id = case_id[:-1] + ("0" if case_id[-1] != "0" else "1")
    style = _select_style(secret, slug)
    contexts = spec["source_contexts"]
    if set(contexts) != set(policy.fields):
        raise ValueError("Each target source needs bespoke editorial context")
    documents = [
        _long_document(
            secret,
            slug,
            serial,
            case_id,
            20 + serial,
            field,
            facts[field],
            style,
            contexts[field],
        )
        for serial, field in enumerate(policy.fields)
    ]
    documents.append(
        _long_document(
            secret,
            slug,
            10,
            case_id,
            8,
            stale_field,
            spec["stale_value"],
            style,
            spec["stale_context"],
        )
    )
    related_contexts = spec["related_contexts"]
    if set(related_contexts) != set(policy.fields):
        raise ValueError("Related case needs three source contexts")
    documents.extend(
        _long_document(
            secret,
            slug,
            20 + serial,
            related_id,
            21 + serial,
            field,
            related[field],
            style,
            related_contexts[field],
        )
        for serial, field in enumerate(policy.fields)
    )
    random.Random(int(opaque(secret, f"v6:{slug}:order", 16), 16)).shuffle(documents)
    state = (
        _governance(policy, case_id) + "\n\n" + "\n\n".join(spec["narrative_sections"])
    )
    state += "\n\n" + "\n\n".join(document["body"] for document in documents)
    words = len(state.split())
    if not 1000 <= words <= 3500:
        raise ValueError("Pilot dossier misses its substantive length window")
    visible = _visible_documents(state, policy)
    selected, sources = _resolve(visible, case_id, policy)
    if (
        selected != facts
        or set(sources) != set(policy.fields)
        or len(set(sources.values())) != 3
    ):
        raise ValueError("Visible reader failed to reconstruct three target facts")
    if (
        governing(state, policy, selected) != current
        or governing(swap_priority(state, policy), policy, selected) != archived
    ):
        raise ValueError("Visible policy-priority intervention failed")
    related_visible, _ = _resolve(visible, related_id, policy)
    if related_visible != related or evaluate(policy, related_visible) == current:
        raise ValueError("Visible related-case identity swap failed")
    without_related, _ = _resolve(
        [row for row in visible if row["case_id"] != related_id], case_id, policy
    )
    if without_related != facts or evaluate(policy, without_related) != current:
        raise ValueError("Related-case deletion changed the target decision")
    # The source deletion result is retained as context, not used as causal proof.
    deletion_outcomes = {}
    for field, doc_id in sources.items():
        reduced, _ = _resolve(
            [row for row in visible if row["doc_id"] != doc_id], case_id, policy
        )
        deletion_outcomes[field] = (
            "INCOMPLETE"
            if set(reduced) != set(policy.fields)
            else evaluate(policy, reduced)
        )
    item_id = opaque(secret, f"v6:{slug}:item", 16)
    prompt = {
        "id": item_id,
        "state": state,
        "questions": {"decision": _question(policy, facts, case_id, secret)},
    }
    target = {
        "id": item_id,
        "kind": policy.kind,
        "answer": _answer(policy, current),
        "source_group": opaque(secret, f"v6:{slug}:source-group", 16),
    }
    trace = {
        "id": item_id,
        "policy_id": policy.id,
        "challenge": "long_dossier",
        "kind": policy.kind,
        "style": style,
        "visible_words": words,
        "selected_facts": selected,
        "selected_sources": sources,
        "current_output": current,
        "archived_output": archived,
        "priority_swap_output": archived,
        "causal_value_interventions": interventions,
        "source_deletion_context_only": deletion_outcomes,
        "stale_output": evaluate(policy, stale),
        "related_output": evaluate(policy, related_visible),
        "decoy_drop_invariant": True,
    }
    return prompt, target, trace


def build_short(
    spec: dict[str, Any], secret: bytes
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    policy = POLICIES[spec["policy_id"]]
    challenge = spec["challenge"]
    if challenge not in {"rule_precedence", "partial_evidence"}:
        raise ValueError("Unknown v6 short challenge")
    facts = spec["facts"]
    validate(policy, facts)
    if len(spec["scene"].split()) < 55 or len(spec["source_notes"]) != 3:
        raise ValueError("Short case needs readable authored context")
    current, archived = _check_oracles(policy, facts)
    if challenge == "rule_precedence" and current == archived:
        raise ValueError("Policy-priority case has no answer conflict")
    field = spec.get("disputed_field") if challenge == "partial_evidence" else None
    if challenge == "partial_evidence":
        if field not in policy.fields or spec["alternative_value"] == facts[field]:
            raise ValueError("No material disputed field")
        alternate = {**facts, field: spec["alternative_value"]}
        validate(policy, alternate)
        second = evaluate(policy, alternate)
        if second != reference(policy, alternate):
            raise ValueError("Independent second-world oracle disagrees")
        if (second == current) != (spec["world_relation"] == "invariant"):
            raise ValueError("Declared world relation is false")
        if spec["world_relation"] not in {"invariant", "sensitive"}:
            raise ValueError("Unknown world relation")
        if spec["world_relation"] == "invariant" and (
            (policy.kind == "choice" and current == "hold")
            or (policy.kind == "noul" and current is not True)
            or (policy.kind == "score" and current == 0)
        ):
            raise ValueError("Invariant case must have a nondefault decision")
        worlds = [current, second]
    else:
        alternate = None
        worlds = [current]
    answer = aggregate(policy, worlds)
    slug = spec["slug"]
    case_id = opaque(secret, f"v6:{slug}:target", 12)
    style = _select_style(secret, slug)
    documents = []
    for serial, source_field in enumerate(policy.fields):
        context = spec["source_notes"][serial]
        doc = _document(
            secret=secret,
            slug=f"v6:{slug}",
            serial=serial,
            case_id=case_id,
            day=20 + serial,
            signer=("Mara Ellis", "Noah Chen", "Amina Patel")[serial],
            genre=("signed ledger", "assessment letter", "scope certificate")[serial],
            intro=context["intro"],
            field=source_field,
            value=facts[source_field],
            style=style,
            closing=context["closing"],
        )
        if source_field == field:
            lines = doc["body"].splitlines()
            lines[2] = _uncertain_line(field, facts[field], alternate[field])
            doc["body"] = "\n".join(lines)
        documents.append(doc)
    random.Random(int(opaque(secret, f"v6:{slug}:order", 16), 16)).shuffle(documents)
    state = _governance(policy, case_id) + "\n\n" + spec["scene"]
    state += "\n\n" + "\n\n".join(document["body"] for document in documents)
    visible = _read_visible(state, policy)
    first, second_visible, sources = _resolve_worlds(
        visible, case_id, policy, disputed=field
    )
    if first != facts or second_visible != (
        alternate if alternate is not None else facts
    ):
        raise ValueError("Visible-world parser disagrees with authored source worlds")
    if (
        len(set(sources.values())) != 3
        or aggregate(
            policy,
            [reference(policy, first)]
            + ([reference(policy, second_visible)] if alternate is not None else []),
        )
        != answer
    ):
        raise ValueError("Independent visible-world aggregation disagrees")
    if governing(state, policy, first) != current:
        raise ValueError("Visible current policy cannot be executed")
    if (
        challenge == "rule_precedence"
        and governing(swap_priority(state, policy), policy, first) != archived
    ):
        raise ValueError("Visible priority swap did not flip")
    item_id = opaque(secret, f"v6:{slug}:item", 16)
    prompt = {
        "id": item_id,
        "state": state,
        "questions": {"decision": _question(policy, facts, case_id, secret)},
    }
    target = {
        "id": item_id,
        "kind": policy.kind,
        "answer": _answer(policy, answer),
        "source_group": opaque(secret, f"v6:{slug}:source-group", 16),
    }
    trace = {
        "id": item_id,
        "policy_id": policy.id,
        "challenge": challenge,
        "kind": policy.kind,
        "style": style,
        "visible_words": len(state.split()),
        "first_world": first,
        "second_world": second_visible if alternate is not None else None,
        "first_output": current,
        "second_output": worlds[1] if len(worlds) == 2 else None,
        "archived_output": archived,
        "answer": answer,
        "priority_swap_output": archived if challenge == "rule_precedence" else None,
        "world_relation": spec.get("world_relation"),
        "selected_sources": sources,
    }
    return prompt, target, trace


def build(specs_path: Path, salt_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    specs = json.loads(specs_path.read_text())
    secret = salt_path.read_bytes()
    if (
        len(secret) < 32
        or len(specs) != 18
        or len({row["slug"] for row in specs}) != 18
    ):
        raise ValueError(
            "v6 DEV pilot needs 18 distinct private authored source scenarios"
        )
    prompts, targets, traces, failures = [], [], [], []
    for spec in specs:
        try:
            generated = (
                build_long(spec, secret)
                if spec["challenge"] == "long_dossier"
                else build_short(spec, secret)
            )
            prompt, target, trace = generated
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
        raise ValueError("Opaque v6 item ID collision")
    order = sorted(
        range(len(prompts)),
        key=lambda index: opaque(secret, f"v6:order:{prompts[index]['id']}", 64),
    )
    prompts, targets, traces = (
        [rows[index] for index in order] for rows in (prompts, targets, traces)
    )
    by_type = Counter(target["kind"] for target in targets)
    by_challenge = Counter(trace["challenge"] for trace in traces)
    by_pair = Counter(trace["policy_id"] for trace in traces)
    partial_by_type = {
        kind: dict(
            Counter(
                trace["world_relation"]
                for trace in traces
                if trace["challenge"] == "partial_evidence" and trace["kind"] == kind
            )
        )
        for kind in ("choice", "noul", "score")
    }
    noul_by_challenge = {
        challenge: dict(
            Counter(
                str(target["answer"]["noul"])
                for target, trace in zip(targets, traces)
                if target["kind"] == "noul" and trace["challenge"] == challenge
            )
        )
        for challenge in ("rule_precedence", "partial_evidence", "long_dossier")
    }
    style_by_challenge = {
        challenge: dict(
            Counter(
                trace["style"] for trace in traces if trace["challenge"] == challenge
            )
        )
        for challenge in by_challenge
    }
    resolved_choice_position = Counter(
        str(
            list(prompt["questions"]["decision"]["criteria"]).index(
                target["answer"]["choice"]
            )
        )
        for prompt, target in zip(prompts, targets)
        if target["kind"] == "choice" and target["answer"]["choice"] != "hold"
    )
    choice_position_gate = (
        sum(resolved_choice_position.values()) == 5
        and len(resolved_choice_position) >= 2
        and max(resolved_choice_position.values(), default=0) <= 3
    )
    metadata_clean = all(
        re.fullmatch(r"[0-9a-f]{16}", prompt["id"])
        and set(prompt) == {"id", "state", "questions"}
        and set(prompt["questions"]) == {"decision"}
        and "source_group" not in json.dumps(prompt)
        for prompt in prompts
    )
    gate = (
        not failures
        and len(prompts) == 18
        and by_type == {"choice": 6, "noul": 6, "score": 6}
        and by_challenge
        == {"rule_precedence": 6, "partial_evidence": 6, "long_dossier": 6}
        and set(by_pair) == set(POLICIES)
        and all(count == 3 for count in by_pair.values())
        and all(
            counts == {"invariant": 1, "sensitive": 1}
            for counts in partial_by_type.values()
        )
        and all(
            counts == {"True": 1, "False": 1} for counts in noul_by_challenge.values()
        )
        and metadata_clean
        and choice_position_gate
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
        "build_version": BUILD_VERSION,
        "status": "AUTOMATED_PROOF_ONLY" if gate else "BLOCKED",
        "human_editor_approved": False,
        "independent_blind_review_passed": False,
        "accepted": len(prompts),
        "rejected": len(failures),
        "failures": failures,
        "by_type": dict(by_type),
        "by_challenge": dict(by_challenge),
        "by_policy": dict(by_pair),
        "partial_by_type": partial_by_type,
        "noul_by_challenge": noul_by_challenge,
        "style_by_challenge": style_by_challenge,
        "resolved_choice_option_index": dict(resolved_choice_position),
        "choice_position_gate": choice_position_gate,
        "opaque_metadata_clean": metadata_clean,
        "dossier_word_min": min(
            (
                trace["visible_words"]
                for trace in traces
                if trace["challenge"] == "long_dossier"
            ),
            default=0,
        ),
        "dossier_word_max": max(
            (
                trace["visible_words"]
                for trace in traces
                if trace["challenge"] == "long_dossier"
            ),
            default=0,
        ),
        "spec_sha256": sha_bytes(specs_path.read_bytes()),
        "salt_commitment_sha256": sha_bytes(secret),
        "prompts_sha256": sha_bytes((output_dir / "prompts.jsonl").read_bytes()),
        "targets_sha256": sha_bytes((private / "targets.jsonl").read_bytes()),
        "proof_traces_sha256": sha_bytes((private / "proof_traces.jsonl").read_bytes()),
    }
    (private / "audit.json").write_bytes(compact(audit))
    (private / "audit.json").chmod(0o600)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--private-salt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.specs, args.private_salt, args.output_dir)
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
                    "dossier_word_min",
                    "dossier_word_max",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
