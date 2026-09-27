"""Private, deterministic English Score v7p case construction.

This produces candidate TRAIN or independent SELECT rows, never a release
benchmark. A separate QA/admission program must approve its output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import os
import random
from pathlib import Path
from typing import Any

from training.model.data import INPUT_FIELDS, digest, validate_row

VERSION = "decision2-score-v7p-evidence-state/1"
LEVELS = (0, 1, 2)
TRAIN_DOMAINS = (
    "municipal records transfer",
    "marine sample release",
    "heritage workshop access",
    "community radio permit",
)
SELECT_DOMAINS = (
    "observatory night access",
    "mobile health shipment",
    "rural bridge inspection",
    "botanical archive loan",
)
OPTIONS = (
    "An operative disqualifier is established",
    "No disqualifier is established, but a required condition remains unresolved",
    "No disqualifier is established and all required conditions are verified",
)


def _secret(path: Path) -> bytes:
    if path.stat().st_mode & 0o077:
        raise PermissionError("Seed file must be mode 0600")
    secret = path.read_bytes()
    if len(secret) != 32:
        raise ValueError("Need a new private 32-byte seed")
    return secret


def _rng(secret: bytes, role: str, index: int) -> random.Random:
    material = hmac.new(
        secret, f"{VERSION}\0{role}\0{index}".encode(), hashlib.sha256
    ).digest()
    return random.Random(int.from_bytes(material[:16], "big"))


def _id(rng: random.Random, prefix: str) -> str:
    return prefix + "-" + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=7))


def _base_facts(rng: random.Random) -> dict[str, Any]:
    """Keep entity, scope, date and document identity fixed in a triplet."""
    review_day = rng.randint(15, 28)
    selected = _id(rng, "K")
    decoy = _id(rng, "K")
    scope = _id(rng, "S")
    other_scope = _id(rng, "S")
    return {
        "selected": selected,
        "decoy": decoy,
        "scope": scope,
        "other_scope": other_scope,
        "review_day": review_day,
        "veto_current": True,
        "veto_starts": review_day - rng.randint(1, 4),
        "archived_scope": scope if rng.randrange(2) else other_scope,
    }


def _variant(base: dict[str, Any], rng: random.Random, level: int) -> dict[str, Any]:
    facts = copy.deepcopy(base)
    alpha = beta = "verified"
    if level == 1:
        if rng.randrange(2):
            alpha = "unresolved"
        else:
            beta = "unresolved"
    elif level == 0:
        # This unresolved requirement is not a shortcut to level 1 because
        # an operative disqualifier has precedence.
        if rng.randrange(2):
            alpha = "unresolved"
        else:
            beta = "unresolved"
    facts.update(
        {
            "alpha": alpha,
            "beta": beta,
            "veto_scope": base["scope"] if level == 0 else base["other_scope"],
            "decoy_alpha": "unresolved" if alpha == "verified" else "verified",
            "decoy_beta": "unresolved" if beta == "verified" else "verified",
        }
    )
    return facts


def _oracle_one(facts: dict[str, Any]) -> int:
    veto = (
        facts["veto_current"]
        and facts["veto_starts"] <= facts["review_day"]
        and facts["veto_scope"] == facts["scope"]
    )
    if veto:
        return 0
    return 2 if facts["alpha"] == facts["beta"] == "verified" else 1


def _documents(facts: dict[str, Any], domain: str, role: str) -> str:
    a = [
        {
            "case": facts["selected"],
            "requirement_A": facts["alpha"],
            "latest_day": facts["review_day"] - 1,
        },
        {
            "case": facts["decoy"],
            "requirement_A": facts["decoy_alpha"],
            "latest_day": facts["review_day"],
        },
    ]
    b = [
        {
            "case": facts["selected"],
            "scope": facts["scope"],
            "requirement_B": facts["beta"],
            "latest_day": facts["review_day"] - 1,
        },
        {
            "case": facts["decoy"],
            "scope": facts["other_scope"],
            "requirement_B": facts["decoy_beta"],
            "latest_day": facts["review_day"],
        },
    ]
    notices = [
        {
            "state": "current",
            "scope": facts["veto_scope"],
            "starts": facts["veto_starts"],
            "kind": "disqualifier",
        },
        {
            "state": "archived",
            "scope": facts["archived_scope"],
            "starts": facts["review_day"] - 3,
            "kind": "disqualifier",
        },
    ]
    if role == "select":
        # New document order and field presentation for source-disjoint
        # selector renderings; the decision rule is still shared.
        a.reverse()
        b.reverse()
        notices.reverse()
    return (
        f"Procedure: {domain}. Review case {facts['selected']} on day "
        f"{facts['review_day']}.\n"
        f"Document A, signed requirement ledger and notices: "
        f"{json.dumps({'ledger': a, 'notices': notices}, sort_keys=True)}\n"
        f"Document B, current scoped registry: {json.dumps(b, sort_keys=True)}"
    )


def _oracle_two(state: str, case_id: str, review_day: int) -> int:
    """Independently parse rendered document facts, then recompute verdict."""
    lines = state.splitlines()
    if len(lines) != 3:
        raise ValueError("Rendered document count differs")
    bundle = json.loads(lines[1].split(": ", 1)[1])
    registry = json.loads(lines[2].split(": ", 1)[1])
    a = next(row for row in bundle["ledger"] if row["case"] == case_id)
    b = next(row for row in registry if row["case"] == case_id)
    blocked = any(
        notice["state"] == "current"
        and notice["kind"] == "disqualifier"
        and notice["scope"] == b["scope"]
        and notice["starts"] <= review_day
        for notice in bundle["notices"]
    )
    if blocked:
        return 0
    if a["requirement_A"] != "verified" or b["requirement_B"] != "verified":
        return 1
    return 2


def build(secret: bytes, role: str, groups: int) -> list[dict[str, Any]]:
    if role not in {"train", "select"} or groups < 1:
        raise ValueError("Unknown role or empty group request")
    domains = TRAIN_DOMAINS if role == "train" else SELECT_DOMAINS
    rows = []
    for index in range(groups):
        rng = _rng(secret, role, index)
        domain = domains[index % len(domains)]
        group = f"d2score_v7p_{role}_{index:04d}"
        base = _base_facts(rng)
        for level in LEVELS:
            facts = _variant(base, rng, level)
            state = _documents(facts, domain, role)
            if _oracle_one(facts) != level:
                raise AssertionError("Structured oracle mismatch")
            if _oracle_two(state, facts["selected"], facts["review_day"]) != level:
                raise AssertionError("Rendered document oracle mismatch")
            row = {
                "id": f"{group}_l{level}",
                "state": state,
                "instructions": (
                    "Use only the latest current records for the named case. "
                    "A current scoped disqualifier that started by the review "
                    "day gives level 0, even if a requirement is unresolved. "
                    "Ignore archived notices and other cases. Otherwise level "
                    "1 means at least one of requirements A and B is unresolved; "
                    "level 2 requires both verified. Choose exactly one level."
                ),
                "options": [
                    {"key": str(i), "description": description}
                    for i, description in enumerate(OPTIONS)
                ],
                "label": level,
                "task_type": "score",
                "family": "evidence_state_join",
                "group_id": group,
                "language": "en",
                "split": role,
                "source": VERSION,
                "evaluation_role": role,
                "render_template": f"v7p-{role}-{index % len(domains)}",
                "audit_metadata": {
                    "domain": domain,
                    "group_index": index,
                    "level": level,
                    "case_id": facts["selected"],
                    "review_day": facts["review_day"],
                    "source_documents": 2,
                    "oracle_version": VERSION,
                },
            }
            row["input_sha256"] = digest({field: row[field] for field in INPUT_FIELDS})
            validate_row(row, role)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", required=True, type=Path)
    parser.add_argument("--role", choices=("train", "select"), required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Candidate data output already exists")
    rows = build(_secret(args.seed_file), args.role, args.groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
