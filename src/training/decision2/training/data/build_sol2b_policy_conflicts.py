"""Generate private, program-oracled Sol 2B policy-conflict source candidates.

This module never reads benchmark gold or protected prompts. Its output is a
TRAIN candidate, not an admitted source; separate rights, blind review,
overlap, shortcut and exact-token gates are still required.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.model.data import INPUT_FIELDS, canonical, digest, validate_row

SEED = "decision2-sol2b-policy-conflicts-v1"
SOURCE = "decision2_sol2b_policy_conflicts_v1"
FAMILIES = ("eligibility", "permission", "scheduling", "inventory")
NAMES = (
    "Alder",
    "Beryl",
    "Cedar",
    "Delta",
    "Elm",
    "Fjord",
    "Grove",
    "Harbor",
    "Iris",
    "Juniper",
    "Kestrel",
    "Linden",
    "Marble",
    "Nimbus",
    "Orchid",
    "Pine",
)
DOMAIN = {
    "eligibility": {
        "desk": "benefits eligibility desk",
        "candidate": "application",
        "credential": "identity certificate",
        "units": "review credits",
        "opening": "opening credit balance",
        "event": "credit posting",
        "exception": "fraud hold",
        "override": "hold release",
        "pending": "household verification",
    },
    "permission": {
        "desk": "facility permission desk",
        "candidate": "access request",
        "credential": "access credential",
        "units": "available entry slots",
        "opening": "opening slot balance",
        "event": "slot adjustment",
        "exception": "security hold",
        "override": "emergency clearance",
        "pending": "escort confirmation",
    },
    "scheduling": {
        "desk": "service scheduling desk",
        "candidate": "appointment request",
        "credential": "booking authorization",
        "units": "open service slots",
        "opening": "opening slot balance",
        "event": "schedule adjustment",
        "exception": "blackout hold",
        "override": "blackout waiver",
        "pending": "arrival confirmation",
    },
    "inventory": {
        "desk": "inventory fulfillment desk",
        "candidate": "fulfillment order",
        "credential": "purchase authorization",
        "units": "available units",
        "opening": "opening stock balance",
        "event": "stock movement",
        "exception": "quality hold",
        "override": "quality release",
        "pending": "inspection confirmation",
    },
}


def _rng(*parts: object) -> random.Random:
    value = "\0".join((SEED, *(str(part) for part in parts)))
    return random.Random(int(hashlib.sha256(value.encode()).hexdigest(), 16))


@dataclass(frozen=True)
class Candidate:
    name: str
    credential: bool
    opening: int
    required: int
    events: tuple[tuple[int, bool], ...]
    hold: bool
    release: bool
    signatures: int
    pending: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "credential": self.credential,
            "opening": self.opening,
            "required": self.required,
            "events": [[delta, active] for delta, active in self.events],
            "hold": self.hold,
            "release": self.release,
            "signatures": self.signatures,
            "pending": self.pending,
        }


def final_balance(candidate: Candidate) -> int:
    return candidate.opening + sum(
        delta for delta, active in candidate.events if active
    )


def oracle_level(candidate: Candidate) -> int:
    if not candidate.credential or final_balance(candidate) < candidate.required:
        return 0
    if candidate.hold and not (candidate.release and candidate.signatures >= 2):
        return 0
    if candidate.pending or final_balance(candidate) == candidate.required:
        return 1
    return 2


def reference_level(candidate: Candidate) -> int:
    """Reexecute priority rules through a separate ordered-rule interpretation."""
    decisions = [
        (not candidate.credential, 0),
        (final_balance(candidate) < candidate.required, 0),
        (
            candidate.hold and (not candidate.release or candidate.signatures < 2),
            0,
        ),
        (candidate.pending, 1),
        (final_balance(candidate) == candidate.required, 1),
        (True, 2),
    ]
    return next(level for condition, level in decisions if condition)


def _candidate(rng: random.Random, name: str) -> Candidate:
    events = tuple((rng.randint(-2, 3), rng.random() < 0.72) for _ in range(3))
    return Candidate(
        name=name,
        credential=rng.random() < 0.8,
        opening=rng.randint(1, 7),
        required=rng.randint(1, 5),
        events=events,
        hold=rng.random() < 0.36,
        release=rng.random() < 0.5,
        signatures=rng.randint(0, 3),
        pending=rng.random() < 0.34,
    )


def _render_candidate(candidate: Candidate, spec: dict[str, str]) -> str:
    postings = "; ".join(
        f"{number}: {delta:+d} ({'posted' if active else 'voided'})"
        for number, (delta, active) in enumerate(candidate.events, 1)
    )
    return (
        f"{candidate.name} is the {spec['candidate']} on this docket. Its "
        f"{spec['credential']} is {'current' if candidate.credential else 'expired'}; "
        f"it requests {candidate.required} {spec['units']}. The recorded "
        f"{spec['opening']} is {candidate.opening}. The dated {spec['event']} "
        f"entries are {postings}. The {spec['exception']} is "
        f"{'active' if candidate.hold else 'absent'}. A written "
        f"{spec['override']} is {'present' if candidate.release else 'absent'} "
        f"and has {candidate.signatures} independent countersignatures. "
        f"The {spec['pending']} is "
        f"{'still pending' if candidate.pending else 'complete'}."
    )


def render_state(family: str, left: Candidate, right: Candidate, style: int) -> str:
    spec = DOMAIN[family]
    policy = (
        f"This is an operating bulletin for the {spec['desk']}. A request "
        "has exactly one present decision tier: 0 blocked, 1 eligible but "
        "conditional, or 2 fully approved. Apply the rules to each request "
        "independently. Do not infer approval from a name, record order, or "
        "the fact that a file has reached the desk. A current credential and "
        "sufficient resources are both mandatory gates. An expired or missing "
        "credential cannot be repaired by an exception release. A shortage "
        "cannot be repaired by an exception release either."
    )
    ledger = (
        f"For the resource gate, start with each file's {spec['opening']} "
        f"and process its dated {spec['event']} entries. Add or subtract a "
        "posted entry exactly once; ignore an entry expressly marked voided. "
        "Compare the resulting balance with that file's stated requirement, "
        "including equality. Do not combine the two candidates' balances, "
        "and do not assume that a positive entry cancels a later negative "
        "entry unless both are posted. The signed deltas are part of the "
        "record, not predictions about future resources."
    )
    exception = (
        f"An active {spec['exception']} blocks a file after its ordinary "
        f"gates pass. A written {spec['override']} lifts that hold only if "
        "the same file has at least two independent countersignatures. One "
        "signature is insufficient; a release document without the required "
        "signatures is also insufficient. A release document does not create "
        "a credential or additional resources. When there is no active hold, "
        "a release document is irrelevant rather than an extra requirement."
    )
    completion = (
        f"After all blocking gates and holds are resolved, {spec['pending']} "
        "or a final balance exactly equal to the requirement makes the file "
        "conditional at tier 1. A file whose check is complete and whose "
        "resource balance strictly exceeds its requirement is approved at "
        "tier 2. If an earlier mandatory gate fails, the tier is 0 even when "
        "a later check is complete. For queue order, process the file at the "
        "higher tier first; this docket has no equal-tier choice."
    )
    blocks = [policy, ledger, exception, completion]
    if style == 1:
        blocks = [ledger, policy, completion, exception]
    elif style == 2:
        blocks = [exception, completion, policy, ledger]
    record = [_render_candidate(left, spec), _render_candidate(right, spec)]
    if style == 2:
        record.reverse()
    return "\n\n".join(
        ["Policy and audit instructions:", *blocks, "Current case records:", *record]
    )


def _native_row(
    *,
    group_id: str,
    family: str,
    task_type: str,
    state: str,
    instructions: str,
    options: list[dict[str, str]],
    correct_key: str,
    fact_digest: str,
    style: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": f"{group_id}:{task_type}",
        "state": state,
        "instructions": instructions,
        "options": options,
        "label": next(
            index
            for index, option in enumerate(options)
            if option["key"] == correct_key
        ),
        "task_type": task_type,
        "family": f"policy_conflict_{family}",
        "group_id": group_id,
        "language": "en",
        "split": "train",
        "source": SOURCE,
        "evaluation_role": "train",
        "render_template": f"policy_conflict_{family}_style{style}_v1",
        "audit_metadata": {
            "origin": "original program-oracled research candidate",
            "fact_digest": fact_digest,
            "generator_version": 1,
            "admission_status": "candidate_not_approved",
        },
    }
    row["input_sha256"] = digest({field: row[field] for field in INPUT_FIELDS})
    return validate_row(row, "train")


def build_group(
    family: str, index: int, *, seed: str = SEED
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if family not in FAMILIES or index < 0:
        raise ValueError("Invalid policy family or group index")
    desired_choice = index % 2
    desired_noul = (index // 2) % 2
    desired_score = index % 3
    rng = _rng(seed, family, index)
    for _attempt in range(4096):
        left_name, right_name = rng.sample(NAMES, 2)
        left, right = _candidate(rng, left_name), _candidate(rng, right_name)
        left_level, right_level = oracle_level(left), oracle_level(right)
        if left_level == right_level or desired_score not in (left_level, right_level):
            continue
        if reference_level(left) != left_level or reference_level(right) != right_level:
            raise AssertionError("Independent ordered oracle disagrees")
        targets = [
            (candidate, mode, int(level == 2 if mode == "full" else level >= 1))
            for candidate, level in ((left, left_level), (right, right_level))
            for mode in ("full", "eligible")
        ]
        matching = [item for item in targets if item[2] == desired_noul]
        if not matching:
            continue
        noul_candidate, noul_mode, _truth = rng.choice(matching)
        break
    else:
        raise ValueError(f"Unable to construct balanced {family} group {index}")

    style = index % 3
    state = render_state(family, left, right, style)
    higher = left if left_level > right_level else right
    lower = right if higher is left else left
    choice_order = [higher, lower] if desired_choice == 0 else [lower, higher]
    score_candidate = left if left_level == desired_score else right
    group_id = (
        "policy-conflict-v1:"
        + family
        + ":"
        + hashlib.sha256(f"{seed}\0{family}\0{index}".encode()).hexdigest()[:16]
    )
    facts = {"left": left.as_dict(), "right": right.as_dict(), "family": family}
    fact_digest = digest(facts)
    choice = _native_row(
        group_id=group_id,
        family=family,
        task_type="choice",
        state=state,
        instructions="Which request should this desk process first under the recorded policy?",
        options=[
            {"key": "A" if position == 0 else "B", "description": f"{candidate.name}"}
            for position, candidate in enumerate(choice_order)
        ],
        correct_key="A" if desired_choice == 0 else "B",
        fact_digest=fact_digest,
        style=style,
    )
    noul_instruction = (
        f"Is {noul_candidate.name} fully approved now under the policy?"
        if noul_mode == "full"
        else f"Is {noul_candidate.name} eligible to enter the decision queue, "
        "including conditional review?"
    )
    noul = _native_row(
        group_id=group_id,
        family=family,
        task_type="noul",
        state=state,
        instructions=noul_instruction,
        options=[
            {"key": "false", "description": "No"},
            {"key": "true", "description": "Yes"},
        ],
        correct_key="true" if desired_noul else "false",
        fact_digest=fact_digest,
        style=style,
    )
    score = _native_row(
        group_id=group_id,
        family=family,
        task_type="score",
        state=state,
        instructions=f"What is the current decision tier for {score_candidate.name}?",
        options=[
            {
                "key": "0",
                "description": "Blocked by a mandatory gate or unresolved hold",
            },
            {"key": "1", "description": "Eligible but conditional"},
            {"key": "2", "description": "Fully approved"},
        ],
        correct_key=str(desired_score),
        fact_digest=fact_digest,
        style=style,
    )
    oracle = {
        "group_id": group_id,
        "fact_digest": fact_digest,
        "facts": facts,
        "levels": {left.name: left_level, right.name: right_level},
        "answers": {
            "choice": higher.name,
            "noul": bool(desired_noul),
            "score": desired_score,
        },
        "noul_subject": noul_candidate.name,
        "noul_query_mode": noul_mode,
        "score_subject": score_candidate.name,
    }
    return [choice, noul, score], oracle


def build_all(
    *, groups_per_family: int = 128, seed: str = SEED
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, oracles = [], []
    for family in FAMILIES:
        for index in range(groups_per_family):
            group_rows, oracle = build_group(family, index, seed=seed)
            rows.extend(group_rows)
            oracles.append(oracle)
    if len({item["fact_digest"] for item in oracles}) != len(oracles):
        raise ValueError("Repeated structured policy situation")
    return rows, oracles


def label_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for row in rows:
        key = row["options"][row["label"]]["key"]
        counts[row["task_type"]][key] += 1
    return {kind: dict(sorted(value.items())) for kind, value in sorted(counts.items())}


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> str:
    content = "".join(canonical(item) + "\n" for item in values)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


def write_private_packet(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    rows, oracles = build_all()
    by_group = collections.defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    blind_ids = {
        item["group_id"]
        for family in FAMILIES
        for item in sorted(
            (oracle for oracle in oracles if oracle["facts"]["family"] == family),
            key=lambda item: hashlib.sha256(
                f"{SEED}\0blind\0{item['group_id']}".encode()
            ).hexdigest(),
        )[:12]
    }
    blind = [
        {
            "group_id": group_id,
            "family": family,
            "questions": [
                {
                    "state": row["state"],
                    "task_type": row["task_type"],
                    "instructions": row["instructions"],
                    "options": row["options"],
                }
                for row in by_group[group_id]
            ],
        }
        for family in FAMILIES
        for group_id in sorted(
            item["group_id"]
            for item in oracles
            if item["facts"]["family"] == family and item["group_id"] in blind_ids
        )
    ]
    receipt = {
        "schema_version": "decision2-sol2b-policy-candidate/1",
        "source": SOURCE,
        "status": "candidate_pending_independent_review_overlap_and_token_admission",
        "groups": len(oracles),
        "native_rows": len(rows),
        "families": dict(
            sorted(
                collections.Counter(item["facts"]["family"] for item in oracles).items()
            )
        ),
        "task_types": dict(
            sorted(collections.Counter(row["task_type"] for row in rows).items())
        ),
        "labels": label_counts(rows),
        "blind_groups": len(blind),
        "blind_groups_per_family": 12,
        "train_sha256": _write_jsonl(output_dir / "train-candidate.jsonl", rows),
        "oracle_sha256": _write_jsonl(output_dir / "oracle-private.jsonl", oracles),
        "blind_sha256": _write_jsonl(output_dir / "blind-review-48.jsonl", blind),
    }
    (output_dir / "manifest-private.json").write_text(
        canonical(receipt) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(canonical(write_private_packet(args.output_dir)))


if __name__ == "__main__":
    main()
