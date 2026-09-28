"""One prospective correction of the v1 Sol 2B Noul sampling shortcut.

Output remains a private TRAIN candidate. Passing generation never admits a
source: independent blind review, overlap, rights, exact budget and parity
screens are separate gates.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import random
from pathlib import Path
from typing import Any

from training.data.build_sol2b_policy_conflicts import (
    FAMILIES,
    NAMES,
    _candidate,
    label_counts,
    oracle_level,
    reference_level,
    render_state,
)
from training.model.data import INPUT_FIELDS, canonical, digest, validate_row

SEED = "decision2-sol2b-policy-conflicts-v2"
SOURCE = "decision2_sol2b_policy_conflicts_v2"
DIAGNOSTIC_SEED = "decision2-sol2b-policy-conflicts-v2-diagnostic"


def _rng(seed: str, family: str, index: int) -> random.Random:
    material = f"{seed}\0{family}\0{index}"
    return random.Random(int(hashlib.sha256(material.encode()).hexdigest(), 16))


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
        "render_template": f"policy_conflict_{family}_style{style}_v2",
        "audit_metadata": {
            "origin": "original program-oracled research candidate",
            "fact_digest": fact_digest,
            "generator_version": 2,
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
    desired_choice = index & 1
    desired_noul = bool((index >> 1) & 1)
    noul_mode = "eligible" if (index >> 2) & 1 else "full"
    subject_side = (index >> 3) & 1
    desired_score = index % 3
    rng = _rng(seed, family, index)

    for _attempt in range(4096):
        left_name, right_name = rng.sample(NAMES, 2)
        left, right = _candidate(rng, left_name), _candidate(rng, right_name)
        left_level, right_level = oracle_level(left), oracle_level(right)
        if left_level == right_level or desired_score not in (left_level, right_level):
            continue
        subject = (left, right)[subject_side]
        subject_level = (left_level, right_level)[subject_side]
        subject_truth = (
            subject_level == 2 if noul_mode == "full" else subject_level >= 1
        )
        if subject_truth != desired_noul:
            continue
        if reference_level(left) != left_level or reference_level(right) != right_level:
            raise AssertionError("Independent ordered oracle disagrees")
        break
    else:
        raise ValueError(f"Unable to construct orthogonal {family} group {index}")

    style = index % 3
    state = render_state(family, left, right, style)
    higher = left if left_level > right_level else right
    lower = right if higher is left else left
    choice_order = [higher, lower] if desired_choice == 0 else [lower, higher]
    score_candidate = left if left_level == desired_score else right
    group_id = (
        "policy-conflict-v2:"
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
            {"key": "A" if position == 0 else "B", "description": candidate.name}
            for position, candidate in enumerate(choice_order)
        ],
        correct_key="A" if desired_choice == 0 else "B",
        fact_digest=fact_digest,
        style=style,
    )
    noul_instruction = (
        f"Is {subject.name} fully approved now under the policy?"
        if noul_mode == "full"
        else f"Is {subject.name} eligible to enter the decision queue, "
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
            "noul": desired_noul,
            "score": desired_score,
        },
        "noul_subject": subject.name,
        "noul_query_mode": noul_mode,
        "noul_subject_side": subject_side,
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
        "schema_version": "decision2-sol2b-policy-candidate/2",
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
