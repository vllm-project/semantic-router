"""Compare a sealed HelpSteer2 ordinal pilot review with its private key.

Run only after the answer-blind review is sealed. The output contains aggregate
statistics and file identities; it never includes source text or record IDs.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def weighted_kappa(gold: list[int], review: list[int]) -> float:
    """Quadratic-weighted agreement on the fixed five-level grade scale."""
    if len(gold) != len(review) or not gold:
        raise ValueError("Grades have unequal or empty lengths")
    g = collections.Counter(gold)
    r = collections.Counter(review)
    n = len(gold)
    observed = sum((a - b) ** 2 for a, b in zip(gold, review)) / n
    expected = sum((a - b) ** 2 * g[a] * r[b] for a in range(5) for b in range(5)) / (
        n * n
    )
    return 1 - observed / expected if expected else (1.0 if observed == 0 else 0.0)


def compare(
    blind: list[dict[str, Any]],
    key: list[dict[str, Any]],
    sealed: dict[str, Any],
) -> dict[str, Any]:
    if sealed.get("gold_accessed") is not False:
        raise ValueError("Reviewer did not declare answer blindness")
    if len(blind) != len(key) or len(key) != len(sealed["rows"]):
        raise ValueError("Packet, key and review lengths differ")
    if len(key) != 24:
        raise ValueError("This frozen pilot must have 24 response rows")
    for name, rows in (("blind", blind), ("key", key), ("review", sealed["rows"])):
        ids = [row["id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate {name} IDs")
    blind_by_id = {row["id"]: row for row in blind}
    key_by_id = {row["id"]: row for row in key}
    review_by_id = {row["id"]: row for row in sealed["rows"]}
    if set(blind_by_id) != set(key_by_id) or set(key_by_id) != set(review_by_id):
        raise ValueError("Packet, key and review IDs differ")

    groups: dict[str, list[str]] = collections.defaultdict(list)
    records = []
    flag_counts: collections.Counter[str] = collections.Counter()
    for item_id, answer in key_by_id.items():
        packet = blind_by_id[item_id]
        judgment = review_by_id[item_id]
        if not (packet["group_id"] == answer["group_id"] == judgment["group_id"]):
            raise ValueError("Group identity mismatch")
        grade = answer["correctness"]
        opinion = judgment["correctness_grade_0_to_4"]
        if type(grade) is not int or type(opinion) is not int:
            raise ValueError("Non-integral grade")
        if grade not in range(5) or opinion not in range(5):
            raise ValueError("Grade outside 0..4")
        if not isinstance(judgment["flags"], list):
            raise ValueError("Flags must be a list")
        flag_counts.update(judgment["flags"])
        groups[answer["group_id"]].append(item_id)
        records.append((grade, opinion, bool(judgment["flags"])))
    if len(groups) != 12 or any(len(ids) != 2 for ids in groups.values()):
        raise ValueError("Pilot must contain twelve two-response groups")
    aggregate = sealed.get("aggregate", {})
    if aggregate.get("sample_count") != 24 or aggregate.get("independent_groups") != 12:
        raise ValueError("Review's own counts do not match frozen packet")

    gold = [a for a, _, _ in records]
    opinion = [b for _, b, _ in records]
    confusion = [[0 for _ in range(5)] for _ in range(5)]
    for a, b in zip(gold, opinion):
        confusion[a][b] += 1
    exact = sum(a == b for a, b in zip(gold, opinion))
    within_one = sum(abs(a - b) <= 1 for a, b in zip(gold, opinion))
    large_errors = sum(abs(a - b) >= 2 for a, b in zip(gold, opinion))
    flag_subgroups = {}
    for flagged in (False, True):
        rows = [(a, b) for a, b, f in records if f == flagged]
        flag_subgroups["flagged" if flagged else "unflagged"] = {
            "n": len(rows),
            "exact": sum(a == b for a, b in rows),
            "within_one": sum(abs(a - b) <= 1 for a, b in rows),
        }

    pair_outcomes: collections.Counter[str] = collections.Counter()
    pair_by_length: dict[str, collections.Counter[str]] = {
        "gold_higher_longer": collections.Counter(),
        "gold_higher_shorter": collections.Counter(),
    }
    pair_by_flag: dict[str, collections.Counter[str]] = {
        "any_flag": collections.Counter(),
        "no_flag": collections.Counter(),
    }
    for ids in groups.values():
        low_id, high_id = sorted(
            ids, key=lambda item_id: key_by_id[item_id]["correctness"]
        )
        if key_by_id[low_id]["correctness"] == key_by_id[high_id]["correctness"]:
            raise ValueError("Frozen pair has equal source grades")
        low_review = review_by_id[low_id]["correctness_grade_0_to_4"]
        high_review = review_by_id[high_id]["correctness_grade_0_to_4"]
        outcome = (
            "concordant"
            if high_review > low_review
            else ("tie" if high_review == low_review else "discordant")
        )
        pair_outcomes[outcome] += 1
        high_longer = len(blind_by_id[high_id]["state"]["candidate_response"]) > len(
            blind_by_id[low_id]["state"]["candidate_response"]
        )
        pair_by_length["gold_higher_longer" if high_longer else "gold_higher_shorter"][
            outcome
        ] += 1
        any_flag = bool(review_by_id[low_id]["flags"] or review_by_id[high_id]["flags"])
        pair_by_flag["any_flag" if any_flag else "no_flag"][outcome] += 1

    return {
        "response_rows": len(records),
        "independent_pairs": len(groups),
        "source_grade_histogram": dict(sorted(collections.Counter(gold).items())),
        "review_grade_histogram": dict(sorted(collections.Counter(opinion).items())),
        "confusion_source_by_review": confusion,
        "exact": exact,
        "within_one": within_one,
        "large_errors_ge_2": large_errors,
        "mean_absolute_error": sum(abs(a - b) for a, b in zip(gold, opinion))
        / len(gold),
        "mean_signed_review_minus_source": sum(b - a for a, b in zip(gold, opinion))
        / len(gold),
        "quadratic_weighted_kappa": weighted_kappa(gold, opinion),
        "pair_outcomes": dict(pair_outcomes),
        "pair_by_length_direction": {k: dict(v) for k, v in pair_by_length.items()},
        "pair_by_flag": {k: dict(v) for k, v in pair_by_flag.items()},
        "flag_counts": dict(sorted(flag_counts.items())),
        "flag_subgroups": flag_subgroups,
        "reviewer_construct_pass": aggregate.get("construct_pass"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--blind-sha256", required=True)
    parser.add_argument("--key-sha256", required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {
        "blind": (args.blind, args.blind_sha256),
        "key": (args.key, args.key_sha256),
        "review": (args.review, args.review_sha256),
    }
    for name, (path, expected) in files.items():
        if sha256(path) != expected:
            raise ValueError(f"{name} hash mismatch")
    if args.review.stat().st_mtime_ns <= max(
        args.blind.stat().st_mtime_ns, args.key.stat().st_mtime_ns
    ):
        raise ValueError("Sealed review does not postdate frozen packet/key")
    sealed = json.loads(args.review.read_text())
    if sealed.get("source_file_sha256") != args.blind_sha256:
        raise ValueError("Sealed review references a different blind packet")
    result = compare(read_jsonl(args.blind), read_jsonl(args.key), sealed)
    receipt = {
        "analysis_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": {name: expected for name, (_, expected) in files.items()},
        "mtime_ns": {
            name: path.stat().st_mtime_ns for name, (path, _) in files.items()
        },
        "review_declared_gold_accessed": sealed["gold_accessed"],
        "metrics": result,
    }
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
