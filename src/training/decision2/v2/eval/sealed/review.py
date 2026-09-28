"""Blind quality review of a C1 build by independent reviewer subagents.

    python3 -m v2.eval.sealed.review packet --build-dir <build> --salt-file <salt> --output-dir <review>
    python3 -m v2.eval.sealed.review score --build-dir <build> --review-dir <review> \
        --reviewer r1 --reviewer r2 --reviewer r3 --output <receipt.json>

`packet` samples each task in SHA-256(salt | review | id) order: 16 items, or 8 when
most of the task's inputs are long. It gives each item an opaque review id, shuffles,
and writes gold-free packets (`packet-short.jsonl`, `packet-long.jsonl`: review id, state
and question only, no source or task name) plus a private id mapping.

Reviewers append one JSON line per item to `answers-<reviewer>-<packet>.jsonl`:
{"review_id", "answer" (choice key | true/false | level index | null), "confidence"
("low"/"medium"/"high"), "flags": [...], "note"}.

`score` applies the preregistered task-level rules. A task fails review if at least 25%
of its sampled items carry a quality flag from at least two reviewers, or if the
reviewers' majority answer agrees with the human gold no more often than chance. Items
are never dropped for disagreeing with the gold. The receipt holds counts and rates only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "dev2-sealed-c1-review/1"
PER_TASK = 16
PER_LONG_TASK = 8
LONG_SHARE = 0.5
FLAG_SHARE = 0.25
QUALITY_FLAGS = (
    "ambiguous",
    "not_answerable_from_state",
    "template_mismatch",
    "answer_hinted",
    "wrong_language_or_garbled",
)
FLAGS = QUALITY_FLAGS + ("sensitive_content",)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(salt: str, *parts: str) -> str:
    return hashlib.sha256("|".join((salt, *parts)).encode("utf-8")).hexdigest()


def write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def sample(
    gold: list[dict[str, Any]], salt: str, tasks: list[str] | None = None
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold:
        if tasks is None or row["task"] in tasks:
            by_task[row["task"]].append(row)
    chosen = []
    for task, rows in sorted(by_task.items()):
        long_share = sum(row["long"] for row in rows) / len(rows)
        size = PER_LONG_TASK if long_share >= LONG_SHARE else PER_TASK
        rows = sorted(rows, key=lambda row: digest(salt, "review", row["id"]))
        chosen.extend(rows[:size])
    return chosen


def packet(args: argparse.Namespace) -> int:
    salt = args.salt_file.read_text(encoding="utf-8").strip()
    gold = read_jsonl(args.build_dir / "gold.jsonl")
    chosen = sample(gold, salt, getattr(args, "task", None))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output_dir, 0o700)
    packets: dict[str, list[dict[str, Any]]] = {"short": [], "long": []}
    mapping = {}
    for row in chosen:
        review_id = "rv-" + digest(salt, "rv", row["id"])[:10]
        mapping[review_id] = row["id"]
        packets["long" if row["long"] else "short"].append(
            {
                "review_id": review_id,
                "state": row["state"],
                "question": row["questions"]["decision"],
            }
        )
    receipt: dict[str, Any] = {"schema": SCHEMA, "packets": {}, "tasks": {}}
    for name, rows in packets.items():
        rows.sort(key=lambda row: row["review_id"])
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        write_new(args.output_dir / f"packet-{name}.jsonl", data)
        receipt["packets"][name] = {
            "items": len(rows),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    write_new(
        args.output_dir / "mapping.json",
        (json.dumps(mapping, indent=1, sort_keys=True) + "\n").encode(),
    )
    receipt["tasks"] = dict(sorted(Counter(row["task"] for row in chosen).items()))
    write_new(
        args.output_dir / "packet-receipt.json",
        (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


def normalise(answer: Any, qtype: str) -> Any:
    if answer is None:
        return None
    if qtype == "noul":
        if isinstance(answer, bool):
            return answer
        text = str(answer).strip().lower()
        return {"true": True, "false": False}.get(text)
    if qtype == "score":
        try:
            return int(answer)
        except (TypeError, ValueError):
            return None
    return str(answer).strip()


def fleiss_kappa(ratings: list[list[Any]], categories: list[Any]) -> float | None:
    """Nominal Fleiss kappa for items rated by the same number of raters."""
    rows = [r for r in ratings if len(r) >= 2 and all(x is not None for x in r)]
    if not rows or len({len(r) for r in rows}) != 1:
        return None
    n = len(rows[0])
    counts = [[r.count(c) for c in categories] for r in rows]
    p_items = [(sum(c * c for c in row) - n) / (n * (n - 1)) for row in counts]
    totals = [sum(row[j] for row in counts) for j in range(len(categories))]
    grand = sum(totals)
    p_cat = [t / grand for t in totals]
    p_bar = sum(p_items) / len(p_items)
    p_exp = sum(p * p for p in p_cat)
    return (p_bar - p_exp) / (1 - p_exp) if p_exp < 1 else None


def score(args: argparse.Namespace) -> int:
    gold = {row["id"]: row for row in read_jsonl(args.build_dir / "gold.jsonl")}
    mapping = json.loads((args.review_dir / "mapping.json").read_text(encoding="utf-8"))
    answers: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for reviewer in args.reviewer:
        for path in sorted(args.review_dir.glob(f"answers-{reviewer}-*.jsonl")):
            for row in read_jsonl(path):
                if row.get("review_id") in mapping:
                    answers[row["review_id"]][reviewer] = row
    by_task: dict[str, list[str]] = defaultdict(list)
    for review_id, item_id in mapping.items():
        by_task[gold[item_id]["task"]].append(review_id)
    tasks = {}
    for task, review_ids in sorted(by_task.items()):
        first = gold[mapping[review_ids[0]]]
        question = first["questions"]["decision"]
        qtype = question["type"]
        categories = (
            [True, False]
            if qtype == "noul"
            else (
                list(question["criteria"])
                if qtype == "choice"
                else list(range(len(question["criteria"])))
            )
        )
        chance = 1 / len(categories)
        flagged_items = majority_agree = within_one = answered_all = 0
        flag_counts: Counter = Counter()
        per_reviewer = Counter()
        ratings = []
        for review_id in review_ids:
            item = gold[mapping[review_id]]
            truth = item["gold"]["decision"]["value"]
            given = answers.get(review_id, {})
            values = []
            quality = 0
            for reviewer in args.reviewer:
                row = given.get(reviewer)
                value = normalise(row.get("answer"), qtype) if row else None
                values.append(value)
                flags = set(row.get("flags") or []) if row else set()
                flag_counts.update(f for f in flags if f in FLAGS)
                quality += bool(flags & set(QUALITY_FLAGS))
                per_reviewer[reviewer] += value == truth
            flagged_items += quality >= 2
            present = [v for v in values if v is not None]
            answered_all += len(present) == len(args.reviewer)
            ratings.append(values)
            if present:
                top, count = Counter(present).most_common(1)[0]
                if count * 2 > len(present):
                    majority_agree += top == truth
                    if qtype == "score":
                        within_one += abs(top - truth) <= 1
        n = len(review_ids)
        flagged_share = flagged_items / n
        agreement = majority_agree / n
        fails = []
        if flagged_share >= FLAG_SHARE:
            fails.append("quality flags")
        if agreement <= chance:
            fails.append("majority agreement not above chance")
        tasks[task] = {
            "type": qtype,
            "sampled": n,
            "answered_by_all": answered_all,
            "flagged_by_two_or_more": flagged_items,
            "flagged_share": flagged_share,
            "flag_counts": dict(sorted(flag_counts.items())),
            "majority_agreement_with_gold": agreement,
            "majority_within_one": within_one / n if qtype == "score" else None,
            "chance": chance,
            "per_reviewer_agreement": {r: per_reviewer[r] / n for r in args.reviewer},
            "fleiss_kappa": fleiss_kappa(ratings, categories),
            "decision": "FAIL: " + "; ".join(fails) if fails else "PASS",
        }
    receipt = {
        "schema": SCHEMA,
        "reviewers": args.reviewer,
        "rule": (
            f"task fails if >= {FLAG_SHARE:.0%} of sampled items carry a quality flag "
            "from >= 2 reviewers, or majority agreement with the human gold <= chance"
        ),
        "tasks": tasks,
        "failed_tasks": sorted(t for t, v in tasks.items() if v["decision"] != "PASS"),
    }
    data = (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode()
    write_new(args.output, data)
    print(
        json.dumps(
            {
                "failed_tasks": receipt["failed_tasks"],
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("packet")
    one.add_argument("--build-dir", type=Path, required=True)
    one.add_argument("--salt-file", type=Path, required=True)
    one.add_argument("--output-dir", type=Path, required=True)
    one.add_argument("--task", action="append", help="only these tasks (re-review)")
    two = commands.add_parser("score")
    two.add_argument("--build-dir", type=Path, required=True)
    two.add_argument("--review-dir", type=Path, required=True)
    two.add_argument("--reviewer", action="append", required=True)
    two.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return packet(args) if args.command == "packet" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
