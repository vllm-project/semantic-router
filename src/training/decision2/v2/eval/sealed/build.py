"""Select the JevArena-C1 items from converter candidates (salted, balanced, capped).

    python3 -m v2.eval.sealed.build select --candidates-dir <dir> --config <config.json> \
        --salt-file <private salt> --overlap-hits <hits.jsonl> --output-dir <private dir>

Per task, in the order of SHA-256(salt | task | source item id):

1. Drop candidates the overlap scan marks OVERLAP, REVIEW with containment >= 0.2, or
   unscreenable (no shingle); drop repeats of a normalised state within the task.
2. Pick the balancing stratum from the eligible pool without looking at item text:
   `gold_length_rank` when the gold is the unique longest option at least 5 points more
   often than chance (per-item option sets); `length` when a length-only baseline
   (majority gold per input-length quintile, 5-fold CV by group) beats the better of
   majority and chance by at least 5 points; otherwise the gold.
3. Take up to the stratum quota (length: equal gold counts within every length
   quintile; gold_length_rank: equal counts per rank; gold: cap // classes, all of a
   smaller class), at most `group_cap` items per source group, stopping at `cap`.

Writes prompts.jsonl (gold-free System One rows with opaque ids), gold.jsonl,
manifest.json (counts and hashes) and a count-only receipt; private files are mode 600.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from v2.eval import leak_audit
from v2.eval.sealed.schema import LONG_INPUT_CHARS, input_chars, normalized

SCHEMA = "dev2-sealed-c1-build/1"
FOLDS = 5
LENGTH_GAIN = 5.0
OPTION_LENGTH_GAIN = 0.05
REVIEW_EXCLUDE = 0.2
BINS = 5


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def salted(salt: str, *parts: str) -> str:
    return hashlib.sha256("|".join((salt, *parts)).encode("utf-8")).hexdigest()


def fold(group: str) -> int:
    return int(hashlib.sha256(group.encode("utf-8")).hexdigest(), 16) % FOLDS


def excluded(hit: dict[str, Any] | None) -> str | None:
    """Why the overlap scan excludes a candidate (None = keep)."""
    if hit is None:
        return "not_scanned"
    if hit.get("shingles", 0) == 0:
        return "unscreenable"
    if hit["verdict"] == "OVERLAP":
        return "overlap"
    if hit["verdict"] == "REVIEW" and hit.get("containment", 0.0) >= REVIEW_EXCLUDE:
        return "partial_overlap"
    return None


def quintiles(values: list[int]) -> list[float]:
    ordered = sorted(values)
    return [
        ordered[min(len(ordered) - 1, (len(ordered) * k) // BINS)]
        for k in range(1, BINS)
    ]


def bin_of(value: int, edges: list[float]) -> int:
    return sum(value > edge for edge in edges)


def length_baseline(rows: list[dict[str, Any]]) -> dict[str, float]:
    """CV accuracy of the majority gold per input-length quintile vs the reference."""
    if len(rows) < 2 * FOLDS:
        return {"baseline": float("nan"), "reference": float("nan"), "gain": 0.0}
    lengths = [row["_chars"] for row in rows]
    edges = quintiles(lengths)
    golds = [json.dumps(row["gold"]) for row in rows]
    folds = [fold(row["group_id"]) for row in rows]
    correct = majority_correct = 0
    for k in range(FOLDS):
        train = [i for i, f in enumerate(folds) if f != k]
        test = [i for i, f in enumerate(folds) if f == k]
        if not train or not test:
            continue
        overall = Counter(golds[i] for i in train).most_common(1)[0][0]
        per_bin: dict[int, Counter] = defaultdict(Counter)
        for i in train:
            per_bin[bin_of(lengths[i], edges)][golds[i]] += 1
        for i in test:
            cell = per_bin.get(bin_of(lengths[i], edges))
            guess = cell.most_common(1)[0][0] if cell else overall
            correct += guess == golds[i]
            majority_correct += overall == golds[i]
    n = len(rows)
    chance = 100.0 * statistics.mean(1.0 / option_count(row) for row in rows)
    reference = max(100.0 * majority_correct / n, chance)
    baseline = 100.0 * correct / n
    return {"baseline": baseline, "reference": reference, "gain": baseline - reference}


def option_count(row: dict[str, Any]) -> int:
    question = row["question"]
    if question["type"] == "noul":
        return 2
    return len(question["criteria"])


def gold_length_rank(row: dict[str, Any]) -> int | None:
    """Rank of the gold option's description length (0 = longest) for choice rows."""
    question = row["question"]
    if question["type"] != "choice":
        return None
    lengths = [len(text) for text in question["criteria"].values()]
    gold = lengths[list(question["criteria"]).index(row["gold"])]
    return sum(length > gold for length in lengths)


def option_length_gain(rows: list[dict[str, Any]]) -> float:
    """Rate the gold is the unique longest option minus the rate expected by chance.

    Only per-item option sets count: with one fixed option set the longest option is
    always the same label, so its rate is the label prior, not a surface cue.
    """
    if any(row["question"]["type"] != "choice" for row in rows):
        return 0.0
    signatures = {
        json.dumps(row["question"]["criteria"], sort_keys=True) for row in rows
    }
    if len(signatures) < 2:
        return 0.0
    hits = expected = 0.0
    for row in rows:
        lengths = [len(text) for text in row["question"]["criteria"].values()]
        top = max(lengths)
        if lengths.count(top) != 1:
            continue
        expected += 1.0 / len(lengths)
        hits += gold_length_rank(row) == 0
    return (hits - expected) / len(rows) if rows else 0.0


def choose_balance(rows: list[dict[str, Any]]) -> tuple[str, dict[str, float]]:
    length = length_baseline(rows)
    option_gain = option_length_gain(rows)
    checks = {"length_gain": length["gain"], "option_length_gain": option_gain}
    if option_gain >= OPTION_LENGTH_GAIN:
        return "gold_length_rank", checks
    if length["gain"] >= LENGTH_GAIN:
        return "length", checks
    return "gold", checks


def select_task(
    rows: list[dict[str, Any]], cap: int, group_cap: int, balance: str
) -> list[dict[str, Any]]:
    """rows are sorted by salted hash; returns the selected rows in that order."""
    if balance == "length":
        edges = quintiles([row["_chars"] for row in rows])
        stratum = {
            id(row): (bin_of(row["_chars"], edges), json.dumps(row["gold"]))
            for row in rows
        }
        golds = sorted({json.dumps(row["gold"]) for row in rows})
        available = Counter(stratum.values())
        per_cell = max(1, cap // (BINS * len(golds)))
        quota = {}
        for b in range(BINS):
            smallest = min(available.get((b, g), 0) for g in golds)
            for g in golds:
                quota[(b, g)] = min(per_cell, smallest)
    elif balance == "gold_length_rank":
        stratum = {id(row): gold_length_rank(row) for row in rows}
        available = Counter(stratum.values())
        ranks = sorted(available)
        smallest = min(available.values())
        quota = {r: min(max(1, cap // len(ranks)), smallest) for r in ranks}
    else:
        stratum = {id(row): json.dumps(row["gold"]) for row in rows}
        available = Counter(stratum.values())
        per_class = max(1, cap // len(available))
        quota = {key: min(per_class, count) for key, count in available.items()}
    taken, per_stratum, per_group = [], Counter(), Counter()
    for row in rows:
        if len(taken) >= cap:
            break
        key = stratum[id(row)]
        if (
            per_stratum[key] >= quota.get(key, 0)
            or per_group[row["group_id"]] >= group_cap
        ):
            continue
        taken.append(row)
        per_stratum[key] += 1
        per_group[row["group_id"]] += 1
    return taken


def gold_record(row: dict[str, Any]) -> dict[str, Any]:
    question, value = row["question"], row["gold"]
    record = {"type": question["type"], "value": value, "semantic_value": value}
    if question["type"] == "choice":
        record["label_to_semantic"] = {key: key for key in question["criteria"]}
    return record


def audit_questions(rows: list[dict[str, Any]]) -> list[leak_audit.Question]:
    out = []
    for row in rows:
        question = row["question"]
        out.append(
            leak_audit.native_question(
                "sealed-c1",
                row["_id"],
                row["task"],
                row["group_id"],
                question,
                row["gold"],
                row["state"],
            )
        )
    return out


def select(args: argparse.Namespace) -> int:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    salt = args.salt_file.read_text(encoding="utf-8").strip()
    if len(salt) < 32:
        raise ValueError("salt must be at least 32 characters")
    hits = {hit["id"]: hit for hit in read_jsonl(args.overlap_hits)}
    inputs, by_task = {}, defaultdict(list)
    exclusions: dict[str, Counter] = defaultdict(Counter)
    for source in config["sources"]:
        path = args.candidates_dir / f"{source}.jsonl"
        inputs[source] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in read_jsonl(path):
            if row["task"] not in config["tasks"]:
                exclusions[row["task"]]["task_not_configured"] += 1
                continue
            reason = excluded(hits.get(f"{row['task']}|{row['source_item_id']}"))
            if reason:
                exclusions[row["task"]][reason] += 1
                continue
            row["_chars"] = input_chars(row["state"], row["question"])
            row["_order"] = salted(salt, row["task"], row["source_item_id"])
            by_task[row["task"]].append(row)
    selected, tasks = [], {}
    for task, spec in sorted(config["tasks"].items()):
        rows = sorted(by_task.get(task, []), key=lambda row: row["_order"])
        seen, unique = set(), []
        for row in rows:
            key = normalized(
                json.dumps(row["state"], ensure_ascii=False, sort_keys=True)
            )
            if key in seen:
                exclusions[task]["duplicate_state"] += 1
                continue
            seen.add(key)
            unique.append(row)
        if not unique:
            tasks[task] = {"selected": 0, "eligible": 0}
            continue
        balance, checks = choose_balance(unique)
        if spec.get("balance"):
            balance = spec["balance"]
        chosen = select_task(unique, spec["cap"], spec.get("group_cap", 1), balance)
        for row in chosen:
            row["_id"] = "c1-" + salted(salt, "id", task, row["source_item_id"])[:16]
        after = {
            "length_gain": length_baseline(chosen)["gain"],
            "option_length_gain": option_length_gain(chosen),
        }
        tasks[task] = {
            "eligible": len(unique),
            "selected": len(chosen),
            "groups": len({row["group_id"] for row in chosen}),
            "balance": balance,
            "pool_checks": checks,
            "selected_checks": after,
            "labels": dict(
                sorted(Counter(json.dumps(row["gold"]) for row in chosen).items())
            ),
            "languages": dict(
                sorted(Counter(row["language"] for row in chosen).items())
            ),
            "long_input": sum(row["_chars"] >= LONG_INPUT_CHARS for row in chosen),
            "type": chosen[0]["question"]["type"] if chosen else None,
            "exclusions": dict(exclusions.get(task, {})),
        }
        selected.extend(chosen)
    selected.sort(key=lambda row: row["_id"])
    if len({row["_id"] for row in selected}) != len(selected):
        raise ValueError("opaque id collision")
    audit = leak_audit.audit_panel(audit_questions(selected), args.replicates)
    prompts = [
        {
            "id": row["_id"],
            "state": row["state"],
            "questions": {"decision": row["question"]},
        }
        for row in selected
    ]
    gold = [
        {
            "id": row["_id"],
            "task": row["task"],
            "source": row["source"],
            "source_item_id": row["source_item_id"],
            "group_id": row["group_id"],
            "language": row["language"],
            "input_chars": row["_chars"],
            "long": row["_chars"] >= LONG_INPUT_CHARS,
            "state": row["state"],
            "questions": {"decision": row["question"]},
            "gold": {"decision": gold_record(row)},
        }
        for row in selected
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output_dir, 0o700)
    digests = {}
    for name, rows in (("prompts.jsonl", prompts), ("gold.jsonl", gold)):
        data = "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows
        ).encode()
        write_new(args.output_dir / name, data)
        digests[name] = hashlib.sha256(data).hexdigest()
    by_type = Counter(row["question"]["type"] for row in selected)
    receipt = {
        "schema": SCHEMA,
        "salt_sha256": hashlib.sha256(salt.encode()).hexdigest(),
        "config": config,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "candidates_sha256": inputs,
        "overlap_hits_sha256": hashlib.sha256(
            args.overlap_hits.read_bytes()
        ).hexdigest(),
        "files_sha256": digests,
        "items": len(selected),
        "by_type": dict(by_type),
        "languages": dict(sorted(Counter(row["language"] for row in selected).items())),
        "long_input": sum(row["_chars"] >= LONG_INPUT_CHARS for row in selected),
        "licence_flags": dict(
            Counter(
                config.get("licence_flags", {}).get(row["source"], "none")
                for row in selected
            )
        ),
        "tasks": tasks,
        "leak_audit": {
            "verdict_option_surface": audit["verdict_option_surface"],
            "verdict_with_state_surface": audit["verdict_with_state_surface"],
            "groups": {
                name: {
                    "verdict": (entry["combined"].get("option_surface") or {}).get(
                        "verdict", "CLEAN"
                    ),
                    "gain": (entry["combined"].get("option_surface") or {}).get(
                        "gain_over_reference"
                    ),
                    "ci95": (entry["combined"].get("option_surface") or {}).get(
                        "gain_ci95"
                    ),
                }
                for name, entry in audit["groups"].items()
            },
        },
    }
    data = (
        json.dumps(receipt, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    write_new(args.output_dir / "manifest.json", data)
    print(
        json.dumps(
            {
                "items": len(selected),
                "by_type": dict(by_type),
                "leak_audit": receipt["leak_audit"]["verdict_option_surface"],
                "manifest_sha256": hashlib.sha256(data).hexdigest(),
                **digests,
            }
        )
    )
    return 0


def write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("select")
    run.add_argument("--candidates-dir", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--salt-file", type=Path, required=True)
    run.add_argument("--overlap-hits", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--replicates", type=int, default=leak_audit.REPLICATES)
    args = parser.parse_args(argv)
    return select(args)


if __name__ == "__main__":
    raise SystemExit(main())
