"""One-shot JevArena-C1 scoring: seal predictions, score them, compare two packages.

    python3 -m v2.eval.sealed.score seal --prompts <prompts.jsonl> --predictions <preds.jsonl> --output <SEAL-C1.json>
    python3 -m v2.eval.sealed.score score --gold <gold.jsonl> --predictions <preds.jsonl> \
        --seal <SEAL-C1.json> --label <name> --output <REPORT-C1.json>
    python3 -m v2.eval.sealed.score compare --gold <gold.jsonl> --left <preds> --right <preds> \
        --left-name A --right-name B --output <PAIRED-C1.json>

`seal` reads prompts and predictions only (never gold): every prompt answered exactly once
from the identical input, with the question keys intact. `score` refuses predictions whose
bytes differ from the seal.

C1 = 100 x the mean over tasks of the task macro-F1 over the task's gold labels. Choice and
Noul use their labels, Score uses its levels, and an invalid or missing answer counts as
wrong. Also reported: per-type means over tasks, per-task macro-F1 and accuracy, quadratic
weighted kappa for Score tasks, and accuracy slices for long inputs (>= 4,000 characters),
non-English items and NC-licensed sources. `compare` draws a paired bootstrap of the C1
difference, resampling source groups within each task (5,000 draws, seed 20260927).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "dev2-sealed-c1-score/1"
LABEL = "independent confirmation (JevArena-C1)"
REPLICATES = 5000
SEED = 20260927
NC_SOURCES = {"hallutruthqa", "innoduel", "wb_reviews"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def input_digest(state: Any, questions: Any) -> str:
    encoded = json.dumps(
        {"state": state, "questions": questions},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_new(path: Path, value: Any) -> str:
    data = (
        json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(data)
    return hashlib.sha256(data).hexdigest()


def seal(args: argparse.Namespace) -> int:
    prompts = {row["id"]: row for row in read_jsonl(args.prompts)}
    seen: set[str] = set()
    identity: dict[str, Counter] = defaultdict(Counter)
    null_answers = 0
    for row in read_jsonl(args.predictions):
        item_id = row.get("id")
        if item_id not in prompts or item_id in seen:
            raise ValueError(f"unknown or duplicate prediction id {item_id!r}")
        prompt = prompts[item_id]
        if row.get("source_input_sha256") != input_digest(
            prompt["state"], prompt["questions"]
        ):
            raise ValueError(f"{item_id} was predicted from different input")
        answers = row.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(prompt["questions"]):
            raise ValueError(f"{item_id}: answer keys differ from question keys")
        null_answers += sum(value is None for value in answers.values())
        for key in ("model_id", "model_revision", "adapter_version", "backend"):
            if key in row:
                identity[key][json.dumps(row[key])] += 1
        seen.add(item_id)
    receipt = {
        "schema": SCHEMA,
        "label": LABEL,
        "gold_read": False,
        "prompts_sha256": sha_file(args.prompts),
        "predictions_sha256": sha_file(args.predictions),
        "items": len(prompts),
        "predicted": len(seen),
        "missing": len(prompts) - len(seen),
        "null_answers": null_answers,
        "identity": {key: dict(counts) for key, counts in identity.items()},
    }
    print(
        json.dumps(
            {
                "seal_sha256": write_new(args.output, receipt),
                "missing": receipt["missing"],
            }
        )
    )
    return 0


def outcomes(gold: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]):
    """Per item: (task, group, source, language, long, gold label, predicted label or None)."""
    from benchmark.score import evaluate_answer

    rows = []
    for item in gold:
        question = item["questions"]["decision"]
        truth = item["gold"]["decision"]
        answer = ((predictions.get(item["id"]) or {}).get("answers") or {}).get(
            "decision"
        )
        predicted = None
        if isinstance(answer, dict):
            result = evaluate_answer(question, truth, answer)
            if result.get("status") == "ok":
                predicted = result.get("point")
        rows.append(
            (
                item["task"],
                f"{item['task']}|{item['group_id']}",
                item["source"],
                item["language"],
                bool(item["long"]),
                truth["value"],
                predicted,
                question["type"],
            )
        )
    return rows


def macro_f1(pairs: list[tuple[Any, Any]]) -> float:
    labels = sorted({json.dumps(g) for g, _ in pairs})
    scores = []
    for label in labels:
        tp = sum(json.dumps(g) == label and json.dumps(p) == label for g, p in pairs)
        fp = sum(
            json.dumps(g) != label and p is not None and json.dumps(p) == label
            for g, p in pairs
        )
        fn = sum(json.dumps(g) == label and json.dumps(p) != label for g, p in pairs)
        scores.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def quadratic_kappa(pairs: list[tuple[int, int | None]]) -> float | None:
    valid = [(g, p) for g, p in pairs if p is not None]
    if len(valid) < 2:
        return None
    levels = sorted({g for g, _ in valid} | {p for _, p in valid})
    if len(levels) < 2:
        return None
    index = {level: i for i, level in enumerate(levels)}
    k = len(levels)
    observed = [[0.0] * k for _ in range(k)]
    for g, p in valid:
        observed[index[g]][index[p]] += 1
    n = len(valid)
    rows = [sum(observed[i]) for i in range(k)]
    cols = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            weight = ((i - j) ** 2) / ((k - 1) ** 2)
            num += weight * observed[i][j]
            den += weight * rows[i] * cols[j] / n
    return 1 - num / den if den else None


def summarize(rows) -> dict[str, Any]:
    by_task: dict[str, list] = defaultdict(list)
    for row in rows:
        by_task[row[0]].append(row)
    tasks = {}
    for task, items in sorted(by_task.items()):
        pairs = [(r[5], r[6]) for r in items]
        tasks[task] = {
            "type": items[0][7],
            "source": items[0][2],
            "items": len(items),
            "valid": sum(r[6] is not None for r in items),
            "macro_f1": macro_f1(pairs),
            "accuracy": sum(r[5] == r[6] for r in items) / len(items),
            "qwk": quadratic_kappa(pairs) if items[0][7] == "score" else None,
        }
    c1 = 100 * sum(t["macro_f1"] for t in tasks.values()) / len(tasks)
    by_type = {}
    for kind in ("choice", "noul", "score"):
        chosen = [t["macro_f1"] for t in tasks.values() if t["type"] == kind]
        by_type[kind] = 100 * sum(chosen) / len(chosen) if chosen else None

    def accuracy(selected) -> dict[str, Any]:
        items = [r for r in rows if selected(r)]
        return {
            "items": len(items),
            "accuracy": (
                sum(r[5] == r[6] for r in items) / len(items) if items else None
            ),
        }

    licence = {
        "nc_tasks_mean_macro_f1": 100
        * _mean([t["macro_f1"] for t in tasks.values() if t["source"] in NC_SOURCES]),
        "permissive_tasks_mean_macro_f1": 100
        * _mean(
            [t["macro_f1"] for t in tasks.values() if t["source"] not in NC_SOURCES]
        ),
    }
    return {
        "c1": c1,
        "by_type": by_type,
        "tasks": tasks,
        "slices": {
            "long_input": accuracy(lambda r: r[4]),
            "short_input": accuracy(lambda r: not r[4]),
            "non_english": accuracy(lambda r: r[3] != "en"),
            "english": accuracy(lambda r: r[3] == "en"),
        },
        "licence_split": licence,
        "items": len(rows),
        "valid": sum(r[6] is not None for r in rows),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def score(args: argparse.Namespace) -> int:
    sealed = json.loads(args.seal.read_text(encoding="utf-8"))
    if sha_file(args.predictions) != sealed["predictions_sha256"]:
        raise ValueError("predictions changed after the seal")
    gold = read_jsonl(args.gold)
    predictions = {row["id"]: row for row in read_jsonl(args.predictions)}
    report = {
        "schema": SCHEMA,
        "label": LABEL,
        "model": args.label,
        "seal_sha256": sha_file(args.seal),
        "gold_sha256": sha_file(args.gold),
        **summarize(outcomes(gold, predictions)),
    }
    digest = write_new(args.output, report)
    print(
        json.dumps(
            {"c1": report["c1"], "by_type": report["by_type"], "report_sha256": digest}
        )
    )
    return 0


def paired(
    gold: list[dict[str, Any]], left: dict, right: dict, replicates: int, seed: int
) -> dict[str, Any]:
    rows_left = outcomes(gold, left)
    rows_right = outcomes(gold, right)
    by_task_group: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(rows_left):
        by_task_group[row[0]][row[1]].append(index)
    point = summarize(rows_left)["c1"] - summarize(rows_right)["c1"]
    rng = random.Random(seed)
    draws = []
    tasks = sorted(by_task_group)
    for _ in range(replicates):
        total = 0.0
        for task in tasks:
            groups = list(by_task_group[task].values())
            picked = [i for _ in groups for i in groups[rng.randrange(len(groups))]]
            total += macro_f1(
                [(rows_left[i][5], rows_left[i][6]) for i in picked]
            ) - macro_f1([(rows_right[i][5], rows_right[i][6]) for i in picked])
        draws.append(100 * total / len(tasks))
    draws.sort()
    return {
        "delta": point,
        "ci95": [
            draws[int(0.025 * (replicates - 1))],
            draws[int(0.975 * (replicates - 1))],
        ],
        "replicates": replicates,
        "seed": seed,
        "unit": "source groups within each task",
    }


def compare(args: argparse.Namespace) -> int:
    gold = read_jsonl(args.gold)
    left = {row["id"]: row for row in read_jsonl(args.left)}
    right = {row["id"]: row for row in read_jsonl(args.right)}
    result = paired(gold, left, right, args.replicates, SEED)
    result.update(
        {
            "schema": SCHEMA,
            "label": LABEL,
            "left": {"name": args.left_name, "predictions_sha256": sha_file(args.left)},
            "right": {
                "name": args.right_name,
                "predictions_sha256": sha_file(args.right),
            },
        }
    )
    write_new(args.output, result)
    print(json.dumps({"delta": result["delta"], "ci95": result["ci95"]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("seal")
    one.add_argument("--prompts", type=Path, required=True)
    one.add_argument("--predictions", type=Path, required=True)
    one.add_argument("--output", type=Path, required=True)
    two = commands.add_parser("score")
    two.add_argument("--gold", type=Path, required=True)
    two.add_argument("--predictions", type=Path, required=True)
    two.add_argument("--seal", type=Path, required=True)
    two.add_argument("--label", required=True)
    two.add_argument("--output", type=Path, required=True)
    three = commands.add_parser("compare")
    three.add_argument("--gold", type=Path, required=True)
    three.add_argument("--left", type=Path, required=True)
    three.add_argument("--right", type=Path, required=True)
    three.add_argument("--left-name", required=True)
    three.add_argument("--right-name", required=True)
    three.add_argument("--replicates", type=int, default=REPLICATES)
    three.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"seal": seal, "score": score, "compare": compare}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
