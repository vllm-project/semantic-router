"""Score HT-DEV v2 (development readout): the held-out CSS15 parallel form.

    python3 -m v2.eval.htdev.score seal --prompts <ht-dev2.prompts.jsonl> --predictions <preds> --output <SEAL-HTDEV2.json>
    python3 -m v2.eval.htdev2.score score --gold <ht-dev2.gold.jsonl> --predictions <preds> \
        --seal <SEAL-HTDEV2.json> --label <name> --output <REPORT-HTDEV2.json>
    python3 -m v2.eval.htdev2.score compare --gold <gold> --left <preds> --right <preds> \
        --left-name A --right-name B --output <PAIRED-HTDEV2.json>

Per task: macro-F1 over the task's labels with the formal CSS scorer's semantics
(`transfer.score`: invalid or missing counts as wrong). Primary: H_dev2 = mean over tasks
of task macro-F1 (prereg `records/htdev2-prereg-2026-09-30.md` §4); secondary: the median.
Noise: item bootstrap within tasks (2,000 draws, seed 20260930); `compare` applies the same
resampled items to both runs. Labelled "development readout": never a release score.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from transfer.score import evaluate, macro_f1, task_metrics
from v2.eval.same_panel import sha_file, utc_now, write_json

SCHEMA = "dev2-htdev2-score/1"
LABEL = "development readout (HT-DEV v2); never a release score"
REPLICATES = 2000
SEED = 20260930


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def outcomes(gold: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]):
    by_task: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in gold:
        prediction = predictions.get(row["id"])
        if (
            prediction is not None
            and prediction.get("source_input_sha256") != row["input_sha256"]
        ):
            raise ValueError(f"{row['id']}: prediction input hash differs from gold")
        by_task[row["task"]].append((row, evaluate(row, prediction)))
    return dict(sorted(by_task.items()))


def headline(task_f1: dict[str, float]) -> dict[str, float]:
    values = list(task_f1.values())
    return {
        "H_dev2": statistics.fmean(values),
        "H_dev2_median": statistics.median(values),
    }


def resampled_f1(pairs, picks) -> float:
    rows = [pairs[i] for i in picks]
    return macro_f1(
        [g["gold"] for g, _ in rows],
        [r["choice"] for _, r in rows],
        rows[0][0]["labels"],
    )


def bootstrap(tasks, replicates: int, seed: int, other=None) -> dict[str, Any]:
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        f1, f1_other = {}, {}
        for task, pairs in tasks.items():
            picks = [rng.randrange(len(pairs)) for _ in pairs]
            f1[task] = resampled_f1(pairs, picks)
            if other is not None:
                f1_other[task] = resampled_f1(other[task], picks)
        head = headline(f1)
        if other is not None:
            head_other = headline(f1_other)
            head = {k: head[k] - head_other[k] for k in head}
        for key, value in head.items():
            draws[key].append(value)
    out = {}
    for key, values in draws.items():
        ordered = sorted(values)
        out[key] = {
            "sd": statistics.pstdev(values),
            "ci95": [
                ordered[int(0.025 * (len(ordered) - 1))],
                ordered[int(0.975 * (len(ordered) - 1))],
            ],
            "p_le_0": sum(v <= 0 for v in values) / len(values),
        }
    return {"replicates": replicates, "seed": seed, **out}


def report(gold, predictions, replicates: int = REPLICATES) -> dict[str, Any]:
    tasks = outcomes(gold, predictions)
    metrics = {
        task: task_metrics([g for g, _ in pairs], [r for _, r in pairs])
        for task, pairs in tasks.items()
    }
    task_f1 = {task: m["macro_f1_all"] for task, m in metrics.items()}
    return {
        "schema": SCHEMA,
        "label": LABEL,
        **headline(task_f1),
        "tasks": {
            task: {
                k: m[k]
                for k in (
                    "n",
                    "valid_n",
                    "invalid_or_missing_n",
                    "accuracy_all",
                    "macro_f1_all",
                )
            }
            for task, m in metrics.items()
        },
        "items": sum(m["n"] for m in metrics.values()),
        "valid": sum(m["valid_n"] for m in metrics.values()),
        "bootstrap": bootstrap(tasks, replicates, SEED),
    }


def score(args: argparse.Namespace) -> int:
    seal = json.loads(args.seal.read_text(encoding="utf-8"))
    if seal["predictions_sha256"] != sha_file(args.predictions):
        raise ValueError("predictions changed after the seal")
    predictions = {row["id"]: row for row in read_jsonl(args.predictions)}
    result = {
        "created_utc": utc_now(),
        "run_label": args.label,
        "gold_sha256": sha_file(args.gold),
        "predictions_sha256": seal["predictions_sha256"],
        **report(read_jsonl(args.gold), predictions, args.replicates),
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("run_label", "H_dev2", "H_dev2_median", "items", "valid")
            }
        )
    )
    return 0


def compare(args: argparse.Namespace) -> int:
    gold = read_jsonl(args.gold)
    left = outcomes(gold, {r["id"]: r for r in read_jsonl(args.left)})
    right = outcomes(gold, {r["id"]: r for r in read_jsonl(args.right)})
    point = {
        k: v - w
        for (k, v), w in zip(
            headline(
                {t: resampled_f1(p, range(len(p))) for t, p in left.items()}
            ).items(),
            headline(
                {t: resampled_f1(p, range(len(p))) for t, p in right.items()}
            ).values(),
        )
    }
    result = {
        "schema": SCHEMA + "/paired",
        "label": LABEL,
        "created_utc": utc_now(),
        "left": args.left_name,
        "right": args.right_name,
        "delta": point,
        "paired_bootstrap": bootstrap(left, args.replicates, SEED, other=right),
        "files": {
            "gold": sha_file(args.gold),
            "left": sha_file(args.left),
            "right": sha_file(args.right),
        },
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {"delta": point, "ci95": result["paired_bootstrap"]["H_dev2"]["ci95"]}
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("score")
    one.add_argument("--gold", type=Path, required=True)
    one.add_argument("--predictions", type=Path, required=True)
    one.add_argument("--seal", type=Path, required=True)
    one.add_argument("--label", required=True)
    one.add_argument("--replicates", type=int, default=REPLICATES)
    one.add_argument("--output", type=Path, required=True)
    two = commands.add_parser("compare")
    two.add_argument("--gold", type=Path, required=True)
    two.add_argument("--left", type=Path, required=True)
    two.add_argument("--right", type=Path, required=True)
    two.add_argument("--left-name", required=True)
    two.add_argument("--right-name", required=True)
    two.add_argument("--replicates", type=int, default=REPLICATES)
    two.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return score(args) if args.command == "score" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
