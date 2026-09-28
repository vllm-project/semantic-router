"""Score HT-DEV v1 (development readout): seal predictions, score them, compare two runs.

    python3 -m v2.eval.htdev.score seal --prompts <ht-dev.prompts.jsonl> --predictions <preds> --output <SEAL-HTDEV.json>
    python3 -m v2.eval.htdev.score score --gold <ht-dev.gold.jsonl> --predictions <preds> \
        --seal <SEAL-HTDEV.json> --label <name> --output <REPORT-HTDEV.json>
    python3 -m v2.eval.htdev.score compare --gold <gold> --left <preds> --right <preds> \
        --left-name A --right-name B --output <PAIRED-HTDEV.json>

`seal` reads prompts and predictions only (never gold); `score` refuses predictions
whose bytes differ from the seal. Per task: macro-F1 over the task's gold labels, a
missing or invalid answer counting as wrong (C1 scorer semantics, reused from
`v2.eval.sealed.score`). Primary: H_dev = the median over tasks of task macro-F1 (the
functional of formal H, as a fraction). Secondary: task mean, per-type means, quadratic
weighted kappa for Score tasks. Panel noise: item bootstrap within tasks (2,000 draws,
seed 20260929) of H_dev. `compare` gives the paired bootstrap of the H_dev difference,
resampling source groups within each task. Labelled "development readout": never a
release score.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from v2.eval.sealed.score import (
    input_digest,
    macro_f1,
    outcomes,
    read_jsonl,
    sha_file,
    summarize,
    write_new,
)

SCHEMA = "dev2-htdev-score/1"
LABEL = "development readout (HT-DEV v1)"
REPLICATES = 2000
SEED = 20260929
TYPES = ("choice", "noul", "score")


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
    digest = write_new(args.output, receipt)
    print(json.dumps({"seal_sha256": digest, "missing": receipt["missing"]}))
    return 0


def by_task(rows) -> dict[str, list[int]]:
    tasks: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        tasks[row[0]].append(index)
    return tasks


def task_scores(rows, picks: dict[str, list[int]]) -> dict[str, float]:
    return {
        task: macro_f1([(rows[i][5], rows[i][6]) for i in indices])
        for task, indices in picks.items()
    }


def headline(summary: dict[str, Any]) -> dict[str, Any]:
    tasks = summary["tasks"]
    values = [t["macro_f1"] for t in tasks.values()]
    kinds = {}
    for kind in TYPES:
        chosen = [t["macro_f1"] for t in tasks.values() if t["type"] == kind]
        kinds[kind] = statistics.fmean(chosen) if chosen else None
    kappas = [t["qwk"] for t in tasks.values() if t["type"] == "score"]
    return {
        "H_dev": statistics.median(values) if values else None,
        "task_mean": statistics.fmean(values) if values else None,
        "by_type_mean": kinds,
        "score_qwk_mean": (
            statistics.fmean([k for k in kappas if k is not None])
            if any(k is not None for k in kappas)
            else None
        ),
    }


def item_bootstrap(rows, replicates: int, seed: int) -> dict[str, Any]:
    tasks = by_task(rows)
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        picks = {
            task: [indices[rng.randrange(len(indices))] for _ in indices]
            for task, indices in sorted(tasks.items())
        }
        draws.append(statistics.median(task_scores(rows, picks).values()))
    ordered = sorted(draws)
    return {
        "sd": statistics.stdev(draws) if len(draws) > 1 else 0.0,
        "ci95": [
            ordered[int(0.025 * (replicates - 1))],
            ordered[int(0.975 * (replicates - 1))],
        ],
        "replicates": replicates,
        "seed": seed,
        "unit": "items within each task",
    }


def report(gold: list[dict[str, Any]], predictions: dict, replicates: int) -> dict:
    rows = outcomes(gold, predictions)
    summary = summarize(rows)
    summary.pop("c1", None)
    summary.pop("licence_split", None)
    return {
        **headline(summary),
        "H_dev_bootstrap": item_bootstrap(rows, replicates, SEED),
        "tasks": summary["tasks"],
        "task_count": len(summary["tasks"]),
        "items": summary["items"],
        "valid": summary["valid"],
        "slices": summary["slices"],
    }


def score(args: argparse.Namespace) -> int:
    sealed = json.loads(args.seal.read_text(encoding="utf-8"))
    if sha_file(args.predictions) != sealed["predictions_sha256"]:
        raise ValueError("predictions changed after the seal")
    gold = read_jsonl(args.gold)
    predictions = {row["id"]: row for row in read_jsonl(args.predictions)}
    result = {
        "schema": SCHEMA,
        "label": LABEL,
        "model": args.label,
        "seal_sha256": sha_file(args.seal),
        "gold_sha256": sha_file(args.gold),
        **report(gold, predictions, args.replicates),
    }
    digest = write_new(args.output, result)
    print(json.dumps({"H_dev": result["H_dev"], "report_sha256": digest}))
    return 0


def paired(
    gold: list[dict[str, Any]], left: dict, right: dict, replicates: int, seed: int
) -> dict[str, Any]:
    rows_left = outcomes(gold, left)
    rows_right = outcomes(gold, right)
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows_left):
        groups[row[0]][row[1]].append(index)
    whole = {task: [i for g in gs.values() for i in g] for task, gs in groups.items()}
    point = statistics.median(
        task_scores(rows_left, whole).values()
    ) - statistics.median(task_scores(rows_right, whole).values())
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        picks = {}
        for task in sorted(groups):
            clusters = list(groups[task].values())
            picks[task] = [
                i for _ in clusters for i in clusters[rng.randrange(len(clusters))]
            ]
        draws.append(
            statistics.median(task_scores(rows_left, picks).values())
            - statistics.median(task_scores(rows_right, picks).values())
        )
    ordered = sorted(draws)
    return {
        "delta_H_dev": point,
        "ci95": [
            ordered[int(0.025 * (replicates - 1))],
            ordered[int(0.975 * (replicates - 1))],
        ],
        "sd": statistics.stdev(draws) if len(draws) > 1 else 0.0,
        "p_left_better": sum(d > 0 for d in draws) / replicates,
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
    print(json.dumps({"delta_H_dev": result["delta_H_dev"], "ci95": result["ci95"]}))
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
    two.add_argument("--replicates", type=int, default=REPLICATES)
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
