"""Paired exact-panel diagnostics against pinned Kai/Laya 1.0 receipts.

Only DEV1600 and CSS pilot1430 have verified same-input Kai/Laya rows here.
This reads development gold after the new A/B prediction files were sealed.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
from pathlib import Path

from benchmark.score import evaluate_answer, score_suite
from small_competitor_score import (
    REPLICATES,
    SEED,
    group_paired_interval,
    paired_interval,
    percentile,
    read_jsonl,
    verify_frozen,
    write_once,
)
from small_competitor_smoke import PANEL_HASHES
from small_decision_competitors import sha_file
from transfer.score import evaluate as evaluate_css
from transfer.score import macro_f1
from transfer.score import score as score_css

BASELINES = {
    "kai": {
        "id": "llm-semantic-router/Decision-1.0-Kai-0.6B",
        "revision": "7185f514f54b8f93c55998b1e8f9c5cc67f0d029",
        "dev_sha256": "a75a8f89f5724279f50cadb1b0a2fc091838330f328c597fbdf5401481fec5fb",
        "css_sha256": "18cf4e4d771d1c0ca899dfac1b42f1e906e7a233992282d07283ddd93a6481ef",
    },
    "laya": {
        "id": "convaiinnovations/laya-typed-decisions",
        "revision": "1a793eb568e6718f15941d08f85432581df534e3",
        "dev_sha256": "2633e4c1c421f8d130308a2ae8a84e0322923d09431e97c47989552f3fecd9aa",
        "css_sha256": "b5449120bd205b0a5f8c1c65de229f4181362840abd8343e04fb4642f5cdc42f",
    },
}
PAIRS = (("a", "b"), ("kai", "a"), ("kai", "b"), ("laya", "a"), ("laya", "b"))


def median_f1_draws(
    css_gold: list[dict], predictions: dict[str, dict[str, dict]]
) -> dict:
    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for row in css_gold:
        by_task[row["task"]].append(row)
    rng = random.Random(SEED)
    models = tuple(predictions)
    choices = {
        name: {
            row["id"]: evaluate_css(row, predictions[name][row["id"]])["choice"]
            for row in css_gold
        }
        for name in models
    }
    draws = {name: [] for name in models}
    for _ in range(REPLICATES):
        task_scores = {name: [] for name in models}
        for rows in by_task.values():
            sampled = [rows[rng.randrange(len(rows))] for _ in rows]
            gold_labels = [row["gold"] for row in sampled]
            labels = rows[0]["labels"]
            for name in models:
                selected = [choices[name][row["id"]] for row in sampled]
                task_scores[name].append(macro_f1(gold_labels, selected, labels))
        for name in models:
            draws[name].append(statistics.median(task_scores[name]))
    return draws


def paired(
    panel_root: Path,
    run_dir: Path,
    kai_dev: Path,
    kai_css: Path,
    laya_dev: Path,
    laya_css: Path,
    output: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    dev_gold_path = panel_root / "runs/dev.gold.jsonl"
    css_gold_path = panel_root / "runs/css-transfer-v1/css-pilot.gold.jsonl"
    if (
        sha_file(dev_gold_path) != PANEL_HASHES["dev_gold"]
        or sha_file(css_gold_path) != PANEL_HASHES["css_gold"]
    ):
        raise ValueError("exact panels changed")
    baseline_paths = {
        "kai": {"dev": kai_dev, "css": kai_css},
        "laya": {"dev": laya_dev, "css": laya_css},
    }
    predictions: dict[str, dict[str, dict[str, dict]]] = {
        arm: {} for arm in ("a", "b", "kai", "laya")
    }
    reports = {}
    for arm in ("a", "b"):
        reports[arm] = {}
        for panel in ("dev", "css"):
            rows, _ = verify_frozen(
                arm, panel, run_dir, PANEL_HASHES[f"{panel}_prompts"]
            )
            predictions[arm][panel] = {row["id"]: row for row in rows}
            path = run_dir / "predictions" / f"{arm}-{panel}.jsonl"
            reports[arm][panel] = (
                score_suite(dev_gold_path, path, arm, "pinned", "rocm-native")
                if panel == "dev"
                else score_css(css_gold_path, path)
            )
    for name, files in baseline_paths.items():
        reports[name] = {}
        for panel, path in files.items():
            if sha_file(path) != BASELINES[name][f"{panel}_sha256"]:
                raise ValueError(f"{name}/{panel} source receipt differs")
            rows = read_jsonl(path)
            predictions[name][panel] = {row["id"]: row for row in rows}
            if len(rows) != len(predictions["a"][panel]):
                raise ValueError(f"{name}/{panel} count differs")
            for row in rows:
                if (
                    row.get("model_id") != BASELINES[name]["id"]
                    or row.get("model_revision") != BASELINES[name]["revision"]
                    or row.get("source_input_sha256")
                    != predictions["a"][panel][row["id"]]["source_input_sha256"]
                ):
                    raise ValueError(f"{name}/{panel} model or input differs")
            if panel == "dev":
                reports[name][panel] = score_suite(
                    dev_gold_path,
                    path,
                    BASELINES[name]["id"],
                    BASELINES[name]["revision"],
                    "prior-native",
                )
            else:
                reports[name][panel] = score_css(css_gold_path, path)
    dev_gold, css_gold = read_jsonl(dev_gold_path), read_jsonl(css_gold_path)
    for panel, gold in (("dev", dev_gold), ("css", css_gold)):
        expected = {row["id"] for row in gold}
        if any(set(predictions[name][panel]) != expected for name in predictions):
            raise ValueError(f"{panel} item IDs differ")
    dev_correct: dict[str, dict[str, bool]] = {name: {} for name in predictions}
    css_correct: dict[str, dict[str, bool]] = {name: {} for name in predictions}
    for row in dev_gold:
        qid, question = next(iter(row["questions"].items()))
        for name in predictions:
            answer = predictions[name]["dev"][row["id"]].get("answers", {}).get(qid)
            result = evaluate_answer(question, row["gold"][qid], answer)
            dev_correct[name][row["id"]] = result["status"] == "ok" and bool(
                result["correct"]
            )
    for row in css_gold:
        for name in predictions:
            result = evaluate_css(row, predictions[name]["css"][row["id"]])
            css_correct[name][row["id"]] = bool(result["valid"] and result["correct"])
    f1_draws = median_f1_draws(
        css_gold, {name: values["css"] for name, values in predictions.items()}
    )
    comparisons = {}
    for baseline, candidate in PAIRS:
        dev_groups: dict[str, dict[str, list[tuple[bool, bool]]]] = (
            collections.defaultdict(lambda: collections.defaultdict(list))
        )
        css_tasks: dict[str, list[tuple[bool, bool]]] = collections.defaultdict(list)
        for row in dev_gold:
            item_id = row["id"]
            dev_groups[row["family"]][row["group_id"]].append(
                (dev_correct[baseline][item_id], dev_correct[candidate][item_id])
            )
        for row in css_gold:
            item_id = row["id"]
            css_tasks[row["task"]].append(
                (css_correct[baseline][item_id], css_correct[candidate][item_id])
            )
        delta_draws = [b - a for a, b in zip(f1_draws[baseline], f1_draws[candidate])]
        base_point = reports[baseline]["css"]["roles"]["pilot"][
            "median_task_macro_f1_all"
        ]
        cand_point = reports[candidate]["css"]["roles"]["pilot"][
            "median_task_macro_f1_all"
        ]
        comparisons[f"{candidate}_minus_{baseline}"] = {
            "dev_group_bootstrap": group_paired_interval(dev_groups),
            "css_task_stratified_item_bootstrap": paired_interval(css_tasks),
            "css_median_task_macro_f1": {
                "baseline": base_point,
                "candidate": cand_point,
                "difference": cand_point - base_point,
                "ci95": [
                    percentile(delta_draws, 0.025),
                    percentile(delta_draws, 0.975),
                ],
                "resampling": "CSS items within each fixed pilot task; deterministic seed",
            },
        }
    result = {
        "scope": "exact same DEV1600/CSS pilot1430 prompts and per-row input hashes; historical pinned Kai/Laya receipts",
        "panel_hashes": {
            key: PANEL_HASHES[key]
            for key in ("dev_prompts", "dev_gold", "css_prompts", "css_gold")
        },
        "baseline_identity": BASELINES,
        "baseline_scores": {
            name: {
                "dev": {
                    "correct": reports[name]["dev"]["overall"]["correct_n"],
                    "valid": reports[name]["dev"]["overall"]["valid_n"],
                    "by_type": reports[name]["dev"]["by_type"],
                },
                "css": {
                    "pilot": reports[name]["css"]["roles"]["pilot"],
                    "tasks": reports[name]["css"]["tasks"],
                },
            }
            for name in BASELINES
        },
        "comparisons": comparisons,
        "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": SEED,
    }
    write_once(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "panel_root",
        "run_dir",
        "kai_dev",
        "kai_css",
        "laya_dev",
        "laya_css",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    result = paired(
        args.panel_root,
        args.run_dir,
        args.kai_dev,
        args.kai_css,
        args.laya_dev,
        args.laya_css,
        args.output,
    )
    print(
        json.dumps(
            {
                "output_sha256": sha_file(args.output),
                "comparisons": result["comparisons"],
            }
        )
    )


if __name__ == "__main__":
    main()
