"""Score frozen competitor predictions and emit only aggregate comparisons.

Run this on the authorized experiment host after both gold-free prediction
files are sealed. Native scorer reports, including any row detail, remain in
the private run directory. The aggregate file is suitable for a research note.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any

from benchmark.score import evaluate_answer, score_suite
from jev_arena.jevbench_public import _evaluate as evaluate_public
from jev_arena.jevbench_public import score as score_public
from small_competitor_smoke import PANEL_HASHES
from small_decision_competitors import (
    A_REVISION,
    B_REVISION,
    json_bytes,
    sha_file,
)
from transfer.score import evaluate as evaluate_css
from transfer.score import score as score_css

MODEL_IDS = {
    "a": "thefloydd/qwen3-0.6b-rlcd",
    "b": "anthonym21/qwen3-0.6b-rlcd-decision",
}
REVISIONS = {"a": A_REVISION, "b": B_REVISION}
BUCKETS = (512, 1536, 2304, 8192)
REPLICATES = 2000
SEED = 20260927


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(json_bytes(value))


def bucket(tokens: int) -> str:
    for bound in BUCKETS:
        if tokens <= bound:
            return f"<= {bound}"
    return "> 8192"


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * p)
    return ordered[index]


def paired_interval(
    strata: dict[str, list[tuple[bool, bool]]], replicates: int = REPLICATES
) -> dict:
    all_rows = [row for rows in strata.values() for row in rows]
    point = sum(b - a for a, b in all_rows) / len(all_rows)
    rng = random.Random(SEED)
    draws = []
    for _ in range(replicates):
        total = 0
        for rows in strata.values():
            for _ in rows:
                a, b = rows[rng.randrange(len(rows))]
                total += b - a
        draws.append(total / len(all_rows))
    return {
        "n": len(all_rows),
        "b_minus_a": point,
        "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "resampling": "within fixed strata; deterministic seed",
    }


def group_paired_interval(
    groups: dict[str, dict[str, list[tuple[bool, bool]]]],
) -> dict:
    all_rows = [
        pair
        for family in groups.values()
        for pairs in family.values()
        for pair in pairs
    ]
    rng = random.Random(SEED)
    draws = []
    for _ in range(REPLICATES):
        total = 0
        for family in groups.values():
            keys = list(family)
            for _ in keys:
                pairs = family[keys[rng.randrange(len(keys))]]
                total += sum(b - a for a, b in pairs)
        draws.append(total / len(all_rows))
    return {
        "n": len(all_rows),
        "groups": sum(map(len, groups.values())),
        "b_minus_a": sum(b - a for a, b in all_rows) / len(all_rows),
        "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "resampling": "DEV groups within fixed families; deterministic seed",
    }


def verify_frozen(
    arm: str, panel: str, run_dir: Path, expected_panel_sha: str
) -> tuple[list[dict], list[dict]]:
    pred_path = run_dir / "predictions" / f"{arm}-{panel}.jsonl"
    plan_path = run_dir / "preflight" / f"{arm}-{panel}.jsonl"
    pred_manifest = json.loads(
        pred_path.with_suffix(pred_path.suffix + ".manifest.json").read_text()
    )
    plan_manifest = json.loads(
        plan_path.with_suffix(plan_path.suffix + ".manifest.json").read_text()
    )
    if (
        pred_manifest["predictions_sha256"] != sha_file(pred_path)
        or pred_manifest["preflight_sha256"] != sha_file(plan_path)
        or pred_manifest["prompts_sha256"] != expected_panel_sha
        or plan_manifest["prompts_sha256"] != expected_panel_sha
        or pred_manifest["model_hashes"] != plan_manifest["model_hashes"]
        or pred_manifest["revision"] != REVISIONS[arm]
    ):
        raise ValueError(f"{arm}/{panel} prediction freeze differs")
    predictions, plans = read_jsonl(pred_path), read_jsonl(plan_path)
    if [row["id"] for row in predictions] != [row["id"] for row in plans]:
        raise ValueError(f"{arm}/{panel} row order differs")
    if len(predictions) != len({row["id"] for row in predictions}):
        raise ValueError(f"{arm}/{panel} duplicate prediction")
    return predictions, plans


def score(panel_root: Path, run_dir: Path, scorer_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    paths = {
        "dev": (
            panel_root / "runs/dev.gold.jsonl",
            PANEL_HASHES["dev_gold"],
            PANEL_HASHES["dev_prompts"],
        ),
        "css": (
            panel_root / "runs/css-transfer-v1/css-pilot.gold.jsonl",
            PANEL_HASHES["css_gold"],
            PANEL_HASHES["css_prompts"],
        ),
        "public": (
            panel_root / "bench/jevbench-public-231/targets.jsonl",
            PANEL_HASHES["public_targets"],
            PANEL_HASHES["public_prompts"],
        ),
    }
    for path, digest, _ in paths.values():
        if sha_file(path) != digest:
            raise ValueError("gold/target panel hash differs")
    expected_sizes = {"dev": 1600, "css": 1430, "public": 231}
    predictions: dict[str, dict[str, dict[str, dict]]] = {}
    plans: dict[str, dict[str, dict[str, dict]]] = {}
    reports = {}
    for arm in ("a", "b"):
        predictions[arm], plans[arm], reports[arm] = {}, {}, {}
        for panel in ("dev", "css", "public"):
            rows, planned = verify_frozen(arm, panel, run_dir, paths[panel][2])
            if len(rows) != expected_sizes[panel]:
                raise ValueError(f"{arm}/{panel} count differs")
            predictions[arm][panel] = {row["id"]: row for row in rows}
            plans[arm][panel] = {row["id"]: row for row in planned}
            raw_path = run_dir / "predictions" / f"{arm}-{panel}.jsonl"
            if panel == "dev":
                report = score_suite(
                    paths[panel][0],
                    raw_path,
                    MODEL_IDS[arm],
                    REVISIONS[arm],
                    "rocm-native",
                )
            elif panel == "css":
                report = score_css(paths[panel][0], raw_path)
            else:
                identified_path = run_dir / "scores" / f"{arm}-public-identified.jsonl"
                if identified_path.exists():
                    raise FileExistsError(identified_path)
                with identified_path.open("xb") as identified:
                    for row in rows:
                        identified.write(
                            json_bytes(
                                row
                                | {
                                    "model_id": MODEL_IDS[arm],
                                    "model_revision": REVISIONS[arm],
                                }
                            )
                        )
                report = score_public(
                    panel_root / "bench/jevbench-public-231",
                    identified_path,
                    MODEL_IDS[arm],
                    REVISIONS[arm],
                    run_dir / "scores" / f"{arm}-public.json",
                )
            if panel != "public":
                write_once(run_dir / "scores" / f"{arm}-{panel}.json", report)
            reports[arm][panel] = report

    dev_gold = read_jsonl(paths["dev"][0])
    css_gold = read_jsonl(paths["css"][0])
    public_targets = read_jsonl(paths["public"][0])
    if (
        set(predictions["a"]["dev"]) != {r["id"] for r in dev_gold}
        or set(predictions["a"]["css"]) != {r["id"] for r in css_gold}
        or set(predictions["a"]["public"]) != {r["id"] for r in public_targets}
    ):
        raise ValueError("scoring panel IDs differ")
    by_context = {arm: {} for arm in ("a", "b")}
    dev_groups: dict[str, dict[str, list[tuple[bool, bool]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    css_strata: dict[str, list[tuple[bool, bool]]] = collections.defaultdict(list)
    public_strata: dict[str, list[tuple[bool, bool]]] = collections.defaultdict(list)
    for panel, gold_rows in (
        ("dev", dev_gold),
        ("css", css_gold),
        ("public", public_targets),
    ):
        context: dict[str, dict[str, collections.Counter[str]]] = {
            arm: collections.defaultdict(collections.Counter) for arm in ("a", "b")
        }
        for row in gold_rows:
            item_id = row["id"]
            pair = {}
            for arm in ("a", "b"):
                prediction = predictions[arm][panel][item_id]
                answer = next(iter(prediction["answers"].values()), None)
                if panel == "dev":
                    qid, question = next(iter(row["questions"].items()))
                    result = evaluate_answer(question, row["gold"][qid], answer)
                    valid = result["status"] == "ok"
                    correct = valid and bool(result["correct"])
                elif panel == "css":
                    result = evaluate_css(row, prediction)
                    valid = bool(result["valid"])
                    correct = valid and bool(result["correct"])
                else:
                    result = evaluate_public(answer, row)
                    valid = bool(result["valid"])
                    correct = valid and bool(result["correct"])
                pair[arm] = (valid, correct)
                b = bucket(plans[arm][panel][item_id]["input_tokens"])
                context[arm][b]["n"] += 1
                context[arm][b]["valid"] += valid
                context[arm][b]["correct"] += correct
            ab = (pair["a"][1], pair["b"][1])
            if panel == "dev":
                dev_groups[row["family"]][row["group_id"]].append(ab)
            elif panel == "css":
                css_strata[row["task"]].append(ab)
            else:
                public_strata[row["tier"]].append(ab)
        for arm in ("a", "b"):
            by_context[arm][panel] = {
                name: dict(counts) for name, counts in sorted(context[arm].items())
            }

    aggregate = {
        "scope": "same frozen development and public panels; no FINAL or CSS15 scoring",
        "model_revisions": REVISIONS,
        "panel_hashes": PANEL_HASHES,
        "scorer_hashes": {
            name: sha_file(scorer_root / name)
            for name in (
                "benchmark/score.py",
                "benchmark/generate.py",
                "transfer/score.py",
                "transfer/build.py",
                "jev_arena/jevbench_public.py",
            )
        },
        "models": {},
        "paired_b_minus_a": {
            "dev_group_bootstrap": group_paired_interval(dev_groups),
            "css_task_stratified_item_bootstrap": paired_interval(css_strata),
            "public_tier_stratified_item_bootstrap": paired_interval(public_strata),
        },
        "context_buckets": by_context,
        "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": SEED,
    }
    for arm in ("a", "b"):
        dev = reports[arm]["dev"]
        css = reports[arm]["css"]
        public = reports[arm]["public"]
        aggregate["models"][arm] = {
            "dev": {
                "overall": dev["overall"],
                "macro_family_accuracy": dev["macro_family_accuracy"],
                "by_type": dev["by_type"],
                "by_family": dev["by_family"],
                "pairs": dev["pairs"],
                "invalid_reasons": dev["invalid_reasons"],
            },
            "css": {"pilot": css["roles"]["pilot"], "tasks": css["tasks"]},
            "public": {
                key: public[key]
                for key in (
                    "items",
                    "valid",
                    "correct",
                    "accuracy_all",
                    "tier_macro_accuracy",
                    "brier_valid",
                    "ece_pmax_15",
                    "tiers",
                )
            },
            "prediction_hashes": {
                panel: sha_file(run_dir / "predictions" / f"{arm}-{panel}.jsonl")
                for panel in ("dev", "css", "public")
            },
            "score_hashes": {
                panel: sha_file(run_dir / "scores" / f"{arm}-{panel}.json")
                for panel in ("dev", "css", "public")
            },
        }
    write_once(output, aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scorer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.panel_root, args.run_dir, args.scorer_root, args.output)
    print(
        json.dumps(
            {
                "aggregate_sha256": sha_file(args.output),
                "paired_b_minus_a": result["paired_b_minus_a"],
            }
        )
    )


if __name__ == "__main__":
    main()
