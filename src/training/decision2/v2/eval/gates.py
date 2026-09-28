"""Release gate checks from stored, sealed same-panel predictions (no GPU).

    python3 -m v2.eval.gates paired --left RUN --right RUN --left-name A --right-name B --output OUT
    python3 -m v2.eval.gates types --run RUN --label L --output OUT

`paired` runs the joint v3 paired bootstrap (5,000 draws, seed 20260927) between two run
directories after checking both seals. It reports the composite and per-axis (T typed,
H human transfer) intervals and writes to OUT, never into either run directory.

`types` is the "no decision type collapsed" check on typed FINAL. For each of Choice, Noul
and Score it reports accuracy with a Wilson 95% interval against chance, the predicted
answer distribution (semantic values for Choice, levels for Score) against the gold
distribution, and per-level recall for Score. A type is COLLAPSED when one predicted
answer takes >= 90% of its answers, or when the lower bound of its accuracy interval is
not above chance.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from v2.eval import panels
from v2.eval.same_panel import (
    PAIRED_REPLICATES,
    PAIRED_SEED,
    prediction_path,
    read_jsonl,
    sha_file,
    write_json,
)

TOP_SHARE = 0.9


def verified(run: Path, panel: str) -> Path:
    sealed = json.loads((run / "SEAL.json").read_text(encoding="utf-8"))
    path = prediction_path(run, panel)
    if sha_file(path) != sealed["panels"][panel]["predictions_sha256"]:
        raise ValueError(f"{run}: {panel} predictions changed after the seal")
    return path


def paired(args: argparse.Namespace) -> int:
    from jev_arena.compare_v3 import compare as compare_v3

    result = compare_v3(
        panels.path(args.panel_root, "typed-final", "gold"),
        panels.path(args.panel_root, "css15", "gold"),
        verified(args.left, "typed-final"),
        verified(args.left, "css15"),
        verified(args.right, "typed-final"),
        verified(args.right, "css15"),
        left_name=args.left_name,
        right_name=args.right_name,
        replicates=PAIRED_REPLICATES,
        seed=PAIRED_SEED,
    )
    result["label"] = "post-key same-panel"
    result["runs"] = {"left": str(args.left), "right": str(args.right)}
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "delta": result["point"]["delta"],
                "ci95": result["ci95"],
                "H_ci95": result["axis_ci95"]["H"]["delta"],
                "T_ci95": result["axis_ci95"]["T"]["delta"],
            }
        )
    )
    return 0


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def type_summary(
    gold: list[dict[str, Any]], predictions: dict[str, Any]
) -> dict[str, Any]:
    from benchmark.score import evaluate_answer

    cells: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "valid": 0,
            "correct": 0,
            "chance": 0.0,
            "pred": Counter(),
            "gold": Counter(),
            "level_hits": Counter(),
            "level_n": Counter(),
        }
    )
    for item in gold:
        answers = (predictions.get(item["id"]) or {}).get("answers")
        for key, question in item["questions"].items():
            kind = question["type"]
            truth = item["gold"][key]
            cell = cells[kind]
            cell["n"] += 1
            count = 2 if kind == "noul" else len(question["criteria"])
            cell["chance"] += 1.0 / count
            gold_value = truth.get("semantic_value", truth["value"])
            cell["gold"][str(gold_value)] += 1
            point = None
            if (
                isinstance(answers, dict)
                and key in answers
                and isinstance(answers[key], dict)
            ):
                result = evaluate_answer(question, truth, answers[key])
                if result.get("status") == "ok":
                    cell["valid"] += 1
                    point = result.get("semantic_point")
                    cell["correct"] += bool(result.get("correct"))
            cell["pred"][str(point)] += 1
            if kind == "score":
                cell["level_n"][str(truth["value"])] += 1
                cell["level_hits"][str(truth["value"])] += point == truth["value"]
    out = {}
    for kind, cell in sorted(cells.items()):
        n = cell["n"]
        chance = cell["chance"] / n
        low, high = wilson(cell["correct"], n)
        top_label, top_count = cell["pred"].most_common(1)[0]
        collapsed = []
        if top_count / n >= TOP_SHARE:
            collapsed.append(f"one answer ({top_label}) takes {top_count / n:.0%}")
        if low <= chance:
            collapsed.append("accuracy interval not above chance")
        entry = {
            "slots": n,
            "valid": cell["valid"],
            "accuracy": cell["correct"] / n,
            "accuracy_ci95": [low, high],
            "chance": chance,
            "gold_majority_share": cell["gold"].most_common(1)[0][1] / n,
            "predicted_distinct": len([k for k in cell["pred"] if k != "None"]),
            "predicted_top_share": top_count / n,
            "predicted_distribution": dict(cell["pred"].most_common()),
            "gold_distribution": dict(cell["gold"].most_common()),
            "verdict": "COLLAPSED: " + "; ".join(collapsed) if collapsed else "OK",
        }
        if kind == "score":
            entry["recall_by_level"] = {
                level: cell["level_hits"][level] / cell["level_n"][level]
                for level in sorted(cell["level_n"])
            }
        out[kind] = entry
    return out


def types(args: argparse.Namespace) -> int:
    from benchmark.score import load_jsonl

    gold = load_jsonl(panels.path(args.panel_root, "typed-final", "gold"))
    predictions = {
        row["id"]: row for row in read_jsonl(verified(args.run, "typed-final"))
    }
    result = {
        "schema": "dev2-gate-types/1",
        "label": args.label,
        "run": str(args.run),
        "rule": f"COLLAPSED if one answer >= {TOP_SHARE:.0%} of a type or the Wilson 95% lower bound <= chance",
        "types": type_summary(gold, predictions),
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                k: {
                    "acc": round(v["accuracy"], 3),
                    "chance": round(v["chance"], 3),
                    "top": round(v["predicted_top_share"], 3),
                    "verdict": v["verdict"],
                }
                for k, v in result["types"].items()
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel-root", type=Path, default=panels.DEFAULT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("paired")
    one.add_argument("--left", type=Path, required=True)
    one.add_argument("--right", type=Path, required=True)
    one.add_argument("--left-name", required=True)
    one.add_argument("--right-name", required=True)
    one.add_argument("--output", type=Path, required=True)
    two = commands.add_parser("types")
    two.add_argument("--run", type=Path, required=True)
    two.add_argument("--label", required=True)
    two.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return paired(args) if args.command == "paired" else types(args)


if __name__ == "__main__":
    raise SystemExit(main())
