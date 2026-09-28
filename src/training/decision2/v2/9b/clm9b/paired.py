"""Paired bootstrap of development-proxy differences between sealed readouts.

Per-answer correctness and CSS choices come from the unchanged scorers'
functions. Typed items are resampled within family and CSS items within task,
the same indices for both models, so each draw recomputes T (family-macro
accuracy), H (task-median macro-F1) and the proxy exactly as the reports do.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import pins


def typed_outcomes(
    gold: dict[str, Any], predictions: dict[str, Any]
) -> dict[str, list[bool]]:
    from benchmark.score import evaluate_answer

    by_family: dict[str, list[bool]] = defaultdict(list)
    for item_id in sorted(gold):
        item = gold[item_id]
        prediction = predictions.get(item_id) or {}
        answers = prediction.get("answers") or {}
        for key, question in item["questions"].items():
            result = (
                evaluate_answer(question, item["gold"][key], answers.get(key))
                if key in answers
                else {"status": "missing"}
            )
            by_family[item["family"]].append(
                result.get("status") == "ok" and bool(result.get("correct"))
            )
    return dict(by_family)


def css_outcomes(gold: dict[str, Any], predictions: dict[str, Any]):
    from transfer.score import evaluate

    by_task: dict[str, dict[str, list]] = defaultdict(
        lambda: {"gold": [], "choice": [], "labels": None}
    )
    for item_id in sorted(gold):
        row = gold[item_id]
        if row["role"] != "pilot":
            continue
        result = evaluate(row, predictions.get(item_id))
        entry = by_task[row["task"]]
        entry["gold"].append(row["gold"])
        entry["choice"].append(result["choice"] if result.get("valid") else None)
        entry["labels"] = row["labels"]
    return dict(by_task)


def proxy(
    typed: dict[str, list[bool]], css, typed_index, css_index
) -> tuple[float, float, float]:
    from transfer.score import macro_f1

    t = statistics.mean(
        sum(typed[family][i] for i in typed_index[family]) / len(typed_index[family])
        for family in typed
    )
    h = statistics.median(
        macro_f1(
            [css[task]["gold"][i] for i in css_index[task]],
            [css[task]["choice"][i] for i in css_index[task]],
            css[task]["labels"],
        )
        for task in css
    )
    return t, h, 100 * math.sqrt(t * h)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--dev-gold", type=Path, required=True)
    parser.add_argument("--css-gold", type=Path, required=True)
    parser.add_argument(
        "--pair", action="append", required=True, help="TREATMENT_TAG:CONTROL_TAG"
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from benchmark.score import load_jsonl
    from transfer.score import read_jsonl

    pins.verify_data("dev", args.dev_gold, pins.GOLD)
    pins.verify_data("css_pilot", args.css_gold, pins.GOLD)
    seal = json.loads((args.readout / "SEAL.json").read_text(encoding="utf-8"))
    sealed = {entry["tag"]: entry for entry in seal["runs"]}
    typed_gold, css_gold = load_jsonl(args.dev_gold), read_jsonl(args.css_gold)
    cache: dict[str, tuple] = {}

    def outcomes(tag: str):
        if tag not in cache:
            folder = args.readout / tag
            for name in ("dev.predictions.jsonl", "css_pilot.predictions.jsonl"):
                if pins.file_sha256(folder / name) != sealed[tag]["files"][name]:
                    raise SystemExit(f"{tag}/{name} changed after the seal")
            cache[tag] = (
                typed_outcomes(
                    typed_gold, load_jsonl(folder / "dev.predictions.jsonl")
                ),
                css_outcomes(
                    css_gold, read_jsonl(folder / "css_pilot.predictions.jsonl")
                ),
            )
        return cache[tag]

    report = {
        "seal_sha256": pins.file_sha256(args.readout / "SEAL.json"),
        "draws": args.draws,
        "pairs": {},
    }
    for pair in args.pair:
        treatment, control = pair.split(":", 1)
        (t_typed, t_css), (c_typed, c_css) = outcomes(treatment), outcomes(control)
        full_typed = {
            family: list(range(len(values))) for family, values in t_typed.items()
        }
        full_css = {
            task: list(range(len(values["gold"]))) for task, values in t_css.items()
        }
        point_t, point_c = proxy(t_typed, t_css, full_typed, full_css), proxy(
            c_typed, c_css, full_typed, full_css
        )
        rng = random.Random(20260928)
        deltas = []
        for _ in range(args.draws):
            typed_index = {
                f: [rng.randrange(len(v)) for _ in v] for f, v in full_typed.items()
            }
            css_index = {
                t: [rng.randrange(len(v)) for _ in v] for t, v in full_css.items()
            }
            deltas.append(
                proxy(t_typed, t_css, typed_index, css_index)[2]
                - proxy(c_typed, c_css, typed_index, css_index)[2]
            )
        deltas.sort()
        report["pairs"][pair] = {
            "treatment": {"T": point_t[0], "H": point_t[1], "proxy": point_t[2]},
            "control": {"T": point_c[0], "H": point_c[1], "proxy": point_c[2]},
            "delta": point_t[2] - point_c[2],
            "ci95": [
                deltas[int(0.025 * len(deltas))],
                deltas[int(0.975 * len(deltas)) - 1],
            ],
        }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(args.output), "sha256": pins.file_sha256(args.output)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
