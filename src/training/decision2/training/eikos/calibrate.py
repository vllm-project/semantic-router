"""Fit type-wise temperatures on hard CAL after LoRA checkpoint selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from training.eikos.data import native_question
from training.eikos.io import atomic_json, atomic_jsonl
from training.eikos.native import load_decider, selected_checkpoint
from training.model.data import file_sha256, load_partition


def probabilities(answer: dict[str, Any], keys: list[str]) -> list[float]:
    if answer["type"] == "noul":
        p = float(answer["noul"])
        mapping = {"yes": p, "no": 1 - p}
    else:
        mapping = answer["probabilities"]
    values = [float(mapping[key]) for key in keys]
    if (
        any(not math.isfinite(value) or value < 0 for value in values)
        or sum(values) <= 0
    ):
        raise ValueError(
            "Calibration requires nonnegative, finite native probabilities"
        )
    return values


def reweight(values: list[float], temperature: float) -> list[float]:
    logits = [math.log(max(p, 1e-30)) / temperature for p in values]
    maximum = max(logits)
    exps = [math.exp(value - maximum) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def objective(log_temperature: float, rows: list[dict[str, Any]]) -> float:
    temperature = math.exp(log_temperature)
    nll = 0.0
    for row in rows:
        p = reweight(row["probabilities"], temperature)
        nll -= math.log(max(p[row["label"]], 1e-30))
    return nll / len(rows) + 0.02 * log_temperature**2


def fit(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 30:
        raise ValueError(
            "Temperature fit needs at least 30 labeled CAL rows per task type"
        )
    left, right = math.log(0.4), math.log(5.0)
    ratio = (math.sqrt(5) - 1) / 2
    a = right - ratio * (right - left)
    b = left + ratio * (right - left)
    fa, fb = objective(a, rows), objective(b, rows)
    for _ in range(64):
        if fa < fb:
            right, b, fb = b, a, fa
            a = right - ratio * (right - left)
            fa = objective(a, rows)
        else:
            left, a, fa = a, b, fb
            b = left + ratio * (right - left)
            fb = objective(b, rows)
    return math.exp((left + right) / 2)


def summarize(
    rows: list[dict[str, Any]], temperatures: dict[str, float]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty CAL partition")
    by_type = {}
    for kind in ("choice", "noul", "score"):
        subset = [row for row in rows if row["task_type"] == kind]
        if not subset:
            continue
        brier = nll = correct = 0.0
        for row in subset:
            p = reweight(row["probabilities"], temperatures[kind])
            correct += max(range(len(p)), key=p.__getitem__) == row["label"]
            brier += (
                sum(
                    (value - float(i == row["label"])) ** 2 for i, value in enumerate(p)
                )
                / 2
            )
            nll -= math.log(max(p[row["label"]], 1e-30))
        by_type[kind] = {
            "n": len(subset),
            "accuracy": correct / len(subset),
            "brier": brier / len(subset),
            "nll": nll / len(subset),
        }
    return {
        "n": len(rows),
        "accuracy": sum(v["accuracy"] * v["n"] for v in by_type.values()) / len(rows),
        "brier": sum(v["brier"] * v["n"] for v in by_type.values()) / len(rows),
        "nll": sum(v["nll"] * v["n"] for v in by_type.values()) / len(rows),
        "by_type": by_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selected = selected_checkpoint(args.run, args.model_path)
    provenance = json.loads((args.run / "provenance.json").read_text(encoding="utf-8"))
    if file_sha256(args.cal) != provenance["data_sha256"]["cal_audited_only"]:
        raise ValueError("Hard CAL does not match the trainer's audited partition")
    rows = load_partition(args.cal, "cal")
    native = load_decider(
        args.model_path,
        selected["adapter"],
        args.model_path / "calib.json",
        device=args.device,
    )
    import decision_core

    collected = []
    for index, row in enumerate(rows):
        question, gold = native_question(row)
        keys = [key for key, _ in decision_core.options_of(question)]
        if row["task_type"] == "noul":
            gold = {"true": "yes", "false": "no"}[gold]
        answer, tokens = native.decide(row["state"], question)
        collected.append(
            {
                "id": row["id"],
                "task_type": row["task_type"],
                "family": row["family"],
                "keys": keys,
                "label": keys.index(gold),
                "probabilities": probabilities(answer, keys),
                "input_sha256": row["input_sha256"],
                "input_tokens": tokens,
            }
        )
        if (index + 1) % 100 == 0:
            print(
                json.dumps({"cal_collected": index + 1, "total": len(rows)}), flush=True
            )
    args.output.mkdir(parents=True)
    atomic_jsonl(args.output / "cal-native-predictions.jsonl", collected)
    temps = {
        kind: fit([row for row in collected if row["task_type"] == kind])
        for kind in ("choice", "noul", "score")
    }
    log_choice = math.log(temps["choice"])
    calibration = {
        "b": log_choice,
        "w": [
            0.0,
            0.0,
            math.log(temps["noul"]) - log_choice,
            math.log(temps["score"]) - log_choice,
        ],
        "features": ["log_ntok", "log_nopts", "noul", "score"],
        "mode": "hard-CAL-typewise-regularized-NLL-v1",
    }
    atomic_json(args.output / "calib.json", calibration)
    atomic_json(
        args.output / "report.json",
        {
            "checkpoint": selected["name"],
            "adapter_weights_sha256": selected["adapter_weights_sha256"],
            "cal_sha256": file_sha256(args.cal),
            "predictions_sha256": file_sha256(
                args.output / "cal-native-predictions.jsonl"
            ),
            "calibration_sha256": file_sha256(args.output / "calib.json"),
            "temperatures": temps,
            "raw": summarize(collected, dict.fromkeys(temps, 1.0)),
            "calibrated": summarize(collected, temps),
            "selection_metrics": selected["selection_metrics"],
            "native_selection_metrics": selected["native_selection_metrics"],
        },
    )
    print(
        json.dumps(
            {
                "checkpoint": selected["name"],
                "temperatures": temps,
                "report": str(args.output / "report.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
