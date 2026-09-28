"""Validate absolute versus candidate-relative Score readouts after the seal.

Reads sealed readout logits, the fitted CAL temperatures and, only after the
prediction seal exists, the typed DEV key's Score items. CAL (five-level) is
reported from each run's own CAL logits with the same temperatures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from . import pins
from .train_heads import option_probabilities


def level_distribution(
    record: dict[str, Any], temperatures: dict[str, float], readout: str
) -> list[float]:
    probabilities = option_probabilities(record, temperatures, readout)
    by_level = [0.0] * len(record["keys"])
    for key, p in zip(record["keys"], probabilities):
        by_level[int(key)] = p
    return by_level


def metrics(pairs: list[tuple[list[float], int]]) -> dict[str, Any]:
    n = len(pairs)
    if not n:
        return {"n": 0}
    width = max(len(p) for p, _ in pairs)
    confusion = [[0] * width for _ in range(width)]
    nll = brier = rps = 0.0
    top_bins: list[list[tuple[float, bool]]] = [[] for _ in range(10)]
    class_bins: list[list[list[tuple[float, float]]]] = [
        [[] for _ in range(10)] for _ in range(width)
    ]
    expected_bins: list[list[tuple[float, float]]] = [[] for _ in range(10)]
    absolute_error = 0.0
    for probs, gold in pairs:
        k = len(probs)
        top = max(probs)
        winners = [i for i, p in enumerate(probs) if abs(p - top) <= 1e-8]
        predicted = winners[0] if len(winners) == 1 else -1
        if predicted >= 0:
            confusion[gold][predicted] += 1
        nll += -math.log(max(probs[gold], 1e-12))
        brier += sum((p - float(i == gold)) ** 2 for i, p in enumerate(probs)) / 2
        cdf = 0.0
        for threshold in range(k - 1):
            cdf += probs[threshold]
            rps += (cdf - float(gold <= threshold)) ** 2 / (k - 1)
        top_bins[min(9, int(top * 10))].append((top, predicted == gold))
        for level, p in enumerate(probs):
            class_bins[level][min(9, int(p * 10))].append((p, float(level == gold)))
        expected = sum(i * p for i, p in enumerate(probs))
        absolute_error += abs(expected - gold)
        scale = k - 1
        expected_bins[min(9, int(expected / scale * 10))].append(
            (expected / scale, gold / scale)
        )

    def gap(bins) -> float:
        total = sum(len(b) for b in bins)
        return sum(
            len(b)
            / total
            * abs(sum(x for x, _ in b) / len(b) - sum(y for _, y in b) / len(b))
            for b in bins
            if b
        )

    correct = sum(confusion[i][i] for i in range(width))
    recall = [confusion[i][i] / max(1, sum(confusion[i])) for i in range(width)]
    gold_counts = [sum(1 for _, g in pairs if g == i) for i in range(width)]
    return {
        "n": n,
        "accuracy": correct / n,
        "confusion_gold_by_predicted": confusion,
        "gold_counts": gold_counts,
        "recall_by_level": [
            recall[i] if gold_counts[i] else None for i in range(width)
        ],
        "nll": nll / n,
        "brier": brier / n,
        "rps": rps / n,
        "ece_10": gap([[(c, float(h)) for c, h in b] for b in top_bins]),
        "classwise_ece_10": sum(gap(bins) for bins in class_bins) / width,
        "expected_score_calibration_error": gap(expected_bins),
        "expected_score_mae": absolute_error / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--dev-gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seal = json.loads((args.readout / "SEAL.json").read_text(encoding="utf-8"))
    pins.verify_data("dev", args.dev_gold, pins.GOLD)
    gold: dict[tuple[str, str], int] = {}
    for line in args.dev_gold.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        for qid, answer in item["gold"].items():
            if answer["type"] == "score":
                gold[(item["id"], qid)] = answer["value"]
    report = {"seal_sha256": pins.file_sha256(args.readout / "SEAL.json"), "runs": {}}
    for entry in seal["runs"]:
        folder = args.readout / entry["tag"]
        if (
            pins.file_sha256(folder / "dev.logits.jsonl")
            != entry["files"]["dev.logits.jsonl"]
        ):
            raise SystemExit(f"{entry['tag']}: sealed logits changed")
        run = Path(entry["run_path"])
        temperatures = json.loads(
            (run / "calibration.json").read_text(encoding="utf-8")
        )["temperature_by_readout"]
        dev_rows = [
            json.loads(line)
            for line in (folder / "dev.logits.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        cal_rows = [
            json.loads(line)
            for line in (run / "cal-logits.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        out: dict[str, Any] = {}
        for readout in ("absolute", "relative"):
            dev_pairs, invalid = [], 0
            for record in dev_rows:
                if record["task_type"] != "score":
                    continue
                item_id, qid = record["id"].rsplit("/", 1)
                if not record["valid"]:
                    invalid += 1
                    continue
                dev_pairs.append(
                    (
                        level_distribution(record, temperatures, readout),
                        gold[(item_id, qid)],
                    )
                )
            cal_pairs = [
                (
                    level_distribution(r, temperatures, readout),
                    int(r["keys"][r["label"]]),
                )
                for r in cal_rows
                if r["valid"] and r["task_type"] == "score"
            ]
            out[readout] = {
                "dev": {**metrics(dev_pairs), "invalid": invalid},
                "cal": metrics(cal_pairs),
            }
        dev_abs, dev_rel = out["absolute"]["dev"], out["relative"]["dev"]
        out["absolute_validation"] = {
            "rule": "DEV ECE-10 <= 0.10 and DEV Brier <= relative Score Brier of the same model",
            "passed": bool(dev_abs["n"])
            and dev_abs["ece_10"] <= 0.10
            and dev_abs["brier"] <= dev_rel["brier"],
        }
        report["runs"][entry["tag"]] = out
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
