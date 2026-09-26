"""Audit cross-process Eikos CSS repeatability without selecting on CSS gold.

The first completed standalone-package prediction file is the primary scored
receipt. Repeats characterize runtime variance only; they cannot replace the
primary or select a checkpoint. The two independent LoRA runs are diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from training.eikos.io import atomic_json


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"Duplicate prediction ID in {path}")
            rows[row["id"]] = row
    if len(rows) != 1430:
        raise ValueError(f"CSS pilot needs all 1430 rows: {path}")
    return rows


def answer_pair(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, float]:
    if left["type"] != right["type"]:
        raise ValueError("Repeat changed answer type")
    kind = left["type"]
    if kind == "choice":
        changed = left["choice"] != right["choice"]
        a, b = left["probabilities"], right["probabilities"]
    elif kind == "score":
        changed = left["native_score"] != right["native_score"]
        a, b = left["probabilities"], right["probabilities"]
    elif kind in {"noul", "boolean"}:
        changed = left["value"] != right["value"]
        a, b = {"yes": left["probability"]}, {"yes": right["probability"]}
    else:
        raise ValueError("Unsupported repeated answer type")
    if set(a) != set(b):
        raise ValueError("Repeat changed offered options")
    drift = max(abs(float(a[key]) - float(b[key])) for key in a)
    if not math.isfinite(drift):
        raise ValueError("Nonfinite repeated probability")
    return changed, drift


def compare(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("Repeat changed CSS row IDs")
    drift, changed = [], []
    for item_id in left:
        a, b = left[item_id], right[item_id]
        if a["source_input_sha256"] != b["source_input_sha256"]:
            raise ValueError(f"Repeat changed input: {item_id}")
        if a.get("usage", {}).get("input_tokens") != b.get("usage", {}).get(
            "input_tokens"
        ):
            raise ValueError(f"Repeat changed token count: {item_id}")
        if set(a["answers"]) != set(b["answers"]) or len(a["answers"]) != 1:
            raise ValueError(f"Repeat changed question set: {item_id}")
        key = next(iter(a["answers"]))
        different, d = answer_pair(a["answers"][key], b["answers"][key])
        drift.append(d)
        if different:
            changed.append(item_id)
    ordered = sorted(drift)
    return {
        "items": len(left),
        "categorical_mismatch_n": len(changed),
        "categorical_mismatch_ids": changed,
        "max_option_probability_drift": max(drift),
        "p99_option_probability_drift": ordered[math.ceil(0.99 * len(ordered)) - 1],
        "mean_option_probability_drift": statistics.mean(drift),
        "rows_over_0_005": sum(value > 0.005 for value in drift),
        "rows_over_0_02": sum(value > 0.02 for value in drift),
    }


def verified_score(predictions: Path, report: Path) -> dict[str, Any]:
    data = json.loads(report.read_text(encoding="utf-8"))
    if data.get("predictions_sha256") != sha(predictions):
        raise ValueError("Transfer score is not bound to the prediction bytes")
    role = data.get("roles", {}).get("pilot", {})
    if role.get("items") != 1430 or role.get("valid_items") != 1430:
        raise ValueError("Repeat score lacks a full valid CSS pilot")
    accuracy = role["micro_accuracy_all"]
    correct = round(accuracy * 1430)
    if abs(accuracy - correct / 1430) > 1e-12:
        raise ValueError("CSS score accuracy does not match an integer numerator")
    return {
        "report_sha256": sha(report),
        "predictions_sha256": sha(predictions),
        "correct": correct,
        "items": 1430,
        "accuracy_all": accuracy,
    }


def receipt(
    *,
    package: Path,
    package_primary: Path,
    package_primary_score: Path,
    package_repeat: Path,
    package_repeat_score: Path,
    lora_primary: Path,
    lora_primary_score: Path,
    lora_repeat: Path,
    lora_repeat_score: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    paths = {
        "package_primary": (package_primary, package_primary_score),
        "package_repeat": (package_repeat, package_repeat_score),
        "lora_primary": (lora_primary, lora_primary_score),
        "lora_repeat": (lora_repeat, lora_repeat_score),
    }
    rows = {name: load(pred) for name, (pred, _) in paths.items()}
    scores = {
        name: verified_score(pred, report) for name, (pred, report) in paths.items()
    }
    model_sha = sha(package / "SHA256SUMS")
    package_cal = None
    for name in ("package_primary", "package_repeat"):
        manifest = Path(str(paths[name][0]) + ".manifest.json")
        m = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            m.get("model_sha256") != model_sha
            or m.get("predictions_sha256") != scores[name]["predictions_sha256"]
            or m.get("counts", {}).get("items") != 1430
        ):
            raise ValueError("Package prediction manifest changed model or output")
        scores[name]["manifest_sha256"] = sha(manifest)
        if package_cal is None:
            package_cal = m["calibration_sha256"]
        if m["calibration_sha256"] != package_cal:
            raise ValueError("Package repeat changed calibration")
        if any(
            row.get("model_sha256") != model_sha
            or row.get("calibration_sha256") != package_cal
            for row in rows[name].values()
        ):
            raise ValueError("Package repeat mixed model or calibration")
    lora_adapter = next(iter(rows["lora_primary"].values()))["adapter_weights_sha256"]
    for name in ("lora_primary", "lora_repeat"):
        if any(
            row.get("adapter_weights_sha256") != lora_adapter
            or row.get("calibration_sha256") != package_cal
            for row in rows[name].values()
        ):
            raise ValueError("LoRA repeat changed adapter or calibration")
    comparisons = {
        "package_repeat": compare(rows["package_primary"], rows["package_repeat"]),
        "lora_repeat": compare(rows["lora_primary"], rows["lora_repeat"]),
        "primary_lora_vs_package": compare(
            rows["lora_primary"], rows["package_primary"]
        ),
    }
    report = {
        "schema_version": "decision2-eikos-cross-process-repeatability/1",
        "scope": "public CSS pilot diagnostics; no final gold or checkpoint selection",
        "frozen_primary": "package_primary",
        "model_sha256": model_sha,
        "calibration_sha256": package_cal,
        "selected_lora_adapter_sha256": lora_adapter,
        "runs": scores,
        "comparisons": comparisons,
        "package_repeat_correct_range": [
            min(
                scores[name]["correct"]
                for name in ("package_primary", "package_repeat")
            ),
            max(
                scores[name]["correct"]
                for name in ("package_primary", "package_repeat")
            ),
        ],
        "all_four_correct_range": [
            min(score["correct"] for score in scores.values()),
            max(score["correct"] for score in scores.values()),
        ],
        "interpretation": "Cross-process BF16/ROCm native readout variance observed; use frozen primary package receipt for reported score, not the highest repeat or same-process parity alone",
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "package",
        "package-primary",
        "package-primary-score",
        "package-repeat",
        "package-repeat-score",
        "lora-primary",
        "lora-primary-score",
        "lora-repeat",
        "lora-repeat-score",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = receipt(
        package=args.package,
        package_primary=args.package_primary,
        package_primary_score=args.package_primary_score,
        package_repeat=args.package_repeat,
        package_repeat_score=args.package_repeat_score,
        lora_primary=args.lora_primary,
        lora_primary_score=args.lora_primary_score,
        lora_repeat=args.lora_repeat,
        lora_repeat_score=args.lora_repeat_score,
        output=args.output,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "model_sha256",
                    "package_repeat_correct_range",
                    "all_four_correct_range",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
