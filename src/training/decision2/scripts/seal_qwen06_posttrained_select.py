"""Seal the fixed official Qwen3-0.6B posttrained SELECT-only source contrast."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from training.model.data import file_sha256

SCHEMA = "decision2-qwen06-official-posttrained-select-decision/1"
FAMILIES = {
    "human_goemotions_choice": 200,
    "human_goemotions_noul": 200,
    "pilot_narrative_reading": 130,
    "pilot_open_world_abstention": 40,
    "pilot_string_composition": 40,
    "targeted_quantized_median": 90,
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_valid(row: dict[str, Any]) -> bool:
    kind = row.get("task_type")
    answer = row.get("answer")
    if kind not in {"choice", "noul", "score"} or not isinstance(answer, dict):
        return False
    if answer.get("type") != kind or row.get("prediction_key") is None:
        return False
    if kind == "noul":
        probability = answer.get("noul")
        return (
            row["prediction_key"] in {"true", "false"}
            and isinstance(probability, (float, int))
            and math.isfinite(probability)
            and 0 <= probability <= 1
        )
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or not 2 <= len(probabilities) <= 255:
        return False
    if (
        row["prediction_key"] not in probabilities
        or row.get("gold_key") not in probabilities
    ):
        return False
    if (
        any(
            not isinstance(value, (float, int))
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in probabilities.values()
        )
        or abs(sum(probabilities.values()) - 1) > 1e-5
    ):
        return False
    if kind == "choice":
        return answer.get("choice") == row["prediction_key"]
    if not 2 <= len(probabilities) <= 10 or any(
        not key.isdigit() for key in probabilities
    ):
        return False
    expected = answer.get("score")
    return (
        isinstance(expected, (float, int))
        and math.isfinite(expected)
        and abs(
            expected - sum(int(key) * value for key, value in probabilities.items())
        )
        <= 1e-5
    )


def seal(
    run: Path,
    launch_lock: Path,
    launch_receipt: Path,
    gpu_gate: Path,
    control_metric: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise ValueError("SELECT decision receipt already exists")
    lock = _read(launch_lock)
    receipt = _read(launch_receipt)
    gate = _read(gpu_gate)
    complete = _read(run / "COMPLETE.json")
    best = _read(run / "BEST.json")
    provenance = _read(run / "provenance.json")
    control = _read(control_metric)
    if lock.get("schema") != "decision2-qwen06-official-posttrained-full466-launch/1":
        raise ValueError("Unknown launch lock")
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("lock_sha256") != file_sha256(launch_lock)
        or receipt.get("complete_sha256") != file_sha256(run / "COMPLETE.json")
        or gate.get("status") != "PASS"
        or file_sha256(gpu_gate) != lock["gpu_gate_receipt_sha256"]
    ):
        raise ValueError("Source technical gate or fixed launch incomplete")
    if (
        complete.get("step") != 466
        or complete.get("planned_updates") != 466
        or complete.get("calibration_status") != "untouched"
        or provenance.get("train_examples") != 7455
        or provenance.get("select_examples") != 700
        or provenance.get("train_tokens") != lock["train_tokens"]
        or provenance["model_source"]["files_sha256"]["model.safetensors"]
        != lock["source_weights_sha256"]
    ):
        raise ValueError("Full-arm provenance differs from preregistration")
    if control.get("correct") != 562 or not math.isclose(
        control.get("family_macro_accuracy"), 0.7725925925925926, abs_tol=1e-12
    ):
        raise ValueError("Completed Base shared-head control differs")
    expected_steps = lock["select_steps"]
    actual_steps = sorted(
        int(path.name.removeprefix("select-step-").removesuffix("-metrics.json"))
        for path in run.glob("select-step-*-metrics.json")
    )
    if actual_steps != expected_steps:
        raise ValueError("SELECT milestone roster differs")
    milestones = []
    for step in expected_steps:
        prefix = f"select-step-{step:07d}"
        metric_path = run / f"{prefix}-metrics.json"
        prediction_path = run / f"{prefix}-predictions.jsonl"
        metric = _read(metric_path)
        predictions = [
            json.loads(line) for line in prediction_path.read_text().splitlines()
        ]
        ids = [row["id"] for row in predictions]
        if (
            metric["n"] != 700
            or len(ids) != 700
            or len(set(ids)) != 700
            or sum(bool(row.get("correct")) for row in predictions) != metric["correct"]
            or any(not _prediction_valid(row) for row in predictions)
            or {key: value["n"] for key, value in metric["by_family"].items()}
            != FAMILIES
            or any(
                not math.isfinite(metric[field])
                for field in ("family_macro_accuracy", "family_macro_brier")
            )
        ):
            raise ValueError(f"Invalid native SELECT report at update {step}")
        milestones.append(
            {
                "step": step,
                "correct": metric["correct"],
                "family_macro_accuracy": metric["family_macro_accuracy"],
                "family_macro_brier": metric["family_macro_brier"],
                "family_correct": {
                    key: value["correct"] for key, value in metric["by_family"].items()
                },
                "metric_sha256": file_sha256(metric_path),
                "predictions_sha256": file_sha256(prediction_path),
            }
        )
    selected = max(
        milestones,
        key=lambda row: (
            row["family_macro_accuracy"],
            -row["family_macro_brier"],
            -row["step"],
        ),
    )
    if best["checkpoint"] != f"checkpoint-{selected['step']:07d}":
        raise ValueError("Trainer selected a different checkpoint")
    rule = lock["select_gate"]
    advanced = (
        selected["correct"] >= rule["correct_min"]
        and selected["family_macro_accuracy"] >= rule["family_macro_accuracy_min"]
    )
    result = {
        "schema": SCHEMA,
        "status": "PASS" if advanced else "HOLD",
        "scope": "SELECT only; no CAL, DEV, transfer, formal, public, or HF scoring",
        "launch_lock_sha256": file_sha256(launch_lock),
        "launch_receipt_sha256": file_sha256(launch_receipt),
        "provenance_sha256": file_sha256(run / "provenance.json"),
        "complete_sha256": file_sha256(run / "COMPLETE.json"),
        "control_metric_sha256": file_sha256(control_metric),
        "selected_step": selected["step"],
        "selected_correct": selected["correct"],
        "selected_family_macro_accuracy": selected["family_macro_accuracy"],
        "control_correct": control["correct"],
        "control_family_macro_accuracy": control["family_macro_accuracy"],
        "gate": rule,
        "all_eight_milestones": milestones,
        "full_gpu_wall_seconds": receipt["elapsed_seconds"],
        "full_gpu_hours": receipt["elapsed_seconds"] / 3600,
        "gpu_cap_seconds": lock["max_gpu_seconds"],
    }
    if result["full_gpu_wall_seconds"] > result["gpu_cap_seconds"]:
        raise ValueError("Fixed GPU cap was exceeded")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "run",
        "launch-lock",
        "launch-receipt",
        "gpu-gate",
        "control-metric",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = seal(
        args.run,
        args.launch_lock,
        args.launch_receipt,
        args.gpu_gate,
        args.control_metric,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_step": result["selected_step"],
                "selected_correct": result["selected_correct"],
                "selected_family_macro_accuracy": result[
                    "selected_family_macro_accuracy"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
