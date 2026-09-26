"""Bounded Qwen3.8-27B hard-CAL transport diagnostic, separate from run CAL.

The original completed run and its audited CAL700 are immutable. Only fresh
Noul/Score hard-CAL rows fit diagnostic temperatures; the Choice temperature
is copied from the original fit. Transformed DEV predictions are exploratory.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from research.calibration_transport import sha256, undo_map
from training.model.calibrate import collect_logits, selected_run
from training.model.calibration import fit_temperature, metrics
from training.model.data import load_partition
from training.model.infer import checkpoint_fingerprint

VERSION = "decision2-qwen38-hardcal-transport-screen/1"
MODEL_SHA = "d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2"
ORIGINAL_CAL_SHA = "e4e0d9fda575d807503299bdf7c67828a7c4d3c3b09e9fa1bf3750a51b79db78"
HARD_CAL_SHA = "bf5bbf29693928a2559ce0aff10e9d6b5b1541b50698634fcdb7725902412dcf"
HARD_MANIFEST_SHA = "063c3fd1f2ddb0e2f08823e77961e5b356e18c9752c24b3ff84d3150270d3e0c"
DEV_PREDICTIONS_SHA = "15ebc30b0207208e971e71deff2fceb4538fc74caf389ba47a2db9fdc9c622b1"
FAMILIES = {
    "cal_hard_archive_approval": "noul",
    "cal_hard_net_credit_grade": "score",
}
DEV_COUNTS = {"choice": 800, "noul": 400, "score": 400}


def checked_rows(hard_cal: Path, hard_manifest: Path) -> list[dict]:
    if sha256(hard_cal) != HARD_CAL_SHA or sha256(hard_manifest) != HARD_MANIFEST_SHA:
        raise ValueError("Frozen hard-CAL input bytes changed")
    rows = [row for row in load_partition(hard_cal, "cal") if row["family"] in FAMILIES]
    if len(rows) != 600 or Counter(row["family"] for row in rows) != dict.fromkeys(
        FAMILIES, 300
    ):
        raise ValueError("Expected exactly 300 fresh rows per hard-CAL family")
    for row in rows:
        if row["task_type"] != FAMILIES[row["family"]]:
            raise ValueError("Hard-CAL family/task type mismatch")
    return rows


def diagnostic_fit(
    run_dir: Path,
    source_path: Path,
    original_cal_path: Path,
    original_fit_path: Path,
    hard_cal_path: Path,
    hard_manifest_path: Path,
    logits_path: Path,
    output: Path,
) -> dict:
    if logits_path.exists() or output.exists():
        raise FileExistsError("Diagnostic output already exists")
    if sha256(original_fit_path) != ORIGINAL_CAL_SHA:
        raise ValueError("Original CAL fit differs")
    original = json.loads(original_fit_path.read_text(encoding="utf-8"))
    selected = selected_run(run_dir, original_cal_path)
    if (
        selected["name"] != "checkpoint-0000368"
        or selected["contract"]["max_length"] != 4096
    ):
        raise ValueError("Completed run selection or context limit changed")
    identity = checkpoint_fingerprint(selected["checkpoint"], source_path)
    if identity["model_sha256"] != MODEL_SHA or original["model_sha256"] != MODEL_SHA:
        raise ValueError("Source/checkpoint/original CAL model differs")
    rows = checked_rows(hard_cal_path, hard_manifest_path)
    records = collect_logits(
        selected["checkpoint"],
        rows,
        source_path=source_path,
        max_length=4096,
        batch_size=1,
        device_name="cuda:0",
    )
    expected = {row["id"]: (row["task_type"], row["label"]) for row in rows}
    observed = {row["id"]: (row["task_type"], row["label"]) for row in records}
    if len(records) != 600 or observed != expected:
        raise ValueError("Hard-CAL logit collection dropped or changed rows")
    by_type = {
        kind: [row for row in records if row["task_type"] == kind]
        for kind in ("noul", "score")
    }
    temperatures = dict(original["temperature_by_type"])
    for kind, subset in by_type.items():
        temperatures[kind] = fit_temperature(subset)
    report = {
        "version": VERSION,
        "diagnostic_only": True,
        "model_sha256": MODEL_SHA,
        "selected_checkpoint": selected["name"],
        "checkpoint_best_sha256": selected["best_sha256"],
        "original_calibration_sha256": ORIGINAL_CAL_SHA,
        "hard_cal_sha256": HARD_CAL_SHA,
        "hard_manifest_sha256": HARD_MANIFEST_SHA,
        "original_cal_sha256": selected["cal_sha256"],
        "fit_method": "per-type hard-label NLL, choice unchanged from original CAL700",
        "temperature_by_type": temperatures,
        "by_type": {
            kind: {
                "original_cal_temperature": metrics(
                    subset, original["temperature_by_type"][kind]
                ),
                "new_hard_cal_temperature": metrics(subset, temperatures[kind]),
            }
            for kind, subset in by_type.items()
        },
        "counts": {kind: len(subset) for kind, subset in by_type.items()},
    }
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    with logits_path.open("x", encoding="utf-8") as target:
        for row in records:
            target.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        target.flush()
        os.fsync(target.fileno())
    report["logits_sha256"] = sha256(logits_path)
    with output.open("x", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, sort_keys=True, indent=2)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    return report


def diagnostic_transform(
    original_predictions: Path,
    original_fit_path: Path,
    hard_fit_path: Path,
    output: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if sha256(original_predictions) != DEV_PREDICTIONS_SHA:
        raise ValueError("Frozen DEV predictions changed")
    if sha256(original_fit_path) != ORIGINAL_CAL_SHA:
        raise ValueError("Original CAL fit changed")
    original = json.loads(original_fit_path.read_text(encoding="utf-8"))
    hard = json.loads(hard_fit_path.read_text(encoding="utf-8"))
    if (
        hard.get("version") != VERSION
        or hard.get("model_sha256") != MODEL_SHA
        or hard.get("original_calibration_sha256") != ORIGINAL_CAL_SHA
        or hard["temperature_by_type"]["choice"]
        != original["temperature_by_type"]["choice"]
    ):
        raise ValueError("Hard-CAL fit contract differs")
    hard_sha = sha256(hard_fit_path)
    counts: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with original_predictions.open(encoding="utf-8") as source, output.open(
        "x", encoding="utf-8"
    ) as sink:
        for line in source:
            row = json.loads(line)
            if (
                row.get("model_sha256") != MODEL_SHA
                or row.get("calibration_sha256") != ORIGINAL_CAL_SHA
            ):
                raise ValueError("Prediction model or original calibration differs")
            for answer in row["answers"].values():
                kind = answer["type"]
                counts[kind] += 1
                if kind == "choice":
                    continue
                exponent = (
                    original["temperature_by_type"][kind]
                    / hard["temperature_by_type"][kind]
                )
                if kind == "noul":
                    probability = float(answer["noul"])
                    answer["noul"] = undo_map(
                        {"false": 1.0 - probability, "true": probability}, exponent
                    )["true"]
                elif kind == "score":
                    answer["probabilities"] = undo_map(
                        answer["probabilities"], exponent
                    )
                    answer["score"] = sum(
                        int(level) * probability
                        for level, probability in answer["probabilities"].items()
                    )
                else:
                    raise ValueError("Unsupported answer type")
            row["diagnostic_original_calibration_sha256"] = row.pop(
                "calibration_sha256"
            )
            row["diagnostic_hard_calibration_sha256"] = hard_sha
            row["diagnostic_transform"] = VERSION
            sink.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    if counts != DEV_COUNTS:
        raise ValueError("Frozen typed DEV answer counts changed")
    return {
        "version": VERSION,
        "diagnostic_only": True,
        "source_predictions_sha256": DEV_PREDICTIONS_SHA,
        "hard_fit_sha256": hard_sha,
        "output_sha256": sha256(output),
        "counts": dict(counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    for name in (
        "run-dir",
        "source-path",
        "original-cal",
        "original-fit",
        "hard-cal",
        "hard-manifest",
        "logits",
        "output",
    ):
        fit.add_argument(f"--{name}", type=Path, required=True)
    transform = subparsers.add_parser("transform")
    for name in ("original-predictions", "original-fit", "hard-fit", "output"):
        transform.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        report = diagnostic_fit(
            args.run_dir,
            args.source_path,
            args.original_cal,
            args.original_fit,
            args.hard_cal,
            args.hard_manifest,
            args.logits,
            args.output,
        )
        print(
            json.dumps(
                {
                    "temperature_by_type": report["temperature_by_type"],
                    "logits_sha256": report["logits_sha256"],
                },
                sort_keys=True,
            )
        )
    else:
        report = diagnostic_transform(
            args.original_predictions, args.original_fit, args.hard_fit, args.output
        )
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
