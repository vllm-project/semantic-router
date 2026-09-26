"""Compare a standalone merged Eikos candidate with selected native LoRA outputs.

The reference is a gold-free prediction receipt. This check reads no benchmark
labels and cannot select a checkpoint; it only measures packaging drift.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from inference.eikos import shared_answer
from inference.run import digest, load_prompts

from training.eikos.io import atomic_json
from training.eikos.native import load_decider, selected_checkpoint
from training.model.data import file_sha256

PUBLICATION_DOCUMENTS = frozenset(
    {
        "README.md",
        "score-table.md",
        "ranking.svg",
        "matrix.svg",
        "card-artifacts/score-table.md",
        "card-artifacts/ranking.svg",
        "card-artifacts/matrix.svg",
        "card-artifacts/manifest.json",
        "publication-manifest.json",
    }
)


def verify_sums(folder: Path) -> int:
    lines = (folder / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    seen = set()
    for line in lines:
        expected, separator, name = line.partition("  ")
        if (
            separator != "  "
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise ValueError("Malformed candidate SHA256SUMS")
        if name in PUBLICATION_DOCUMENTS:
            raise ValueError(
                f"Publication document must not alter functional model identity: {name}"
            )
        if (
            name in seen
            or (folder / name).is_symlink()
            or file_sha256(folder / name) != expected
        ):
            raise ValueError(f"Candidate file changed: {name}")
        seen.add(name)
    actual = {
        str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()
    }
    if any((folder / name).is_symlink() for name in actual):
        raise ValueError("Candidate package contains a symlink")
    if (actual - {"SHA256SUMS"} - PUBLICATION_DOCUMENTS) != seen:
        raise ValueError("Candidate file set differs from SHA256SUMS")
    return len(seen)


def compare_answers(
    expected: dict[str, Any], actual: dict[str, Any]
) -> tuple[bool, float, float]:
    if expected["type"] != actual["type"]:
        raise ValueError("Packaged answer type changed")
    if expected["type"] == "choice":
        same = expected["choice"] == actual["choice"]
        left, right = expected["probabilities"], actual["probabilities"]
    elif expected["type"] == "score":
        # The shared Score schema puts the continuous expectation in `score`;
        # the categorical answer being evaluated is the native modal level.
        same = expected["native_score"] == actual["native_score"]
        left, right = expected["probabilities"], actual["probabilities"]
    elif expected["type"] in {"noul", "boolean"}:
        same = expected["value"] == actual["value"]
        left, right = {"yes": expected["probability"]}, {"yes": actual["probability"]}
    else:
        raise ValueError("Unexpected typed answer")
    if set(left) != set(right):
        raise ValueError("Packaged option set changed")
    drift = max(abs(float(left[key]) - float(right[key])) for key in left)
    if not math.isfinite(drift):
        raise ValueError("Nonfinite packaged probability")
    if expected["type"] in {"noul", "boolean"}:
        pmax_left, pmax_right = max(float(left["yes"]), 1 - float(left["yes"])), max(
            float(right["yes"]), 1 - float(right["yes"])
        )
    else:
        pmax_left, pmax_right = max(float(p) for p in left.values()), max(
            float(p) for p in right.values()
        )
    return same, drift, abs(pmax_left - pmax_right)


def verify(
    *,
    model_path: Path,
    run: Path,
    merged: Path,
    prompts: Path,
    reference: Path | None,
    output: Path,
    device: str = "cuda:0",
    max_items: int | None = None,
    repeat_selected: bool = False,
    direct_selected: bool = False,
) -> dict[str, Any]:
    if repeat_selected and direct_selected:
        raise ValueError("Repeat control and direct parity are different checks")
    if reference is None and not direct_selected:
        raise ValueError(
            "A reference prediction file is required for historical parity"
        )
    if output.exists():
        raise FileExistsError(output)
    checked_files = verify_sums(merged)
    selection = selected_checkpoint(run, model_path)
    receipt = json.loads(
        (merged / "decision2_provenance.json").read_text(encoding="utf-8")
    )
    if (
        receipt["source_revision"] != "582ffb13f19a4da3f455e3db198584190bd7755b"
        or receipt["selected_checkpoint"] != selection["name"]
        or receipt["adapter_weights_sha256"] != selection["adapter_weights_sha256"]
        or receipt["calibration_sha256"] != file_sha256(merged / "calib.json")
    ):
        raise ValueError("Candidate provenance does not bind selected native LoRA")
    rows = load_prompts(prompts)
    if max_items is not None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        rows = rows[:max_items]
    predictions = {}
    if reference is not None:
        with reference.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if row["id"] in predictions:
                        raise ValueError("Duplicate reference prediction")
                    predictions[row["id"]] = row
        if set(predictions) != {row["id"] for row in load_prompts(prompts)}:
            raise ValueError("Reference prediction IDs differ from prompt IDs")
        if any(
            row.get("adapter_weights_sha256") != selection["adapter_weights_sha256"]
            or row.get("calibration_sha256") != receipt["calibration_sha256"]
            or row.get("checkpoint") != selection["name"]
            for row in predictions.values()
        ):
            raise ValueError(
                "Reference predictions do not use the selected LoRA/calibration"
            )

    decider = load_decider(
        model_path if repeat_selected else merged,
        selection["adapter"] if repeat_selected else None,
        merged / "calib.json",
        device=device,
    )
    selected_decider = (
        load_decider(
            model_path, selection["adapter"], merged / "calib.json", device=device
        )
        if direct_selected
        else None
    )
    mismatches: list[str] = []
    drifts = []
    pmax_drifts = []
    largest: list[dict[str, Any]] = []
    for row in rows:
        payload = {"state": row["state"], "questions": row["questions"]}
        if direct_selected:
            selected_results = selected_decider.decide_all(**payload)
            prior_answers = {
                key: shared_answer(row["questions"][key], answer)
                for key, (answer, _) in selected_results.items()
            }
        else:
            prior = predictions[row["id"]]
            if prior["source_input_sha256"] != digest(payload):
                raise ValueError(f"Reference input changed: {row['id']}")
            prior_answers = prior["answers"]
        results = decider.decide_all(**payload)
        if set(results) != set(prior_answers):
            raise ValueError("Packaged question set changed")
        for key, (answer, _) in results.items():
            shared = shared_answer(row["questions"][key], answer)
            same, drift, pmax_drift = compare_answers(prior_answers[key], shared)
            if not same:
                mismatches.append(f"{row['id']}:{key}")
            drifts.append(drift)
            pmax_drifts.append(pmax_drift)
            largest.append(
                {
                    "id": row["id"],
                    "question": key,
                    "type": shared["type"],
                    "max_probability_drift": drift,
                    "pmax_drift": pmax_drift,
                    "categorical_match": same,
                }
            )
    ordered = sorted(drifts)
    p99 = ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)]
    max_drift = max(drifts)
    mean_drift = sum(drifts) / len(drifts)
    gate = len(mismatches) == 0 and p99 <= 0.005 and max_drift <= 0.02
    report = {
        "role": (
            "gold-free direct selected-LoRA versus merged package parity"
            if direct_selected
            else (
                "gold-free selected-LoRA repeat control"
                if repeat_selected
                else "gold-free historical packaging parity; no labels read"
            )
        ),
        "inference_variant": (
            "selected_lora_and_merged_same_process"
            if direct_selected
            else (
                "selected_native_lora_repeat"
                if repeat_selected
                else "merged_full_weights"
            )
        ),
        "candidate": str(merged),
        "candidate_manifest_sha256": file_sha256(merged / "SHA256SUMS"),
        "candidate_files_checked": checked_files,
        "source_release": receipt["source_release"],
        "selected_checkpoint": selection["name"],
        "adapter_weights_sha256": selection["adapter_weights_sha256"],
        "calibration_sha256": receipt["calibration_sha256"],
        "prompt_sha256": file_sha256(prompts),
        "reference_prediction_sha256": (
            file_sha256(reference) if reference is not None else None
        ),
        "items": len(rows),
        "answers": len(drifts),
        "choice_mismatch_n": len(mismatches),
        "choice_mismatch_ids": mismatches,
        "probability_drift_mean_max_per_answer": mean_drift,
        "probability_drift_p99": p99,
        "probability_drift_max": max_drift,
        "pmax_abs_drift_mean": sum(pmax_drifts) / len(pmax_drifts),
        "pmax_abs_drift_max": max(pmax_drifts),
        "largest_probability_drifts": sorted(
            largest, key=lambda item: item["max_probability_drift"], reverse=True
        )[:10],
        "predeclared_gate": (
            {
                "categorical_mismatches": 0,
                "probability_drift_p99_lte": 0.005,
                "probability_drift_max_lte": 0.02,
                "pass": gate,
            }
            if direct_selected
            else None
        ),
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--repeat-selected", action="store_true")
    parser.add_argument("--direct-selected", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                model_path=args.model_path,
                run=args.run,
                merged=args.merged,
                prompts=args.prompts,
                reference=args.reference,
                output=args.output,
                device=args.device,
                max_items=args.max_items,
                repeat_selected=args.repeat_selected,
                direct_selected=args.direct_selected,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
