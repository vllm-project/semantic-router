"""Gold-free, matched-batch repeatability gate for the frozen Sol 2B source.

The private lock names an existing merged source, SELECT partition, and old
zero-step predictions. This module never optimizes a model or writes labels.
Run once per fresh process; a failed historical gate ends the experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

from .data import file_sha256, load_partition
from .decision_model import DecisionModel
from .inline_replay import roster_sha256
from .inline_teacher import (
    native_probabilities_batch,
    source_pairs_for_selection,
    verify_materialization,
)

SCHEMA = "decision2-sol2b-zero-step-repeat/1"
LOCK_SCHEMA = "decision2-sol2b-zero-step-prereg/1"
HISTORIC_MAX_DRIFT = 1e-4
REPEAT_MAX_DRIFT = 1e-6


def _write_once(path: Path, document: dict) -> None:
    if not path.parent.is_dir():
        raise ValueError("Output parent does not exist")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _prediction(probabilities: dict[str, float], task_type: str) -> str | None:
    if task_type == "noul":
        p_true = probabilities["true"]
        return None if p_true == 0.5 else "true" if p_true > 0.5 else "false"
    maximum = max(probabilities.values())
    winners = [
        key for key, value in probabilities.items() if abs(value - maximum) <= 1e-8
    ]
    return winners[0] if len(winners) == 1 else None


def _reference_probabilities(reference: dict) -> dict[str, float]:
    answer = reference["answer"]
    if reference["task_type"] == "noul":
        return {"false": 1.0 - answer["noul"], "true": answer["noul"]}
    return answer["probabilities"]


def compare_historical(records: list[dict], references: dict[str, dict]) -> dict:
    if len(records) != 32 or len({record["id"] for record in records}) != 32:
        raise ValueError("Zero-step roster is not exactly 32 unique rows")
    maximum = 0.0
    categorical = 0
    for record in records:
        reference = references[record["id"]]
        if any(
            record[key] != reference[key]
            for key in ("id", "prompt_sha256", "token_ids_sha256", "task_type")
        ):
            raise ValueError("Historical zero-step input identity differs")
        old = _reference_probabilities(reference)
        new = record["probabilities"]
        if set(old) != set(new):
            raise ValueError("Historical zero-step option keys differ")
        if any(not math.isfinite(value) for value in [*old.values(), *new.values()]):
            raise ValueError("Nonfinite zero-step probability")
        maximum = max(maximum, *(abs(new[key] - old[key]) for key in old))
        categorical += record["prediction_key"] != reference["prediction_key"]
    return {
        "checked": 32,
        "categorical_changes": categorical,
        "max_absolute_probability_drift": maximum,
        "status": (
            "PASS" if categorical == 0 and maximum <= HISTORIC_MAX_DRIFT else "FAIL"
        ),
    }


def compare_repeats(first: dict, second: dict) -> dict:
    if first["lock_sha256"] != second["lock_sha256"]:
        raise ValueError("Different prospective locks")
    if first["source_model_sha256"] != second["source_model_sha256"]:
        raise ValueError("Different model source")
    if first["backend"] != second["backend"]:
        raise ValueError("Different native backends")
    left = {record["id"]: record for record in first["predictions"]}
    right = {record["id"]: record for record in second["predictions"]}
    if len(left) != 32 or left.keys() != right.keys():
        raise ValueError("Incomplete or different zero-step rosters")
    maximum = 0.0
    categorical = 0
    for identifier in left:
        a, b = left[identifier], right[identifier]
        if (
            any(
                a[key] != b[key]
                for key in ("prompt_sha256", "token_ids_sha256", "task_type")
            )
            or a["probabilities"].keys() != b["probabilities"].keys()
        ):
            raise ValueError("Fresh-process native inputs differ")
        maximum = max(
            maximum,
            *(
                abs(a["probabilities"][key] - b["probabilities"][key])
                for key in a["probabilities"]
            ),
        )
        categorical += a["prediction_key"] != b["prediction_key"]
    return {
        "checked": 32,
        "categorical_changes": categorical,
        "max_absolute_probability_drift": maximum,
        "status": (
            "PASS" if categorical == 0 and maximum <= REPEAT_MAX_DRIFT else "FAIL"
        ),
    }


def _lock(path: Path) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("Wrong private preregistration schema")
    if lock.get("script_sha256") != file_sha256(Path(__file__)):
        raise ValueError("Runner differs from preregistered code")
    if (
        lock.get("optimizer_updates") != 0
        or lock.get("sample_count") != 32
        or lock.get("max_length") != 8192
        or lock.get("historical_max_drift") != HISTORIC_MAX_DRIFT
        or lock.get("repeat_max_drift") != REPEAT_MAX_DRIFT
    ):
        raise ValueError("Private preregistration changed the frozen protocol")
    if file_sha256(Path(lock["select_path"])) != lock["select_sha256"]:
        raise ValueError("SELECT differs from preregistration")
    if file_sha256(Path(lock["control_path"])) != lock["control_sha256"]:
        raise ValueError("Historical control differs from preregistration")
    return lock


def _historical_references(path: Path) -> dict[str, dict]:
    references = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in references:
                raise ValueError("Duplicate historical control ID")
            references[row["id"]] = row
    if len(references) != 700:
        raise ValueError("Historical control is not SELECT700")
    return references


def run(lock_path: Path, output: Path) -> dict:
    started = time.monotonic()
    lock = _lock(lock_path)
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16 GPU must be visible")
    torch.use_deterministic_algorithms(True)
    from training.eikos.published_infer import use_torch_reference_gated_delta

    backend = use_torch_reference_gated_delta()
    source_files, model_sha = verify_materialization(
        Path(lock["model_path"]),
        Path(lock["materialization_receipt_path"]),
        lock["materialization_receipt_sha256"],
    )
    if model_sha != lock["source_model_sha256"]:
        raise ValueError("Materialized model identity differs")
    rows = load_partition(Path(lock["select_path"]), "select")
    if len(rows) != 700:
        raise ValueError("SELECT row count differs")
    chosen = sorted(
        rows, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest()
    )[:32]
    if roster_sha256(chosen) != lock["parity_roster_sha256"]:
        raise ValueError("32-row zero-step roster differs")
    selected_ids = {row["id"] for row in chosen}
    model, tokenizer = DecisionModel.from_checkpoint(Path(lock["model_path"]))
    model = model.float().to(torch.device("cuda:0"))
    model.eval()
    model.backbone.config.use_cache = False
    results = []
    with torch.inference_mode():
        for pair in source_pairs_for_selection(rows, selected_ids):
            actuals = native_probabilities_batch(model, tokenizer, pair, 8192)
            for row, (item, probabilities) in zip(pair, actuals):
                if row["id"] not in selected_ids:
                    continue
                results.append(
                    {
                        "id": row["id"],
                        "prompt_sha256": item["prompt_sha256"],
                        "token_ids_sha256": item["token_ids_sha256"],
                        "task_type": row["task_type"],
                        "prediction_key": _prediction(probabilities, row["task_type"]),
                        "probabilities": probabilities,
                    }
                )
    if len(results) != 32:
        raise ValueError("Incomplete fixed zero-step roster")
    historical = compare_historical(
        results, _historical_references(Path(lock["control_path"]))
    )
    document = {
        "schema_version": SCHEMA,
        "lock_sha256": file_sha256(lock_path),
        "source_model_sha256": model_sha,
        "source_files_sha256": source_files,
        "backend": backend,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "gpu_name": torch.cuda.get_device_name(0),
        "parity_roster_sha256": lock["parity_roster_sha256"],
        "historical": historical,
        "elapsed_process_seconds": time.monotonic() - started,
        "predictions": results,
    }
    _write_once(output, document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = run(args.lock, args.output)
    print(
        json.dumps(
            {
                "historical_status": document["historical"]["status"],
                "categorical_changes": document["historical"]["categorical_changes"],
                "max_absolute_probability_drift": document["historical"][
                    "max_absolute_probability_drift"
                ],
                "output_sha256": file_sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
