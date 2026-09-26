"""Pure-CPU per-type post-hoc temperature fitting and inference contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .data import canonical, file_sha256

CALIBRATION_VERSION = "decision2-per-type-temperature/1"
TASK_TYPES = ("choice", "noul", "score")
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def validate_temperatures(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(TASK_TYPES):
        raise ValueError("Calibration needs exact Choice/Noul/Score temperatures")
    result = {}
    for kind in TASK_TYPES:
        raw = value[kind]
        if (
            type(raw) not in (float, int)
            or not math.isfinite(raw)
            or not MIN_TEMPERATURE <= raw <= MAX_TEMPERATURE
        ):
            raise ValueError(
                f"{kind} temperature must be finite in [{MIN_TEMPERATURE},{MAX_TEMPERATURE}]"
            )
        result[kind] = float(raw)
    return result


def validate_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {kind: [] for kind in TASK_TYPES}
    seen: set[str] = set()
    for row in records:
        if not isinstance(row, dict) or row.get("task_type") not in grouped:
            raise ValueError(
                "Calibration logits need a native Choice/Noul/Score task type"
            )
        item_id, logits, label = row.get("id"), row.get("logits"), row.get("label")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise ValueError("Calibration logits need unique nonempty IDs")
        if not isinstance(logits, (list, tuple)) or not 2 <= len(logits) <= 255:
            raise ValueError(f"{item_id}: expected 2..255 candidate logits")
        if type(label) is not int or not 0 <= label < len(logits):
            raise ValueError(f"{item_id}: gold index is outside candidate logits")
        if any(
            type(value) not in (float, int) or not math.isfinite(value)
            for value in logits
        ):
            raise ValueError(f"{item_id}: logits must be finite numbers")
        seen.add(item_id)
        grouped[row["task_type"]].append(row)
    missing = [kind for kind, subset in grouped.items() if not subset]
    if missing:
        raise ValueError(
            f"CAL needs at least one example of every native task type: missing {missing}"
        )
    return grouped


def probabilities(
    logits: list[float] | tuple[float, ...], temperature: float
) -> list[float]:
    scaled = [float(value) / temperature for value in logits]
    maximum = max(scaled)
    exponents = [math.exp(value - maximum) for value in scaled]
    total = sum(exponents)
    return [value / total for value in exponents]


def nll(records: list[dict[str, Any]], temperature: float) -> float:
    total = 0.0
    for row in records:
        scaled = [float(value) / temperature for value in row["logits"]]
        maximum = max(scaled)
        total += (
            maximum
            + math.log(sum(math.exp(value - maximum) for value in scaled))
            - scaled[row["label"]]
        )
    return total / len(records)


def fit_temperature(records: list[dict[str, Any]]) -> float:
    """Minimize CAL hard-label NLL in log-temperature by golden-section search."""
    if not records:
        raise ValueError("Cannot fit temperature on an empty type")
    left, right = math.log(MIN_TEMPERATURE), math.log(MAX_TEMPERATURE)
    ratio = (math.sqrt(5) - 1) / 2
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    f1, f2 = nll(records, math.exp(x1)), nll(records, math.exp(x2))
    for _ in range(80):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = nll(records, math.exp(x1))
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = nll(records, math.exp(x2))
    candidates = (1.0, MIN_TEMPERATURE, MAX_TEMPERATURE, math.exp((left + right) / 2))
    return min(
        candidates, key=lambda value: (nll(records, value), abs(math.log(value)))
    )


def metrics(records: list[dict[str, Any]], temperature: float) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot score empty calibration rows")
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(10)]
    brier = correct = 0.0
    for row in records:
        probs = probabilities(row["logits"], temperature)
        label = row["label"]
        top = max(probs)
        winners = [
            index for index, value in enumerate(probs) if abs(value - top) <= 1e-8
        ]
        hit = len(winners) == 1 and winners[0] == label
        correct += hit
        brier += (
            sum(
                (value - float(index == label)) ** 2
                for index, value in enumerate(probs)
            )
            / 2
        )
        bins[min(9, int(top * 10))].append((top, hit))
    ece = sum(
        len(bucket)
        / len(records)
        * abs(
            sum(confidence for confidence, _ in bucket) / len(bucket)
            - sum(hit for _, hit in bucket) / len(bucket)
        )
        for bucket in bins
        if bucket
    )
    return {
        "n": len(records),
        "nll": nll(records, temperature),
        "brier": brier / len(records),
        "ece_10": ece,
        "accuracy_all": correct / len(records),
    }


def fit_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = validate_records(records)
    temperatures = {kind: fit_temperature(grouped[kind]) for kind in TASK_TYPES}
    return {
        "temperature_by_type": temperatures,
        "by_type": {
            kind: {
                "before": metrics(grouped[kind], 1.0),
                "after": metrics(grouped[kind], temperatures[kind]),
            }
            for kind in TASK_TYPES
        },
        "overall": {
            "before": metrics(records, 1.0),
            "after": _overall_after(grouped, temperatures),
        },
    }


def _overall_after(
    grouped: dict[str, list[dict[str, Any]]], temperatures: dict[str, float]
) -> dict[str, Any]:
    # ECE is recomputed from all item confidences, not averaged over type ECEs.
    rows = []
    for kind, subset in grouped.items():
        for row in subset:
            rows.append((row, temperatures[kind]))
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(10)]
    total_nll = total_brier = correct = 0.0
    for row, temperature in rows:
        probs = probabilities(row["logits"], temperature)
        label, top = row["label"], max(probs)
        winners = [
            index for index, value in enumerate(probs) if abs(value - top) <= 1e-8
        ]
        hit = len(winners) == 1 and winners[0] == label
        correct += hit
        total_nll += nll([row], temperature)
        total_brier += (
            sum(
                (value - float(index == label)) ** 2
                for index, value in enumerate(probs)
            )
            / 2
        )
        bins[min(9, int(top * 10))].append((top, hit))
    n = len(rows)
    ece = sum(
        len(bucket)
        / n
        * abs(
            sum(confidence for confidence, _ in bucket) / len(bucket)
            - sum(hit for _, hit in bucket) / len(bucket)
        )
        for bucket in bins
        if bucket
    )
    return {
        "n": n,
        "nll": total_nll / n,
        "brier": total_brier / n,
        "ece_10": ece,
        "accuracy_all": correct / n,
    }


def verified_materialization_origin(
    checkpoint: Path,
    merged_model_sha256: str,
    receipt_path: Path | None = None,
) -> dict[str, str] | None:
    """Prove that a full checkpoint was merged from a specific LoRA identity."""
    metadata = json.loads(
        (checkpoint / "decision_config.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(metadata, dict)
        or metadata.get("initialization") != "merged-peft-lora"
    ):
        return None
    origin = metadata.get("lora_origin")
    if not isinstance(origin, dict) or not isinstance(origin.get("adapter"), dict):
        raise ValueError("Materialized checkpoint lacks embedded LoRA lineage")
    source_sha = _sha(
        origin["adapter"].get("model_sha256"), "materialized LoRA source model SHA"
    )
    if receipt_path is None:
        portable = checkpoint / "materialization_receipt.json"
        receipt_path = (
            portable
            if portable.is_file()
            else checkpoint.with_name(checkpoint.name + ".materialization.json")
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        not isinstance(receipt, dict)
        or receipt.get("materialization_version") != "decision2-merged-peft-lora/1"
    ):
        raise ValueError("Materialization receipt has an unknown format")
    if (
        receipt.get("source_model_sha256") != source_sha
        or receipt.get("merged_model_sha256") != merged_model_sha256
    ):
        raise ValueError(
            "Materialization receipt does not bind source and merged model hashes"
        )
    files = receipt.get("merged_model_files_sha256")
    if (
        not isinstance(files, dict)
        or hashlib.sha256(canonical(files).encode("utf-8")).hexdigest()
        != merged_model_sha256
    ):
        raise ValueError(
            "Materialization receipt's merged file hashes disagree with its model hash"
        )
    return {
        "source_model_sha256": source_sha,
        "receipt_sha256": file_sha256(receipt_path),
    }


def load_calibration(
    path: Path,
    expected_model_sha256: str,
    *,
    materialized_source_sha256: str | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("calibration_version") != CALIBRATION_VERSION
    ):
        raise ValueError("Unknown Decision 2.0 calibration format")
    _sha(expected_model_sha256, "inference model_sha256")
    reported_model = _sha(report.get("model_sha256"), "calibration model_sha256")
    if (
        reported_model != expected_model_sha256
        and reported_model != materialized_source_sha256
    ):
        raise ValueError("Calibration model hash differs from the inference checkpoint")
    for name in (
        "checkpoint_sha256",
        "cal_sha256",
        "best_sha256",
        "complete_sha256",
        "provenance_sha256",
    ):
        _sha(report.get(name), name)
    if (
        report.get("fit_split") != "cal"
        or report.get("selection_policy") != "completed_run_best_only"
    ):
        raise ValueError(
            "Calibration was not fit on a completed run's frozen CAL partition"
        )
    temperatures = validate_temperatures(report.get("temperature_by_type"))
    return temperatures, report
