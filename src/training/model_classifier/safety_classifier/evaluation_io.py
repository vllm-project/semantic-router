"""Prepared-row parsing and deterministic evaluation artifact writers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _row_matches(row: dict[str, Any], key: str, expected: str | None) -> bool:
    if expected is None or key not in row:
        return True
    return str(row[key]) == expected


def read_prepared_jsonl(
    path: str | Path,
    *,
    task_name: str | None = None,
    split: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read prepared examples while accepting the canonical text/prompt aliases."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_number} of {path}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} of {path} is not a JSON object")
            if not _row_matches(row, "task", task_name) or not _row_matches(
                row,
                "split",
                split,
            ):
                continue
            text = row.get("text", row.get("prompt"))
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"line {line_number} of {path} has no non-empty text")
            prepared = dict(row)
            prepared["text"] = text
            prepared["_line_number"] = line_number
            rows.append(prepared)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no matching examples found in {path}")
    return rows


def _label_value(
    row: dict[str, Any],
    task_name: str,
    label2id: dict[str, int],
) -> int:
    candidates = (
        f"{task_name}_label_id",
        f"{task_name}_label",
        "label_id",
        "label",
    )
    for key in candidates:
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, str) and value in label2id:
            return int(label2id[value])
        try:
            label_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid {key} on JSONL line {row['_line_number']}"
            ) from exc
        if label_id not in label2id.values():
            raise ValueError(f"out-of-range {key} on JSONL line {row['_line_number']}")
        return label_id
    raise ValueError(f"missing label on JSONL line {row['_line_number']}")


def _strict_single_target(row: dict[str, Any]) -> bool:
    for key in ("strict_single_target", "is_strict_single_target"):
        if key in row:
            return bool(row[key])
    if "is_multitarget" in row:
        return not bool(row["is_multitarget"])
    for key in ("mapped_labels", "mapped_targets", "target_labels"):
        value = row.get(key)
        if isinstance(value, list):
            return len(value) == 1
    if "target_count" in row:
        return row["target_count"] == 1
    return False


def _sample_identity(row: dict[str, Any], fallback_index: int) -> str:
    for key in ("sample_id", "fingerprint", "source_id", "id", "example_id"):
        if key in row:
            return str(row[key])
    return f"row-{fallback_index}"


def _prediction_record(
    row: dict[str, Any],
    index: int,
    *,
    reference: int,
    prediction: int,
    scores: list[float],
    names: list[str],
    include_text: bool,
) -> dict[str, Any]:
    text = row["text"]
    record: dict[str, Any] = {
        "sample_id": _sample_identity(row, index),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "reference": reference,
        "reference_label": names[reference],
        "prediction": prediction,
        "prediction_label": names[prediction],
        "scores": scores,
        "strict_single_target": _strict_single_target(row),
    }
    for key in ("source", "split", "source_split"):
        if key in row:
            record[key] = row[key]
    if include_text:
        record["text"] = text
    return record


def _write_outputs(
    output_dir: str | Path,
    metrics_payload: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.json"
    predictions_path = destination / "predictions.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            metrics_payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    with predictions_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(
                json.dumps(
                    prediction,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")
    return metrics_path, predictions_path
