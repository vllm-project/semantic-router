"""Strict, flattened training-row contract and split-leakage checks.

Benchmark gold JSONL is intentionally a different format. It cannot be passed
to this loader or silently converted into training examples.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

MAX_OPTIONS = 255
ROLES = {"train": {"train"}, "select": {"select"}, "cal": {"cal", "calibrate"}}
INPUT_FIELDS = ("state", "instructions", "options", "task_type")
REQUIRED = {
    "id",
    "state",
    "instructions",
    "options",
    "label",
    "task_type",
    "family",
    "group_id",
    "language",
    "split",
    "source",
    "evaluation_role",
    "render_template",
    "audit_metadata",
    "input_sha256",
}


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _nonempty_string(row: dict[str, Any], field: str) -> None:
    if not isinstance(row.get(field), str) or not row[field].strip():
        raise ValueError(
            f"{row.get('id', '<missing-id>')}: {field} must be a nonempty string"
        )


def validate_row(row: Any, partition: str, *, replay: bool = False) -> dict[str, Any]:
    if partition not in ROLES:
        raise ValueError(f"Unknown partition {partition}")
    if not isinstance(row, dict):
        raise ValueError("Every JSONL line must be an object")
    missing = REQUIRED - row.keys()
    if missing:
        raise ValueError(f"{row.get('id', '<missing-id>')}: missing {sorted(missing)}")
    for field in ("id", "family", "group_id", "language", "source", "render_template"):
        _nonempty_string(row, field)
    if not isinstance(row["audit_metadata"], dict):
        raise ValueError(f"{row['id']}: audit_metadata must be an object")
    if row["split"] != partition or row["evaluation_role"] not in ROLES[partition]:
        raise ValueError(
            f"{row['id']}: expected split={partition} and role in {sorted(ROLES[partition])}"
        )
    if row["task_type"] not in {"choice", "noul", "score"}:
        raise ValueError(f"{row['id']}: unsupported task_type")
    if (
        not isinstance(row["instructions"], (str, dict, list))
        or not row["instructions"]
    ):
        raise ValueError(f"{row['id']}: instructions must be nonempty text or JSON")
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(
            f"{row['id']}: options must contain 2..{MAX_OPTIONS} candidates"
        )
    for option in options:
        if (
            not isinstance(option, dict)
            or not isinstance(option.get("key"), str)
            or not option["key"]
        ):
            raise ValueError(f"{row['id']}: every option needs a nonempty string key")
        if "description" not in option or not isinstance(
            option["description"], (str, dict, list)
        ):
            raise ValueError(
                f"{row['id']}: every option needs a text or structured JSON description"
            )
        try:
            canonical(option["description"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{row['id']}: option description must be finite JSON"
            ) from exc
    keys = [option["key"] for option in options]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{row['id']}: duplicate option keys")
    if row["task_type"] == "noul" and set(keys) != {"false", "true"}:
        raise ValueError(f"{row['id']}: noul keys must be false/true")
    if row["task_type"] == "score" and set(keys) != {str(i) for i in range(len(keys))}:
        raise ValueError(f"{row['id']}: score keys must enumerate levels 0..K-1")
    if type(row["label"]) is not int or not 0 <= row["label"] < len(options):
        raise ValueError(f"{row['id']}: label must index a real option")
    expected_hash = digest({field: row[field] for field in INPUT_FIELDS})
    if row["input_sha256"] != expected_hash:
        raise ValueError(f"{row['id']}: input_sha256 disagrees with canonical payload")
    if "target_probs" in row:
        raise ValueError(
            f"{row['id']}: target_probs is a legacy soft-label field; use replay teacher_probs"
        )
    if replay:
        teacher = row.get("teacher_probs")
        if not isinstance(teacher, dict) or set(teacher) != set(keys):
            raise ValueError(
                f"{row['id']}: replay teacher_probs must cover exactly the option keys"
            )
        if any(
            type(p) not in (int, float) or not math.isfinite(p) or p < 0
            for p in teacher.values()
        ):
            raise ValueError(
                f"{row['id']}: teacher_probs must be finite and nonnegative"
            )
        if abs(sum(teacher.values()) - 1.0) > 1e-5:
            raise ValueError(f"{row['id']}: teacher_probs must sum to one")
    elif "teacher_probs" in row:
        raise ValueError(
            f"{row['id']}: teacher_probs is permitted only in a separate replay input"
        )
    return row


def load_partition(
    path: str | Path, partition: str, *, replay: bool = False
) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            try:
                row = validate_row(json.loads(line), partition, replay=replay)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if row["id"] in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty partition")
    return rows


def check_partition_isolation(partitions: dict[str, list[dict[str, Any]]]) -> None:
    """Disallow exact ID, lineage group, or canonical input sharing across roles."""
    seen: dict[str, dict[str, str]] = {
        name: {} for name in ("id", "group_id", "input_sha256")
    }
    for partition, rows in partitions.items():
        role = "train" if partition in {"train", "replay"} else partition
        for row in rows:
            for field in seen:
                value = row[field]
                previous = seen[field].get(value)
                if previous is not None and previous != role:
                    raise ValueError(
                        f"Leakage: {field}={value} appears in {previous} and {role}"
                    )
                seen[field][value] = role
