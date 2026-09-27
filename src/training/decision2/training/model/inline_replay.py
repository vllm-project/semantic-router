"""Bind source-model soft targets to existing TRAIN prompts without replay rows."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .data import digest

SCHEMA = "decision2-inline-soft-replay/1"


def roster_sha256(rows: list[dict[str, Any]]) -> str:
    """Preserve TRAIN order while committing only row and native-input identities."""
    return digest(
        [{"id": row["id"], "input_sha256": row["input_sha256"]} for row in rows]
    )


def attach_inline_teacher(
    path: str | Path,
    train_rows: list[dict[str, Any]],
    *,
    train_sha256: str,
    source_files_sha256: dict[str, str],
    expected_source_model_sha256: str,
    expected_materialization_receipt_sha256: str,
    expected_roster_sha256: str,
    expected_control_baseline_sha256: str,
    expected_parity_roster_sha256: str,
) -> int:
    """Validate the complete private teacher artifact, then attach in memory.

    The original TRAIN bytes and prompt fields stay untouched. The teacher
    artifact must refer to the exact initialization source and a frozen,
    ordered subset of TRAIN rows. No benchmark row or extra example is read.
    """
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA:
        raise ValueError("Invalid inline teacher schema")
    if artifact.get("train_sha256") != train_sha256:
        raise ValueError("Inline teacher TRAIN hash differs")
    if artifact.get("source_files_sha256") != source_files_sha256:
        raise ValueError("Inline teacher initialization source differs")
    if artifact.get("source_merged_model_sha256") != expected_source_model_sha256:
        raise ValueError("Inline teacher merged-model hash differs")
    if (
        artifact.get("materialization_receipt_sha256")
        != expected_materialization_receipt_sha256
    ):
        raise ValueError("Inline teacher materialization receipt differs")
    if artifact.get("roster_sha256") != expected_roster_sha256:
        raise ValueError("Inline teacher frozen roster hash differs")
    parity = artifact.get("zero_step_parity")
    if (
        not isinstance(parity, dict)
        or parity.get("status") != "PASS"
        or parity.get("count") != 32
        or parity.get("control_baseline_sha256") != expected_control_baseline_sha256
        or parity.get("parity_roster_sha256") != expected_parity_roster_sha256
        or type(parity.get("max_absolute_probability_drift")) not in (int, float)
        or not math.isfinite(parity["max_absolute_probability_drift"])
        or not 0 <= parity["max_absolute_probability_drift"] <= 1e-4
    ):
        raise ValueError("Inline teacher zero-step parity differs or failed")
    entries = artifact.get("rows")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Inline teacher needs a nonempty row list")
    by_id = {row["id"]: row for row in train_rows}
    if len(by_id) != len(train_rows):
        raise ValueError("TRAIN row IDs are not unique")
    selected = []
    teachers: dict[str, dict[str, float]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "id",
            "input_sha256",
            "teacher_probs",
        }:
            raise ValueError("Malformed inline teacher row")
        identifier = entry["id"]
        if not isinstance(identifier, str) or identifier not in by_id:
            raise ValueError("Inline teacher row is absent from TRAIN")
        if identifier in teachers:
            raise ValueError("Duplicate inline teacher row")
        row = by_id[identifier]
        if entry["input_sha256"] != row["input_sha256"]:
            raise ValueError("Inline teacher native input hash differs")
        keys = [option["key"] for option in row["options"]]
        probabilities = entry["teacher_probs"]
        if not isinstance(probabilities, dict) or set(probabilities) != set(keys):
            raise ValueError("Inline teacher option keys differ")
        if (
            any(
                type(probabilities[key]) not in (int, float)
                or not math.isfinite(probabilities[key])
                or probabilities[key] < 0
                for key in keys
            )
            or abs(sum(probabilities.values()) - 1.0) > 1e-5
        ):
            raise ValueError("Inline teacher probabilities are invalid")
        selected.append(row)
        teachers[identifier] = probabilities
    if roster_sha256(selected) != expected_roster_sha256:
        raise ValueError("Inline teacher rows or order differ from frozen roster")
    # The source artifact is known-valid in full before mutating any row.
    for row in selected:
        row["teacher_probs"] = teachers[row["id"]]
    return len(selected)
