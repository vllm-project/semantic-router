"""Input-only projection plan for a future strict protected inventory.

This module has no CLI and does not open a real task artifact on import. The
caller must supply already sealed native prompt files and pinned rights-clean
partitions. Passing these structural checks does not establish data rights,
semantic non-overlap, or equivalence of an altered inference request.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from publication.package_native_arena import input_digest, load_gold_free

from training.data.audit_27b_teacher_admission import _contains_gold_key
from training.model.data import INPUT_FIELDS, canonical, digest, file_sha256

NATIVE_ROLE_COUNTS = {
    "typed_dev": 1600,
    "css_pilot": 1430,
    "typed_final_goldfree": 1600,
    "css15_goldfree": 6547,
    "jevbench_public231": 231,
}
PARTITION_ROLE_COUNTS = {
    "rights_clean_train": ("train", 7455),
    "rights_clean_select": ("select", 700),
    "rights_clean_cal": ("cal", 700),
}
CORE_ROLES = frozenset(NATIVE_ROLE_COUNTS) | frozenset(PARTITION_ROLE_COUNTS)
PROJECTION_FIELDS = {"id", "state", "instructions", "options"}


def _input_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return canonical(value)


def _check_projection(row: dict[str, Any]) -> None:
    if (
        not isinstance(row.get("id"), str)
        or not row["id"]
        or not set(row) <= PROJECTION_FIELDS
        or not isinstance(row.get("state"), str)
        or not row["state"]
        or any(not isinstance(row[field], str) for field in row if field != "id")
        or _contains_gold_key(row)
    ):
        raise ValueError("Projected prompt is not strict input-only text")


def _project_native_rows(
    source: list[dict[str, Any]], expected_rows: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """Reuse the package's actual sealed-prompt loader before flattening.

    The existing native roles can contain *input* names such as question ID
    ``label`` or ``state.target={entity,item}``. These must not be deleted from
    model requests. The separate overlap view stores the entire question
    object as text so no answer-like nested JSON key enters its strict schema.
    """
    if len(source) != expected_rows:
        raise ValueError("Sealed prompt role count changed")
    projected = []
    source_hashes = []
    for row in source:
        result = {
            "id": row["id"],
            "state": _input_text(row["state"]),
            "instructions": canonical(row["questions"]),
        }
        _check_projection(result)
        if json.loads(result["instructions"]) != row["questions"]:
            raise ValueError("Native input changed in protected projection")
        # The input identity belongs in a private sidecar, never the prompt.
        source_hashes.append(input_digest(row))
        projected.append(result)
    if len({row["id"] for row in projected}) != len(projected):
        raise ValueError("Duplicate projected native ID")
    return projected, source_hashes


def project_native_file(
    path: Path, role: str, expected_sha256: str
) -> tuple[list[dict[str, Any]], list[str]]:
    if role not in NATIVE_ROLE_COUNTS:
        raise ValueError("Unsupported sealed native prompt role")
    if file_sha256(path) != expected_sha256:
        raise ValueError("Sealed native prompt hash changed")
    return _project_native_rows(load_gold_free(path), NATIVE_ROLE_COUNTS[role])


def project_partition_row(row: dict[str, Any], partition: str) -> dict[str, Any]:
    """Select only TRAIN/SELECT/CAL *input* fields; never consult ``label``.

    This verifies the source's own input hash without accessing any target
    field. A source-level review must still establish that input content does
    not itself disclose a target.
    """
    if partition not in {"train", "select", "cal"}:
        raise ValueError("Unsupported rights-clean partition")
    required = {"id", "split", "input_sha256", *INPUT_FIELDS}
    if not isinstance(row, dict) or not required.issubset(row):
        raise ValueError("Partition row lacks input identity")
    if row["split"] != partition or row["input_sha256"] != digest(
        {field: row[field] for field in INPUT_FIELDS}
    ):
        raise ValueError("Partition role or input hash changed")
    if not isinstance(row["id"], str) or not row["id"]:
        raise ValueError("Invalid partition row ID")
    result = {
        "id": row["id"],
        "state": _input_text(row["state"]),
        "instructions": canonical(
            {"task_type": row["task_type"], "instructions": row["instructions"]}
        ),
        "options": canonical(row["options"]),
    }
    _check_projection(result)
    return result


def project_partition_rows(
    source: list[dict[str, Any]], role: str
) -> list[dict[str, Any]]:
    if role not in PARTITION_ROLE_COUNTS:
        raise ValueError("Unsupported rights-clean role")
    partition, expected = PARTITION_ROLE_COUNTS[role]
    if len(source) != expected:
        raise ValueError("Rights-clean partition count changed")
    projected = [project_partition_row(row, partition) for row in source]
    if len({row["id"] for row in projected}) != len(projected):
        raise ValueError("Duplicate rights-clean ID")
    return projected


def validate_core_rows(roles: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """Check the eight required roles, uniqueness and strict projected schema."""
    if set(roles) != CORE_ROLES:
        raise ValueError("Core inventory roles are incomplete or unexpected")
    counts = {}
    for role, rows in sorted(roles.items()):
        expected = (
            NATIVE_ROLE_COUNTS[role]
            if role in NATIVE_ROLE_COUNTS
            else PARTITION_ROLE_COUNTS[role][1]
        )
        if len(rows) != expected:
            raise ValueError("Projected role count changed")
        ids = set()
        for row in rows:
            _check_projection(row)
            if row["id"] in ids:
                raise ValueError("Duplicate projected role ID")
            ids.add(row["id"])
        counts[role] = len(rows)
    return counts


def projected_jsonl(rows: list[dict[str, Any]]) -> bytes:
    """Deterministic serialization for a private, versioned future manifest."""
    if not rows:
        raise ValueError("Projected role must not be empty")
    for row in rows:
        _check_projection(row)
    return (
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
        + "\n"
    ).encode("utf-8")
