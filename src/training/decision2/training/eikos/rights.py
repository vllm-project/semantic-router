"""Bind Eikos data receipts to either clean data or noncommercial research terms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.model.data import file_sha256

SCHEMA = "decision2-rights-clean-splits/1"
SCOPE = "trained weights/modelcard only; no raw source rows, SELECT/CAL rows, or individual text predictions"
SPLIT_NAMES = ("rights_clean.train.jsonl", "select.jsonl", "cal.jsonl")
RESTRICTED_PILOT_PARTITIONS = frozenset(
    {
        "d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38",
        "bf5bbf29693928a2559ce0aff10e9d6b5b1541b50698634fcdb7725902412dcf",
    }
)
NONCOMMERCIAL_SCHEMA = "decision2-noncommercial-research-attestation/1"
NONCOMMERCIAL_SCOPE = "noncommercial research model weights/modelcard only; no raw rows"
PILOT_SCHEMA = "decision2-balanced-human-5824/1"
PILOT_SPLIT_NAMES = ("balanced_human_5824.train.jsonl", "select.jsonl", "cal.jsonl")
PILOT_HOLDOUT_SOURCE_COUNTS = {
    "select": {
        "css_pilot:semeval_stance": 200,
        "css_pilot:implicit_hate": 200,
        "css_pilot:discourse": 200,
    },
    "cal": {
        "css_pilot:semeval_stance": 100,
        "css_pilot:implicit_hate": 100,
        "css_pilot:discourse": 100,
        "decision2_cal_hard_original_v2": 600,
    },
}


def check_statement(
    statement: dict[str, Any],
    manifest_sha: str,
    provenance_sha: str,
    data_sha256: dict[str, str],
    sources: dict[str, int],
) -> None:
    source_groups = statement.get("source_groups")
    holdout_groups = statement.get("holdout_groups")
    conditions = statement.get("rights_conditions")
    if (
        statement.get("schema_version") != NONCOMMERCIAL_SCHEMA
        or statement.get("noncommercial_use") is not True
        or statement.get("publication_scope") != NONCOMMERCIAL_SCOPE
        or statement.get("no_raw_training_rows") is not True
        or statement.get("data_manifest_sha256") != manifest_sha
        or statement.get("training_provenance_sha256") != provenance_sha
        or statement.get("data_sha256") != data_sha256
        or not isinstance(sources, dict)
        or not isinstance(source_groups, dict)
        or set(source_groups) != set(sources)
        or statement.get("holdout_source_counts") != PILOT_HOLDOUT_SOURCE_COUNTS
        or not isinstance(holdout_groups, dict)
        or set(holdout_groups) != set(PILOT_HOLDOUT_SOURCE_COUNTS)
        or any(
            set(holdout_groups[name]) != set(counts)
            for name, counts in PILOT_HOLDOUT_SOURCE_COUNTS.items()
        )
        or not isinstance(conditions, dict)
        or not conditions
        or any(
            not isinstance(value, str) or value not in conditions
            for value in source_groups.values()
        )
        or any(
            not isinstance(value, str) or value not in conditions
            for groups in holdout_groups.values()
            for value in groups.values()
        )
        or any(
            not isinstance(value, dict)
            or not isinstance(value.get("terms"), str)
            or not value["terms"]
            or not isinstance(value.get("evidence"), str)
            or not value["evidence"]
            for value in conditions.values()
        )
    ):
        raise ValueError(
            "Noncommercial research attestation is missing source rights or exact data binding"
        )


def verify_output_bindings(
    receipt: dict[str, Any], expected_sha: dict[str, str], expected_rows: dict[str, int]
) -> None:
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(expected_sha):
        raise ValueError("Data receipt lacks exact TRAIN/SELECT/CAL outputs")
    for name, expected in expected_sha.items():
        item = outputs[name]
        if (
            not isinstance(item, dict)
            or item.get("sha256") != expected
            or item.get("rows") != expected_rows[name]
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] < 1
        ):
            raise ValueError(f"Data receipt differs for {name}")


def clean_manifest(
    path: Path, expected_sha: dict[str, str], expected_rows: dict[str, int]
) -> dict[str, Any]:
    if RESTRICTED_PILOT_PARTITIONS.intersection(expected_sha.values()):
        raise ValueError("Restricted pilot SELECT/CAL cannot qualify a release")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != SCHEMA
        or receipt.get("publication_eligible") is not True
        or receipt.get("publication_scope") != SCOPE
    ):
        raise ValueError("Eikos release requires the rights-cleared split manifest")
    if (
        set(expected_sha) != set(SPLIT_NAMES)
        or set(expected_rows) != set(SPLIT_NAMES)
        or not isinstance(receipt.get("source_rights"), (list, dict))
        or not receipt["source_rights"]
        or not receipt.get("publication_conditions")
        or not receipt.get("overlap_audits")
    ):
        raise ValueError("Clean split receipt lacks rights or isolation evidence")
    verify_output_bindings(receipt, expected_sha, expected_rows)
    return receipt


def verify_clean_files(
    manifest_path: Path,
    train: Path,
    select: Path,
    cal: Path,
    counts: tuple[int, int, int],
) -> dict[str, Any]:
    paths = (train, select, cal)
    if tuple(path.name for path in paths) != SPLIT_NAMES:
        raise ValueError("Unexpected rights-cleared split filenames")
    expected_sha = {path.name: file_sha256(path) for path in paths}
    expected_rows = dict(zip(SPLIT_NAMES, counts))
    receipt = clean_manifest(manifest_path, expected_sha, expected_rows)
    for path in paths:
        if receipt["outputs"][path.name].get("bytes") != path.stat().st_size:
            raise ValueError(f"Rights-cleared split byte size differs for {path.name}")
    return receipt


def verify_noncommercial_attestation(
    attestation_path: Path,
    manifest_path: Path,
    provenance_path: Path,
    data_sha256: dict[str, str],
    counts: tuple[int, int, int],
) -> dict[str, Any]:
    """Check a separately disclosed research-use statement for the old pilot.

    The original split manifest did not record rights. The attestation covers
    every manifest source key, binds exact old splits and training provenance,
    and states the upstream conditions without redistributing source rows.
    """
    receipt = json.loads(manifest_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != PILOT_SCHEMA:
        raise ValueError("Expected the original balanced-human pilot manifest")
    expected_sha = dict(
        zip(
            PILOT_SPLIT_NAMES,
            (
                data_sha256["train"],
                data_sha256["select"],
                data_sha256["cal_audited_only"],
            ),
        )
    )
    verify_output_bindings(receipt, expected_sha, dict(zip(PILOT_SPLIT_NAMES, counts)))
    statement = json.loads(attestation_path.read_text(encoding="utf-8"))
    check_statement(
        statement,
        file_sha256(manifest_path),
        file_sha256(provenance_path),
        data_sha256,
        receipt.get("counts", {}).get("source"),
    )
    return statement


def verify_noncommercial_package_attestation(
    attestation_path: Path, package_receipt: dict[str, Any]
) -> dict[str, Any]:
    """Bind an old merged package to an external, disclosed research-use statement."""
    statement = json.loads(attestation_path.read_text(encoding="utf-8"))
    try:
        check_statement(
            statement,
            package_receipt.get("training_data_manifest_sha256"),
            package_receipt.get("training_provenance_sha256"),
            package_receipt.get("training_data_sha256"),
            package_receipt.get("training_source_counts"),
        )
    except ValueError as exc:
        raise ValueError(
            "Package lacks a complete exact noncommercial-use attestation"
        ) from exc
    return statement
