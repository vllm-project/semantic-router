"""Fail-closed rights binding for portable Decision 2.0 model packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CLEAN_SCHEMA = "decision2-rights-clean-splits/1"
CLEAN_SCOPE = "trained weights/modelcard only; no raw source rows, SELECT/CAL rows, or individual text predictions"
NONCOMMERCIAL_SCHEMA = "decision2-noncommercial-research-attestation/1"
NONCOMMERCIAL_SCOPE = "noncommercial research model weights/modelcard only; no raw rows"
RESEARCH_DATA_SCHEMAS = frozenset(
    {"decision2-balanced-human-5824/1", "decision2-nox4b-structured-replay/1"}
)
OLD_HOLDOUT_SHA = {
    "select": "d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38",
    "cal": "bf5bbf29693928a2559ce0aff10e9d6b5b1541b50698634fcdb7725902412dcf",
}
OLD_HOLDOUT_COUNTS = {
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_rights(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("Clean data manifest lacks source rights")
    for entry in value:
        if (
            not isinstance(entry, dict)
            or not all(
                isinstance(entry.get(key), str) and entry[key]
                for key in ("source", "license", "evidence")
            )
            or entry.get("partition_scope", "TRAIN") not in {"TRAIN", "SELECT/CAL"}
            or type(entry.get("rows")) is not int
            or entry["rows"] < 1
        ):
            raise ValueError("Clean data manifest has incomplete source rights")


def verify_rights(
    *,
    data: dict[str, Any],
    data_manifest_path: Path,
    run_provenance_path: Path,
    partition_sha: dict[str, str],
    partition_rows: dict[str, int],
    source_counts: dict[str, int],
    attestation_path: Path | None,
    license_id: str,
) -> dict[str, Any]:
    """Return public-safe rights details after exact data/run/source binding."""
    schema = data.get("schema_version")
    if schema == CLEAN_SCHEMA:
        if attestation_path is not None:
            raise ValueError(
                "Clean release must use its source manifest, not a research-use override"
            )
        if (
            data.get("publication_eligible") is not True
            or data.get("publication_scope") != CLEAN_SCOPE
            or not isinstance(data.get("overlap_audits"), dict)
            or not data["overlap_audits"]
            or not isinstance(data.get("publication_conditions"), list)
            or not data["publication_conditions"]
            or any(
                not isinstance(note, str) or not note.strip()
                for note in data["publication_conditions"]
            )
        ):
            raise ValueError("Clean data lacks audited publication eligibility")
        if any(partition_sha[role] == old for role, old in OLD_HOLDOUT_SHA.items()):
            raise ValueError("Old CSS pilot holdouts cannot qualify a clean release")
        _source_rights(data.get("source_rights"))
        train_rights_rows = sum(
            item["rows"]
            for item in data["source_rights"]
            if item.get("partition_scope", "TRAIN") == "TRAIN"
        )
        if train_rights_rows != partition_rows["train"]:
            raise ValueError("Clean data source rights do not cover every TRAIN row")
        holdout_rights_rows = sum(
            item["rows"]
            for item in data["source_rights"]
            if item.get("partition_scope") == "SELECT/CAL"
        )
        if holdout_rights_rows != partition_rows["select"] + partition_rows["cal"]:
            raise ValueError("Clean data source rights do not cover SELECT/CAL rows")
        holdout_sources = {}
        counts = data.get("partition_counts")
        if not isinstance(counts, dict):
            raise ValueError("Clean data lacks separate SELECT/CAL source counts")
        for role in ("select", "cal"):
            sources = counts.get(role, {}).get("source")
            if (
                not isinstance(sources, dict)
                or not sources
                or any(
                    type(value) is not int or value < 1 for value in sources.values()
                )
                or sum(sources.values()) != partition_rows[role]
            ):
                raise ValueError(
                    f"Clean data {role} source counts differ from frozen split"
                )
            holdout_sources[role] = sources
        return {
            "mode": "rights_clean",
            "publication_scope": CLEAN_SCOPE,
            "conditions": data["publication_conditions"],
            "data_manifest_sha256": _sha(data_manifest_path),
            "attestation_sha256": None,
            "holdout_source_counts": holdout_sources,
        }
    if schema not in RESEARCH_DATA_SCHEMAS:
        raise ValueError("Training data has no recognized rights release path")
    if (
        attestation_path is None
        or attestation_path.is_symlink()
        or not attestation_path.is_file()
    ):
        raise ValueError("Noncommercial pilot requires an exact rights attestation")
    if license_id != "other":
        raise ValueError(
            "Noncommercial research package license metadata must be 'other'"
        )
    if (
        partition_sha["select"] != OLD_HOLDOUT_SHA["select"]
        or partition_sha["cal"] != OLD_HOLDOUT_SHA["cal"]
    ):
        raise ValueError(
            "Pilot attestation is only defined for the frozen CSS SELECT/CAL"
        )
    statement = json.loads(attestation_path.read_text(encoding="utf-8"))
    groups = statement.get("source_groups")
    holdout_groups = statement.get("holdout_groups")
    conditions = statement.get("rights_conditions")
    if (
        statement.get("schema_version") != NONCOMMERCIAL_SCHEMA
        or statement.get("noncommercial_use") is not True
        or statement.get("publication_scope") != NONCOMMERCIAL_SCOPE
        or statement.get("no_raw_training_rows") is not True
        or statement.get("data_manifest_sha256") != _sha(data_manifest_path)
        or statement.get("training_provenance_sha256") != _sha(run_provenance_path)
        or statement.get("data_sha256") != partition_sha
        or statement.get("source_counts") != source_counts
        or not isinstance(groups, dict)
        or set(groups) != set(source_counts)
        or statement.get("holdout_source_counts") != OLD_HOLDOUT_COUNTS
        or not isinstance(holdout_groups, dict)
        or set(holdout_groups) != set(OLD_HOLDOUT_COUNTS)
        or any(
            set(holdout_groups[role]) != set(counts)
            for role, counts in OLD_HOLDOUT_COUNTS.items()
        )
        or not isinstance(conditions, dict)
        or not conditions
        or any(
            not isinstance(value, str) or value not in conditions
            for value in groups.values()
        )
        or any(
            not isinstance(value, str) or value not in conditions
            for roles in holdout_groups.values()
            for value in roles.values()
        )
        or any(
            not isinstance(value, dict)
            or not isinstance(value.get("terms"), str)
            or not value["terms"].strip()
            or not isinstance(value.get("evidence"), str)
            or not value["evidence"].strip()
            for value in conditions.values()
        )
        or not isinstance(statement.get("limitations", []), list)
        or any(
            not isinstance(note, str) or not note.strip()
            for note in statement.get("limitations", [])
        )
    ):
        raise ValueError(
            "Noncommercial rights attestation differs from exact run/data/source receipts"
        )
    return {
        "mode": "noncommercial_research",
        "publication_scope": NONCOMMERCIAL_SCOPE,
        "conditions": conditions,
        "data_manifest_sha256": _sha(data_manifest_path),
        "attestation_sha256": _sha(attestation_path),
        "holdout_source_counts": OLD_HOLDOUT_COUNTS,
        "source_groups": groups,
        "holdout_groups": holdout_groups,
        "limitations": statement.get("limitations", []),
    }
