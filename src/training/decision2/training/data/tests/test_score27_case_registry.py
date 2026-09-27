"""Small private-file fixtures for the prospective 27B source gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.data.score27_case_registry import audit_registry, file_sha256


def _private_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def _entry(root: Path, name: str, role: str = "train") -> dict:
    documents = []
    for number in (1, 2):
        source = root / f"{name}-source-{number}.txt"
        source.write_text(f"Distinct source {name} number {number}", encoding="utf-8")
        source.chmod(0o600)
        documents.append(
            {
                "source_id": f"{name}-{number}",
                "rights_id": "original-authoring-ledger",
                "origin": "authored",
                "path": str(source),
                "sha256": file_sha256(source),
            }
        )
    case = {
        "case_id": name,
        "role": role,
        "mechanism": "dependency_readiness",
        "author_id": f"{role}-author",
        "source_family_id": f"family-{name}",
        "documents": documents,
        "variants": [
            {
                "group_id": name,
                "task_type": "score",
                "label": level,
                "state": f"The relevant source facts yield level {level}.",
                "instructions": "Select the level justified by the source facts.",
                "options": [
                    {"key": str(key), "description": f"Ordinal level {key}"}
                    for key in range(3)
                ],
                "structured_facts": {"level": level},
            }
            for level in range(3)
        ],
    }
    path = root / f"{name}.json"
    _private_json(path, case)
    return {"path": str(path), "sha256": file_sha256(path)}


def test_partial_registry_is_a_hold_with_exact_missing_counts(tmp_path: Path) -> None:
    receipt = audit_registry([_entry(tmp_path, "case-one")])
    assert receipt["status"] == "HOLD_CANDIDATE_INCOMPLETE"
    assert receipt["case_groups"] == 1
    assert receipt["rows"] == 3
    assert receipt["source_documents"] == 2
    assert receipt["missing_groups"]["train"]["dependency_readiness"] == 13
    assert receipt["missing_groups"]["select"]["dependency_readiness"] == 14


def test_tampered_source_is_rejected(tmp_path: Path) -> None:
    entry = _entry(tmp_path, "case-one")
    (tmp_path / "case-one-source-1.txt").write_text("Changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source digest differs"):
        audit_registry([entry])


def test_cross_role_author_and_source_reuse_are_rejected(tmp_path: Path) -> None:
    train = _entry(tmp_path, "train-case")
    select = _entry(tmp_path, "select-case", "select")
    select_path = Path(select["path"])
    case = json.loads(select_path.read_text())
    case["author_id"] = "train-author"
    _private_json(select_path, case)
    select["sha256"] = file_sha256(select_path)
    with pytest.raises(ValueError, match="distinct authors"):
        audit_registry([train, select])
    case["author_id"] = "select-author"
    case["documents"][0]["source_id"] = "train-case-1"
    _private_json(select_path, case)
    select["sha256"] = file_sha256(select_path)
    with pytest.raises(ValueError, match="document recurs"):
        audit_registry([train, select])


def test_registry_cannot_claim_admission_with_complete_metadata(tmp_path: Path) -> None:
    # File-based fixture avoids using training labels or benchmark material.
    entries = []
    for role in ("train", "select"):
        for mechanism, count in (
            ("dependency_readiness", 14),
            ("timed_feasibility", 14),
            ("stock_uncertainty", 13),
            ("scoped_policy_exception", 13),
            ("multi_source_attestation", 13),
            ("state_reconciliation", 13),
        ):
            for number in range(count):
                name = f"{role}-{mechanism}-{number}"
                entry = _entry(tmp_path, name, role)
                path = Path(entry["path"])
                case = json.loads(path.read_text())
                case["mechanism"] = mechanism
                _private_json(path, case)
                entry["sha256"] = file_sha256(path)
                entries.append(entry)
    receipt = audit_registry(entries)
    assert receipt["case_groups"] == 160
    assert receipt["rows"] == 480
    assert receipt["status"] == "HOLD_PENDING_ORACLE_OVERLAP_TOKENS_AND_BLIND_REVIEW"
    assert (
        sum(sum(counts.values()) for counts in receipt["missing_groups"].values()) == 0
    )
