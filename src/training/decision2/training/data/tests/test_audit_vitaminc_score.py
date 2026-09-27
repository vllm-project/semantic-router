"""Fail-closed contracts for the aggregate VitaminC source screen."""

from __future__ import annotations

import json
import zipfile

import pytest

from training.data import audit_vitaminc_score as audit


def record(
    case: str, page: str, claim: str, evidence: str, label: str
) -> dict[str, str]:
    return {
        "unique_id": f"{case}-{evidence}",
        "case_id": case,
        "wiki_revision_id": f"revision-{case}",
        "label": label,
        "claim": claim,
        "evidence": evidence,
        "page": page,
        "revision_type": "real",
    }


def test_real_cases_keep_contrastive_claim_pairs_and_page_groups() -> None:
    cases = {
        "c1": [
            record("c1", "p1", "claim one", "old claim one", "REFUTES"),
            record("c1", "p1", "claim one", "new claim one", "SUPPORTS"),
        ],
        "c2": [
            record("c2", "p2", "claim two", "old claim two", "NOT ENOUGH INFO"),
            record("c2", "p2", "claim two", "new claim two", "SUPPORTS"),
        ],
        "c3": [
            record("c3", "p1", "another claim", "old other", "REFUTES"),
            record("c3", "p1", "another claim", "new other", "SUPPORTS"),
        ],
    }
    rows, summary = audit.select_cases(cases, per_stratum=1)
    assert summary["selected_cases"] == 2
    assert summary["selected_pages"] == 2
    assert summary["selected_labels"] == {
        "REFUTES": 1,
        "NOT ENOUGH INFO": 1,
        "SUPPORTS": 2,
    }
    assert summary["claim_only_memorization_upper_bound"] == 0.5
    assert len(rows) == 4


def test_one_label_or_mixed_source_rejected() -> None:
    rows = [
        record("a", "p", "claim", "old", "SUPPORTS"),
        record("a", "p", "claim", "new", "SUPPORTS"),
    ]
    assert audit.eligible_case(rows) is None
    rows[1]["revision_type"] = "synthetic"
    assert audit.eligible_case(rows) is None


def test_archive_sha_and_schema_fail_closed(tmp_path, monkeypatch) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr(
            "vitaminc/train.jsonl",
            json.dumps(record("a", "p", "c", "e", "SUPPORTS")) + "\n",
        )
        source.writestr("vitaminc/dev.jsonl", "not opened")
    with pytest.raises(ValueError, match="SHA-256"):
        audit.load_train(archive)
    monkeypatch.setattr(audit, "PUBLISHER_ZIP_SHA256", audit.sha_file(archive))
    cases, summary = audit.load_train(archive)
    assert summary["rows"] == 1
    assert len(cases) == 1
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("vitaminc/train.jsonl", '{"unexpected":1}\n')
    monkeypatch.setattr(audit, "PUBLISHER_ZIP_SHA256", audit.sha_file(archive))
    with pytest.raises(ValueError, match="schema"):
        audit.load_train(archive)


def test_overlap_quarantines_whole_page_group() -> None:
    rows = [
        record(
            "a",
            "page-a",
            "a supported statement here",
            "an exact protected evidence passage",
            "SUPPORTS",
        ),
        record("b", "page-a", "other claim", "another evidence paragraph", "REFUTES"),
    ]
    screened = audit.overlap_screen(
        rows, [("typed_final_goldfree", "an exact protected evidence passage")]
    )
    assert screened["suspected_page_groups_total"] == 1
    assert (
        screened["suspected_page_groups_by_role"]["typed_final_goldfree"]["exact"] == 1
    )
    assert "page-a" not in str(screened)


def test_protected_manifest_hash_and_answer_fields_fail_closed(
    tmp_path, monkeypatch
) -> None:
    role = tmp_path / "role.jsonl"
    role.write_text(
        json.dumps({"state": "the protected prompt", "answer": "secret"}) + "\n"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"role": name, "path": str(role), "sha256": audit.sha_file(role)}
                for name in sorted(audit.REQUIRED_ROLES)
            ]
        )
    )
    with pytest.raises(ValueError, match="SHA-256"):
        audit.protected_texts(manifest)
    monkeypatch.setattr(audit, "PROTECTED_INVENTORY_SHA256", audit.sha_file(manifest))
    with pytest.raises(ValueError, match="answer-bearing"):
        audit.protected_texts(manifest)
