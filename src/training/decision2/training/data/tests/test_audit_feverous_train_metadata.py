"""Synthetic publisher-layout tests; no source rows are stored here."""

import hashlib
import json

import pytest

from training.data.audit_feverous_train_metadata import audit, normalized_claim


def _row(source_id, label, page):
    element = f"{page}_sentence_0"
    return {
        "id": source_id,
        "label": label,
        "claim": f"Claim {source_id}",
        "evidence": [{"content": [element], "context": {element: []}}],
    }


def _source(tmp_path, rows):
    data = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    path = tmp_path / "train.jsonl"
    path.write_bytes(data)
    return path, data


def _audit(path, data):
    return audit(
        path,
        expected_bytes=len(data),
        expected_md5=hashlib.md5(data, usedforsecurity=False).hexdigest(),
    )


def test_empty_label_is_counted_and_not_a_candidate(tmp_path):
    unlabeled = _row(2, "", "b")
    unlabeled["id"] = "publisher-placeholder"
    path, data = _source(tmp_path, [_row(1, "SUPPORTS", "a"), unlabeled])
    report = _audit(path, data)
    assert report["empty_label_quarantine"] == 1
    assert report["total_source_rows"] == 2
    assert report["text_only_rows"] == {"SUPPORTS": 1}
    assert report["metadata_floor_pass"] is False
    assert report["train_admission"] == "HOLD"


def test_duplicate_id_and_unknown_label_fail_closed(tmp_path):
    path, data = _source(tmp_path, [_row(1, "SUPPORTS", "a"), _row(1, "REFUTES", "b")])
    with pytest.raises(ValueError, match="duplicate"):
        _audit(path, data)
    path, data = _source(tmp_path, [_row(1, "UNCERTAIN", "a")])
    with pytest.raises(ValueError, match="nonempty"):
        _audit(path, data)


def test_page_disjoint_roster_and_claim_normalization(tmp_path):
    rows = [_row(1, "SUPPORTS", "shared"), _row(2, "REFUTES", "shared")]
    path, data = _source(tmp_path, rows)
    report = _audit(path, data)
    assert sum(report["page_disjoint_candidate_rows"].values()) == 1
    assert report["page_linked_text_only_components"] == 1
    assert normalized_claim("  A  B  ") == "a b"


def test_wrong_publisher_digest_fails_before_parsing(tmp_path):
    path, data = _source(tmp_path, [_row(1, "SUPPORTS", "a")])
    with pytest.raises(ValueError, match="identity"):
        audit(path, expected_bytes=len(data), expected_md5="0" * 32)
