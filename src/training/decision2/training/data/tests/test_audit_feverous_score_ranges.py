"""Synthetic records only; no publisher text or protected keys."""

import json
from unittest.mock import patch

import pytest

from training.data import audit_feverous_score_ranges as screen


def _row(source_id, label, element):
    return {
        "id": source_id,
        "label": label,
        "claim": "Synthetic claim",
        "evidence": [{"content": [element], "context": {element: ["page_title"]}}],
    }


def _chunk(rows):
    body = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    header = f"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-{len(body) - 1}/{len(body)}\r\n".encode()
    return header, body


def test_text_only_requires_a_complete_sentence_evidence_set():
    assert screen.evidence_profile(_row(1, "SUPPORTS", "page_sentence_0")) == (
        {"sentence"},
        {"page"},
    )
    assert screen.evidence_profile(_row(2, "REFUTES", "page_cell_0_1_1")) == (
        {"cell"},
        None,
    )


def test_missing_context_cannot_be_text_only():
    row = _row(1, "SUPPORTS", "page_sentence_0")
    row["evidence"][0]["context"] = {}
    assert screen.evidence_profile(row)[1] is None


def test_incomplete_boundary_rows_are_dropped():
    row = json.dumps(_row(1, "SUPPORTS", "page_sentence_0")).encode()
    assert screen.complete_rows(b"partial\n" + row + b"\ntruncated", 17) == [
        json.loads(row)
    ]


def test_http_range_identity_is_mandatory():
    header, body = _chunk([_row(1, "SUPPORTS", "page_sentence_0")])
    with patch.object(screen, "RANGE_BYTES", len(body)), patch.object(
        screen, "TOTAL_BYTES", len(body)
    ):
        assert len(screen.check_range(header, body, 0)) == 64
        with pytest.raises(ValueError, match="identity"):
            screen.check_range(header, body, 1)


def test_unlabeled_source_row_forces_hold_even_with_valid_rows():
    rows = [
        _row(1, "SUPPORTS", "page_sentence_0"),
        _row(2, "", "other_sentence_0"),
    ]
    header, body = _chunk(rows)
    with patch.object(screen, "RANGE_BYTES", len(body)), patch.object(
        screen, "TOTAL_BYTES", len(body)
    ):
        report = screen.aggregate([(header, body)], offsets=(0,))
    assert report["invalid_label_rows"] == 1
    assert report["source_label_counts"]["SUPPORTS"] == 1
    assert report["small_screen_floor_pass"] is False
    assert report["train_admission"] == "HOLD"
