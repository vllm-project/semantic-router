"""Blind human-response forms preserve native type and chronology gates."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jev_arena.authored_release_scale_v1 import file_sha, write_private
from jev_arena.authored_release_scale_v2_review import seal, template


def _private_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path.chmod(0o600)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    packet = tmp_path / "original-review-a.private.jsonl"
    _private_rows(
        packet,
        [
            {
                "review_id": "one",
                "state": "Two records",
                "questions": {
                    "decision": {
                        "type": "choice",
                        "criteria": {"Z": "first", "A": "second"},
                    }
                },
            },
            {
                "review_id": "two",
                "state": "Two records",
                "questions": {
                    "decision": {"type": "noul", "criteria": {"true": "yes"}}
                },
            },
            {
                "review_id": "three",
                "state": "Two records",
                "questions": {
                    "decision": {"type": "score", "criteria": ["none", "some", "all"]}
                },
            },
        ],
    )
    receipt = tmp_path / "receipt.private.json"
    write_private(
        receipt,
        {
            "status": "BLIND_PACKET_SEALED_REVIEW_PENDING",
            "packet_sha256": {"original-review-a": file_sha(packet)},
            "reviewer_assignments": 0,
            "human_reviews_completed": 0,
            "sealed_at_utc": (
                datetime.now(timezone.utc) - timedelta(minutes=1)
            ).isoformat(),
        },
    )
    answer_file = tmp_path / "answers.private.jsonl"
    template(packet, receipt, "original-review-a", answer_file)
    reviewer = tmp_path / "reviewer.private.txt"
    reviewer.write_text("Reviewer R1\n")
    reviewer.chmod(0o600)
    return packet, receipt, answer_file, reviewer


def _completed(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row, answer in zip(rows, ("Z", False, 2)):
        row.update(
            native_answer=answer,
            source_a_evidence="Source A paragraph 1",
            source_b_evidence="Source B paragraph 2",
            both_sources_necessary=True,
            ambiguity="none",
            document_realism="plausible",
            shortcut_risk="none",
            rights_concern=False,
            all_paragraphs_checked=True,
            paragraph_notes="Both paragraphs carry causal evidence.",
        )
    _private_rows(path, rows)
    return rows


def test_review_template_and_seal_all_native_types(tmp_path: Path) -> None:
    packet, receipt, answers, reviewer = _fixture(tmp_path)
    rows = _completed(answers)
    result = seal(
        packet, receipt, "original-review-a", answers, reviewer, tmp_path / "sealed"
    )
    assert result["reviewed_rows"] == 3
    assert result["quality_flagged_rows"] == 0
    assert rows[0]["native_answer"] == "Z"
    meta = json.loads((tmp_path / "sealed" / "receipt.private.json").read_text())
    assert meta["review_sha256"] == file_sha(answers)
    assert meta["key_opened"] is False
    assert meta["release_qualified"] is False
    with pytest.raises(FileExistsError):
        seal(
            packet, receipt, "original-review-a", answers, reviewer, tmp_path / "sealed"
        )


@pytest.mark.parametrize("bad", (True, "false", 3))
def test_native_answer_type_or_range_is_required(tmp_path: Path, bad: object) -> None:
    packet, receipt, answers, reviewer = _fixture(tmp_path)
    rows = _completed(answers)
    index = 0 if bad is True else 1 if bad == "false" else 2
    rows[index]["native_answer"] = bad
    _private_rows(answers, rows)
    with pytest.raises(ValueError, match="native question"):
        seal(
            packet, receipt, "original-review-a", answers, reviewer, tmp_path / "sealed"
        )


def test_review_must_be_complete_and_packet_unchanged(tmp_path: Path) -> None:
    packet, receipt, answers, reviewer = _fixture(tmp_path)
    rows = _completed(answers)
    _private_rows(answers, rows[:-1])
    with pytest.raises(ValueError, match="Every blind packet row"):
        seal(
            packet, receipt, "original-review-a", answers, reviewer, tmp_path / "sealed"
        )
    _private_rows(answers, rows)
    packet.write_text(packet.read_text() + "\n")
    with pytest.raises(ValueError, match="seal"):
        seal(
            packet, receipt, "original-review-a", answers, reviewer, tmp_path / "sealed"
        )


def test_quality_concern_seals_without_claiming_acceptance(tmp_path: Path) -> None:
    packet, receipt, answers, reviewer = _fixture(tmp_path)
    rows = _completed(answers)
    rows[0]["ambiguity"] = "material"
    rows[0]["both_sources_necessary"] = False
    _private_rows(answers, rows)
    result = seal(
        packet, receipt, "original-review-a", answers, reviewer, tmp_path / "sealed"
    )
    assert result["quality_flagged_rows"] == 1
    assert result["status"] == "HUMAN_REVIEW_SEALED_ADJUDICATION_PENDING"
