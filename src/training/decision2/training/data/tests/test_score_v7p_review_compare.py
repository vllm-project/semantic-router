"""Sealed blind review must bind packet, key, coverage and realism decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from training.data.score_v7p_build import build
from training.data.score_v7p_quality import write
from training.data.score_v7p_review_compare import compare
from training.model.data import file_sha256


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> argparse.Namespace:
    train = tmp_path / "train.jsonl"
    select = tmp_path / "select.jsonl"
    _jsonl(train, build(b"a" * 32, "train", 80))
    _jsonl(select, build(b"b" * 32, "select", 80))
    packet_dir = tmp_path / "packet"
    write(argparse.Namespace(new_train=train, new_select=select, output_dir=packet_dir))
    items, groups, manual_groups = [], [], {}
    for role in ("train", "select"):
        key = json.loads((packet_dir / f"{role}-sealed-key.json").read_text())
        for line in (
            (packet_dir / f"{role}-blind-packet.jsonl").read_text().splitlines()
        ):
            group = json.loads(line)
            answers = []
            for item in group["items"]:
                answer = str(key[item["review_id"]]["label"])
                answers.append(answer)
                items.append(
                    {
                        "split": role,
                        "group": group["review_group"],
                        "review_id": item["review_id"],
                        "answer": answer,
                        "ambiguous": False,
                        "evidence_sufficient": True,
                        "blind_text_sha256": hashlib.sha256(
                            json.dumps(
                                item, sort_keys=True, ensure_ascii=False
                            ).encode()
                        ).hexdigest(),
                    }
                )
            groups.append(
                {
                    "split": role,
                    "group": group["review_group"],
                    "answerable": True,
                    "ambiguous": False,
                    "realism": "limited: abstract template",
                    "answers_in_packet_order": answers,
                }
            )
            if len(manual_groups) < 8:
                manual_groups[group["review_group"]] = answers
    reviewer_script = tmp_path / "reviewer.py"
    reviewer_script.write_text("# test reviewer\n", encoding="utf-8")
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "schema": "decision2-score-v7p-independent-blind-review/1",
                "decision": "HOLD_REALISM_REVIEW",
                "reviewer_script_sha256": file_sha256(reviewer_script),
                "input_sha256": {
                    role: file_sha256(packet_dir / f"{role}-blind-packet.jsonl")
                    for role in ("train", "select")
                },
                "items": items,
                "groups": groups,
            }
        ),
        encoding="utf-8",
    )
    manual = tmp_path / "manual.json"
    manual.write_text(
        json.dumps(
            {
                "schema": "decision2-score-v7p-manual-spot/1",
                "primary_receipt_sha256": file_sha256(review),
                "item_count": 24,
                "groups": manual_groups,
            }
        ),
        encoding="utf-8",
    )
    return argparse.Namespace(
        packet_dir=packet_dir,
        quality_report=packet_dir / "quality-report.json",
        review=review,
        review_sha256=file_sha256(review),
        manual_spot=manual,
        manual_spot_sha256=file_sha256(manual),
        reviewer_script=reviewer_script,
        output=tmp_path / "comparison.json",
    )


def test_sealed_review_matches_answers_but_realism_stays_hold(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    result = compare(args)
    assert result["status"] == "HOLD_REALISM"
    assert result["item_count"] == 480
    assert result["answer_mismatch_count"] == 0
    assert result["realism_concern_groups"] == 160
    assert result["manual_spot_items"] == 24


def test_receipt_sha_change_cannot_be_rescored_silently(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    review = json.loads(args.review.read_text())
    review["items"][0]["answer"] = "2" if review["items"][0]["answer"] != "2" else "0"
    args.review.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt SHA differs"):
        compare(args)
