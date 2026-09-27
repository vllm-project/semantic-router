"""Synthetic contracts for the official RACE TRAIN source screen."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.data import audit_race06_choice_source as race
from training.model.decision_model import segments as production_segments


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "RACE"
    for level in ("middle", "high"):
        directory = root / "train" / level
        directory.mkdir(parents=True)
        for ordinal in range(2):
            article = f"The {level} article {ordinal} states that a permit is required."
            payload = {
                "id": f"{level}-{ordinal}",
                "article": article,
                "questions": ["What is required?"],
                "options": [["A permit", "A ticket", "Nothing", "A badge"]],
                "answers": ["A"],
            }
            (directory / f"{ordinal}.txt").write_text(json.dumps(payload))
    return root


def test_source_groups_sample_and_blind_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _source(tmp_path)
    monkeypatch.setattr(race, "SOURCE_SPLITS", {"middle": 2, "high": 2})
    rows, profile = race.read_train(root)
    assert profile["passages"] == 4
    assert profile["questions_by_level"] == {"middle": 2, "high": 2}
    assert profile["independent_normalized_passages"] == 4
    selected = race.choose_passages(rows, 1)
    assert len(selected) == 2
    assert selected == race.choose_passages(rows, 1)
    packet, key = race.review_packet(selected)
    assert len(packet) == 4
    assert len(key) == 2
    assert {row["condition"] for row in packet} == {"present", "removed"}
    assert "source_answer" not in json.dumps(packet)
    assert all(row["source_answer"] == "A" for row in key)


def test_malformed_source_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _source(tmp_path)
    monkeypatch.setattr(race, "SOURCE_SPLITS", {"middle": 2, "high": 2})
    path = root / "train" / "middle" / "0.txt"
    row = json.loads(path.read_text())
    row["options"][0] = ["one", "two", "three"]
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Malformed RACE TRAIN question"):
        race.read_train(root)


def test_private_receipt_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "private" / "receipt.json"
    race._write_private(path, {"count": 1})
    assert path.stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        race._write_private(path, {"count": 2})


def test_source_empty_question_passage_is_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _source(tmp_path)
    monkeypatch.setattr(race, "SOURCE_SPLITS", {"middle": 2, "high": 2})
    path = root / "train" / "middle" / "0.txt"
    row = json.loads(path.read_text())
    row.update(questions=[], options=[], answers=[])
    path.write_text(json.dumps(row))
    rows, profile = race.read_train(root)
    assert len(rows) == 3
    assert profile["excluded_empty_question_passages"] == {"middle": 1}


def test_torch_free_renderer_matches_production(tmp_path: Path) -> None:
    source = _source(tmp_path)
    row = json.loads((source / "train" / "middle" / "0.txt").read_text())
    passage = race.Passage(
        row["id"],
        "middle",
        row["article"],
        tuple(row["questions"]),
        tuple(tuple(slate) for slate in row["options"]),
        tuple(row["answers"]),
        race._sha(race._norm(row["article"])),
    )
    race.verify_renderer_identity()
    choice = race.choice_row(passage, 0, with_passage=True)
    assert race.native_segments(choice) == production_segments(choice)
