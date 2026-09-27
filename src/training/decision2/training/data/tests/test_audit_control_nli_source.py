"""ConTRoL source parser and provenance guards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.data import audit_control_nli_source as source


def _source_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "publisher"
    (repo / "data").mkdir(parents=True)
    (repo / "readme.md").write_text(
        "Creative Commons Attribution-NonCommercial-ShareAlike 4.0"
    )
    rows = [
        {
            "uid": "one",
            "premise": "A rule has an exception.",
            "hypothesis": "A is allowed.",
            "label": "e",
        },
        {
            "uid": "two",
            "premise": "A rule has an exception.",
            "hypothesis": "A is blocked.",
            "label": "c",
        },
        {
            "uid": "three",
            "premise": "A second rule is undecided.",
            "hypothesis": "A is allowed.",
            "label": "n",
        },
    ]
    raw = "".join(json.dumps(row) + "\n" for row in rows)
    (repo / "data/train.jsonl").write_text(raw)
    monkeypatch.setattr(
        source, "TRAIN_SHA256", hashlib.sha256(raw.encode()).hexdigest()
    )
    monkeypatch.setattr(
        source.subprocess,
        "check_output",
        lambda args, text: (
            source.PUBLISHER_COMMIT + "\n" if "rev-parse" in args else ""
        ),
    )
    return repo


def test_publisher_mapping_and_premise_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _source_repo(tmp_path, monkeypatch)
    rows, receipt = source.read_train(repo)
    assert len(rows) == receipt["source_ids"] == 3
    assert receipt["premise_groups"] == 2
    assert [row.label for row in rows] == ["entailment", "contradiction", "neutral"]
    assert rows[0].group == rows[1].group
    assert "A rule" not in json.dumps(receipt)


def test_changed_source_and_duplicate_id_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _source_repo(tmp_path, monkeypatch)
    with (repo / "data/train.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="bytes differ"):
        source.read_train(repo)
    first = json.loads((repo / "data/train.jsonl").read_text().splitlines()[0])
    raw = json.dumps(first) + "\n" + json.dumps(first) + "\n"
    (repo / "data/train.jsonl").write_text(raw)
    monkeypatch.setattr(
        source, "TRAIN_SHA256", hashlib.sha256(raw.encode()).hexdigest()
    )
    with pytest.raises(ValueError, match="IDs repeat"):
        source.read_train(repo)
