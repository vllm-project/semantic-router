"""Sparse screens keep the full benchmark's per-answer validity rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from research.compare_generative_screen import compare


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_sparse_compare_counts_invalid_as_wrong_and_checks_hash(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    head = tmp_path / "head.jsonl"
    source = tmp_path / "source.jsonl"
    _write(
        gold,
        [
            {
                "id": "x",
                "group_id": "g",
                "questions": {"q": {"type": "score", "criteria": ["low", "high"]}},
                "gold": {"q": {"type": "score", "value": 1}},
                "provenance": {"payload_sha256": "digest"},
            }
        ],
    )
    _write(
        head,
        [{"id": "x", "source_input_sha256": "digest", "answers": {"q": {"score": 0}}}],
    )
    _write(
        source,
        [{"id": "x", "source_input_sha256": "digest", "answers": {"q": {"score": 1}}}],
    )
    report = compare(gold, head, source)
    assert report["by_type"]["score"]["gain"] == 1
    assert report["by_type"]["score"]["source_only"] == 1
    assert report["independent_groups"] == 1
    assert not report["gate_for_full_dev"]

    _write(
        source,
        [
            {
                "id": "x",
                "source_input_sha256": "digest",
                "answers": {"q": {"error": "invalid"}},
            }
        ],
    )
    report = compare(gold, head, source)
    assert report["by_type"]["score"]["source_invalid"] == 1
    assert report["by_type"]["score"]["source_correct"] == 0

    _write(source, [{"id": "x", "source_input_sha256": "wrong", "answers": {}}])
    with pytest.raises(ValueError, match="input hash differs"):
        compare(gold, head, source)
