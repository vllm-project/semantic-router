"""Confidence verification uses caller-owned inputs and report destinations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tuning.verify_confidence import main


def test_verification_reads_explicit_inputs_and_writes_report(tmp_path: Path):
    data_dir = tmp_path / "inputs"
    data_dir.mkdir()
    small_results = [
        {
            "question_id": "q1",
            "category": "math",
            "avg_logprob": -2.0,
            "predicted": "A",
            "correct_answer": "B",
        }
    ]
    large_results = [{"question_id": "q1", "predicted": "B", "correct_answer": "B"}]
    (data_dir / "small_results.json").write_text(json.dumps(small_results))
    (data_dir / "large_results.json").write_text(json.dumps(large_results))
    output = tmp_path / "reports" / "confidence.json"

    main(["--data-dir", str(data_dir), "--output", str(output)])

    report = json.loads(output.read_text())
    assert report["num_questions"] == 1
    assert report["num_categories"] == 1
    assert report["baselines"] == {"always_7b": 0.0, "always_72b": 100.0}
    assert report["per_category"]["math"]["strategy"] == "ESCALATE"


def test_verification_rejects_unmatched_results(tmp_path: Path):
    for name in ("small_results.json", "large_results.json"):
        (tmp_path / name).write_text("[]")
    output = tmp_path / "confidence.json"

    with pytest.raises(SystemExit) as error:
        main(["--data-dir", str(tmp_path), "--output", str(output)])

    assert error.value.code == 2
    assert not output.exists()
