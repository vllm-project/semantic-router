"""CPU contracts for the human ordinal source screen."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from training.data import audit_helpsteer2_score as audit


def fixture_rows() -> list[dict]:
    rows = []
    for pair in audit.PILOT_PAIRS:
        for direction in (0, 1):
            prompt = (
                f"Please evaluate the factual answer for case {pair}-{direction}: "
                + "request " * 9
            )
            lo_len, hi_len = (140, 180) if direction else (180, 140)
            for label, n in zip(pair, (lo_len, hi_len)):
                rows.append(
                    {
                        "prompt": prompt,
                        "response": f"Grade {label} case {pair}-{direction}. "
                        + "x" * n,
                        **dict.fromkeys(audit.ATTRIBUTES, label),
                    }
                )
    return rows


def test_source_schema_and_group_stats(tmp_path: Path) -> None:
    rows = fixture_rows()
    archive = tmp_path / "train.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    loaded = audit.read_train(archive)
    assert len(loaded) == 24
    summary = audit.summarize(loaded)
    assert summary["normalized_prompt_groups"] == 12
    assert summary["group_multiplicity"] == {2: 12}
    assert summary["attribute_histograms"]["correctness"] == {
        "0": 4,
        "1": 4,
        "2": 6,
        "3": 4,
        "4": 6,
    }
    assert summary["different_label_pairs"] == 12
    assert summary["longer_response_higher_correctness"] == 6
    loaded[0]["correctness"] = 5
    with gzip.open(archive, "wt", encoding="utf-8") as out:
        for row in loaded:
            out.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="ordinal label outside"):
        audit.read_train(archive)


def test_pilot_packet_has_no_gold_and_is_grouped() -> None:
    pilot = audit.choose_pilot(fixture_rows())
    blind, key = audit.private_pilot_packets(pilot)
    questions = [json.loads(line) for line in blind.splitlines()]
    answers = [json.loads(line) for line in key.splitlines()]
    assert len(pilot) == 12
    assert len(questions) == len(answers) == 24
    assert len({row["group_id"] for row in questions}) == 12
    assert all("correctness" not in row for row in questions)
    assert sorted(row["correctness"] for row in answers).count(2) == 6
    assert all(
        len(row["questions"]["correctness"]["criteria"]) == 5 for row in questions
    )


def test_exact_and_near_protected_roles(tmp_path: Path) -> None:
    rows = fixture_rows()
    pilot = audit.choose_pilot(rows)
    inventory = []
    for role in (
        "typed_dev",
        "css_pilot",
        "typed_final_goldfree",
        "css15_goldfree",
        "jevbench_public231",
    ):
        p = tmp_path / f"{role}.prompts.jsonl"
        state = (
            rows[0]["prompt"] if role == "typed_dev" else "Unrelated protected state."
        )
        p.write_text(json.dumps({"id": "x", "state": state, "questions": {}}) + "\n")
        inventory.append({"role": role, "path": str(p), "sha256": audit.file_sha(p)})
    manifest = tmp_path / "protected.json"
    manifest.write_text(json.dumps(inventory))
    result = audit.reference_audit(rows, pilot, manifest, [])
    assert result["typed_dev"]["full_train_exact_state_matches"] == 1
    assert result["typed_dev"]["full_train_near_prompt_matches_heuristic"] == 1
    assert result["typed_dev"]["full_train_near_prompt_groups_heuristic"] == 1
    assert result["typed_dev"]["full_train_near_prompt_source_rows_heuristic"] == 2
    assert result["typed_dev"]["pilot_near_state_matches"] == 1
    assert result["css_pilot"]["full_train_exact_state_matches"] == 0
    p = tmp_path / "typed_dev.prompts.jsonl"
    p.write_text(json.dumps({"id": "x", "state": "changed", "questions": {}}) + "\n")
    with pytest.raises(ValueError, match="Missing or changed protected role"):
        audit.reference_audit(rows, pilot, manifest, [])


def test_private_write_is_exclusive_and_0600(tmp_path: Path) -> None:
    p = tmp_path / "key.jsonl"
    audit.write_private(p, b"{}\n")
    assert p.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        audit.write_private(p, b"replace")
