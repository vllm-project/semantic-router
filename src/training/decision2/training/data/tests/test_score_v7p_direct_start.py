"""Receipt binding for the separately versioned Score v7p LoRA start."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

from training.data import score_v7p_direct_start as direct
from training.model.data import file_sha256


def _admission(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "decision2-score-v7p-matched-arms/1",
                "status": "CANDIDATE_PENDING_BLIND_QA_AND_ZERO_STEP",
                "arm_sha256": {"A": "a" * 64, "B": "b" * 64, "C": "a" * 64},
            }
        ),
        encoding="utf-8",
    )


def test_admission_rejects_missing_or_changed_objective_arm(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _admission(path)
    assert direct._admission(path)["A"] == direct._admission(path)["C"]
    value = json.loads(path.read_text(encoding="utf-8"))
    del value["arm_sha256"]["B"]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="matched-arm admission"):
        direct._admission(path)
    value["arm_sha256"]["B"] = "b" * 64
    value["arm_sha256"]["C"] = "c" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="matched-arm admission"):
        direct._admission(path)


def test_prediction_reader_rejects_invalid_or_reordered_items(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    roster = [{"id": f"q{i}"} for i in range(32)]
    path.write_text(
        "".join(json.dumps({"id": row["id"]}) + "\n" for row in roster),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "decision2-score-v7p-zero-step-prediction/1",
        "predictions_sha256": file_sha256(path),
        "mode": "reference",
        "arm": None,
        "model_sha256": direct.SOURCE_SHA,
        "roster_sha256": direct.ROSTER_SHA,
        "counts": {
            "valid_questions": 32,
            "invalid_questions": 0,
            "over_budget_questions": 0,
            "truncated_questions": 0,
        },
        "max_length": direct.MAX_LENGTH,
        "temperature": 1.0,
    }
    direct._manifest_path(path).write_text(json.dumps(manifest), encoding="utf-8")
    direct._read(path, "reference", None, roster)
    with pytest.raises(ValueError, match="prediction rows differ"):
        direct._read(path, "reference", None, list(reversed(roster)))
    manifest["counts"]["truncated_questions"] = 1
    direct._manifest_path(path).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="prediction manifest differs"):
        direct._read(path, "reference", None, roster)


def test_compare_needs_all_three_native_starts_and_reports_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    roster = [
        {"id": f"q{i}", "questions": {"decision": {"type": "choice"}}}
        for i in range(32)
    ]
    roster_manifest = {"token_ids_sha256": ["t"] * 32, "token_count": [10] * 32}
    monkeypatch.setattr(
        direct, "_roster", lambda path: (roster, roster_manifest, tmp_path / "roster")
    )
    monkeypatch.setattr(
        direct, "_admission", lambda path: {"A": "a" * 64, "B": "b" * 64, "C": "a" * 64}
    )
    monkeypatch.setattr(direct, "prompt_input_sha256", lambda prompt: "p")

    def predictions() -> list[dict]:
        return [
            {
                "id": f"q{i}",
                "input_sha256": "p",
                "usage": {"input_tokens": 10},
                "adapter_status": "ok",
                "adapter_errors": [],
                "answers": {
                    "decision": {
                        "type": "choice",
                        "choice": "A",
                        "probabilities": {"A": 0.6, "B": 0.4},
                    }
                },
            }
            for i in range(32)
        ]

    base_manifest = {
        "input_sha256": direct.ROSTER_SHA,
        "token_ids_sha256": ["t"] * 32,
        "adapter_sha256": "c" * 64,
        "adapter_files_sha256": {"code": "d" * 64},
        "source_files_sha256": {"base": "e" * 64},
        "initial_adapter_sha256": "f" * 64,
        "initial_head_sha256": "0" * 64,
        "execution": "native",
        "software": {"torch": "test"},
        "hardware": {"accelerator_name": "test"},
        "container_image_id": direct.IMAGE_ID,
    }
    values = {
        "reference": predictions(),
        "A": predictions(),
        "B": predictions(),
        "C": predictions(),
    }

    def fake_read(path, mode, arm, roster):
        manifest = copy.deepcopy(base_manifest)
        manifest["train_sha256"] = (
            "b" * 64 if arm == "B" else "a" * 64 if arm in {"A", "C"} else None
        )
        return values[arm or "reference"], manifest

    monkeypatch.setattr(direct, "_read", fake_read)

    def fake_hash(path):
        name = Path(path).name
        if name == "train-A" or name == "train-C":
            return "a" * 64
        if name == "train-B":
            return "b" * 64
        return "c" * 64

    monkeypatch.setattr(direct, "file_sha256", fake_hash)
    args = argparse.Namespace(
        output=tmp_path / "pass.json",
        roster_dir=tmp_path,
        admission_manifest=tmp_path / "admission.json",
        reference=tmp_path / "reference",
        arm_a=tmp_path / "A",
        arm_b=tmp_path / "B",
        arm_c=tmp_path / "C",
        train_a=tmp_path / "train-A",
        train_b=tmp_path / "train-B",
        train_c=tmp_path / "train-C",
    )
    assert direct.compare(args)["status"] == "PASS"
    values["B"][0]["answers"]["decision"] = {
        "type": "choice",
        "choice": "B",
        "probabilities": {"A": 0.4, "B": 0.6},
    }
    args.output = tmp_path / "blocked.json"
    result = direct.compare(args)
    assert result["status"] == "BLOCKED_START_PARITY"
    assert result["arms"]["B"]["same_argmax"] == 31
