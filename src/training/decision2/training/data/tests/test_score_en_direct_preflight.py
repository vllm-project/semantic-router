"""Both direct-LoRA starts must pass the same frozen native-output gate."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from training.data import score_en_direct_preflight as gate


def test_compare_requires_both_zero_step_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    roster = [
        {
            "id": f"item-{index}",
            "questions": {"decision": {"type": "noul"}},
        }
        for index in range(32)
    ]
    roster_manifest = {
        "token_ids_sha256": ["x"] * 32,
        "token_count": [7] * 32,
    }
    monkeypatch.setattr(
        gate,
        "_roster",
        lambda _: (roster, roster_manifest, tmp_path / "prompts.jsonl"),
    )
    monkeypatch.setattr(gate, "prompt_input_sha256", lambda _: "input")
    reference = [
        {
            "id": row["id"],
            "input_sha256": "input",
            "usage": {"input_tokens": 7},
            "adapter_status": "ok",
            "adapter_errors": {},
            "answers": {"decision": {"type": "noul", "noul": 0.6}},
        }
        for row in roster
    ]
    candidate = [dict(row) for row in reference]
    common = {
        "input_sha256": gate.ROSTER_SHA,
        "token_ids_sha256": ["x"] * 32,
        "adapter_sha256": "code",
        "adapter_files_sha256": {},
        "source_files_sha256": {},
        "execution": "native",
        "software": {},
        "hardware": {},
        "initial_adapter_sha256": "adapter",
        "initial_head_sha256": "head",
        "container_image_id": "container",
    }
    args = Namespace(
        roster_dir=tmp_path,
        reference=tmp_path / "reference.jsonl",
        arm_a=tmp_path / "a.jsonl",
        arm_b=tmp_path / "b.jsonl",
        train_a=tmp_path / "train-a.jsonl",
        train_b=tmp_path / "train-b.jsonl",
        output=tmp_path / "receipt.json",
    )
    monkeypatch.setattr(
        gate,
        "file_sha256",
        lambda path: {
            str(args.train_a): gate.ARM_SHA["A"],
            str(args.train_b): gate.ARM_SHA["B"],
            str(tmp_path / "prompts.jsonl"): gate.ROSTER_SHA,
        }.get(str(path), "f" * 64),
    )

    def read(path, *_):
        return (
            reference if path == args.reference else candidate,
            common,
        )

    monkeypatch.setattr(gate, "_read", read)
    report = gate.compare(args)
    assert report["status"] == "PASS"
    assert report["arms"]["A"]["same_argmax"] == 32
    assert report["arms"]["B"]["same_argmax"] == 32

    args.output = tmp_path / "blocked.json"
    candidate[31] = {
        **candidate[31],
        "answers": {"decision": {"type": "noul", "noul": 0.4}},
    }
    report = gate.compare(args)
    assert report["status"] == "BLOCKED_START_PARITY"
    assert report["arms"]["A"]["same_argmax"] == 31
    assert report["arms"]["B"]["same_argmax"] == 31
