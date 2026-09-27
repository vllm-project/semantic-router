"""Small deterministic checks for the prospective Score v8 quality pilot."""

from __future__ import annotations

import json

import pytest

from training.data import score_v8_pilot as pilot
from training.data.score_v8_pilot_audit import _candidate
from training.model.data import check_partition_isolation, load_partition


def test_five_mechanisms_and_balanced_triplets() -> None:
    train = pilot.build(b"x" * 32, "train")
    select = pilot.build(b"x" * 32, "select")
    assert len(train) == 45 and len(select) == 30
    assert len({row["group_id"] for row in train}) == 15
    assert len({row["group_id"] for row in select}) == 10
    assert len({row["family"] for row in train}) == 5
    assert len({row["family"] for row in select}) == 5
    assert _candidate(train, "train")["level_counts"] == {0: 15, 1: 15, 2: 15}
    assert _candidate(select, "select")["level_counts"] == {0: 10, 1: 10, 2: 10}
    assert (
        min(len(row["state"]) for row in train if row["family"] == "score_v8_long_memo")
        >= 1500
    )
    check_partition_isolation({"train": train, "select": select})


def test_rendered_oracle_uses_actual_record_and_evidence() -> None:
    rows = pilot.build(b"y" * 32, "train")
    for row in rows:
        meta = row["audit_metadata"]
        assert (
            pilot.rendered_oracle(
                row["state"],
                meta["mechanism"],
                meta["record"],
                meta["review_day"],
                site=meta["site"],
                activity=meta["activity"],
            )
            == row["label"]
        )
    numeric = next(
        row
        for row in rows
        if row["family"] == "score_v8_numeric_limits" and row["label"] == 2
    )
    altered = numeric["state"].replace(" | current", " | missing", 1)
    meta = numeric["audit_metadata"]
    assert (
        pilot.rendered_oracle(
            altered, "numeric_limits", meta["record"], meta["review_day"]
        )
        == 1
    )


def test_blind_packet_hides_answer_and_roundtrips(tmp_path) -> None:
    seed = tmp_path / "seed"
    seed.write_bytes(b"z" * 32)
    seed.chmod(0o600)
    output = tmp_path / "pilot"
    manifest = pilot.write(seed, output)
    assert manifest["status"] == "PENDING_INDEPENDENT_BLIND_REVIEW"
    for role, expected in (("train", 45), ("select", 30)):
        rows = load_partition(output / f"{role}.jsonl", role)
        packet = [
            json.loads(line)
            for line in (output / f"{role}-blind-packet.jsonl").read_text().splitlines()
        ]
        key = json.loads((output / f"{role}-sealed-key.json").read_text())
        assert len(rows) == expected and len(key) == expected
        assert len(packet) == expected // 3
        assert all(
            "label" not in item and "source_id" not in item
            for group in packet
            for item in group["items"]
        )
        assert all(len(group["items"]) == 3 for group in packet)
    with pytest.raises(ValueError):
        pilot.write(seed, output)
