"""Prospective CPU checks for the repaired Score v8.2 rendered pilot."""

from __future__ import annotations

import json

import pytest

from training.data import score_v82_pilot as pilot
from training.data import score_v82_pilot_audit as audit
from training.model.data import check_partition_isolation


def test_balanced_new_groups_and_two_source_necessity() -> None:
    train = pilot.build(b"a" * 32, "train")
    select = pilot.build(b"a" * 32, "select")
    assert audit._candidate(train, "train")["level_counts"] == {0: 15, 1: 15, 2: 15}
    assert audit._candidate(select, "select")["level_counts"] == {0: 10, 1: 10, 2: 10}
    assert len(audit._source_necessity([*train, *select])) == 5
    document = audit._document_realism([*train, *select])
    assert document["normalized_pair_similarity_max"] < 0.85
    assert min(item["min_chars"] for item in document["groups"]) >= 1500
    check_partition_isolation({"train": train, "select": select})
    assert all(row["source"] == pilot.VERSION for row in [*train, *select])


def test_one_source_shortcut_is_rejected() -> None:
    rows = pilot.build(b"b" * 32, "train") + pilot.build(b"b" * 32, "select")
    group = next(
        row["group_id"]
        for row in rows
        if row["family"] == "score_v8_evidence_sufficiency"
    )
    for row in rows:
        if row["group_id"] != group:
            continue
        record = row["audit_metadata"]["record"]
        lines = row["state"].splitlines()
        row["state"] = "\n".join(
            (
                line.replace("normal observed", "outage observed").replace(
                    "unavailable observed", "outage observed"
                )
                if line.startswith(f"Dispatch diary for {record},")
                else line
            )
            for line in lines
        )
    with pytest.raises(ValueError, match="single source"):
        audit._source_necessity(rows)


def test_new_blind_packets_hide_keys(tmp_path) -> None:
    seed = tmp_path / "seed"
    seed.write_bytes(b"c" * 32)
    seed.chmod(0o600)
    out = tmp_path / "candidate"
    manifest = pilot.write(seed, out)
    assert manifest["schema_version"] == pilot.VERSION
    for role, expected in (("train", 45), ("select", 30)):
        packet = [
            json.loads(line)
            for line in (out / f"{role}-blind-packet.jsonl").read_text().splitlines()
        ]
        key = json.loads((out / f"{role}-sealed-key.json").read_text())
        assert len(key) == expected and len(packet) == expected // 3
        assert all(
            set(item) == {"review_id", "state", "instructions", "options"}
            for group in packet
            for item in group["items"]
        )
        assert (out / f"{role}-blind-packet.jsonl").stat().st_mode & 0o077 == 0
    with pytest.raises(ValueError):
        pilot.write(seed, out)
