"""Prospective v7p candidate generation, without protected benchmark labels."""

from __future__ import annotations

import json

from training.data import score_v7p_build as build
from training.model.data import validate_row


def test_triplets_have_fixed_case_and_two_independent_oracles() -> None:
    rows = build.build(b"x" * 32, "train", 120)
    assert len(rows) == 360
    for index in range(120):
        triplet = rows[3 * index : 3 * index + 3]
        assert {row["label"] for row in triplet} == {0, 1, 2}
        assert len({row["group_id"] for row in triplet}) == 1
        assert len({row["audit_metadata"]["case_id"] for row in triplet}) == 1
        assert len({row["audit_metadata"]["review_day"] for row in triplet}) == 1
        for row in triplet:
            assert validate_row(row, "train") == row
            assert (
                build._oracle_two(
                    row["state"],
                    row["audit_metadata"]["case_id"],
                    row["audit_metadata"]["review_day"],
                )
                == row["label"]
            )


def test_select_has_separate_domains_and_no_shared_input() -> None:
    train = build.build(b"x" * 32, "train", 120)
    select = build.build(b"y" * 32, "select", 80)
    assert len(select) == 240
    assert {r["audit_metadata"]["domain"] for r in train}.isdisjoint(
        {r["audit_metadata"]["domain"] for r in select}
    )
    assert {r["input_sha256"] for r in train}.isdisjoint(
        {r["input_sha256"] for r in select}
    )
    for row in select:
        assert validate_row(row, "select") == row


def test_rendered_oracle_rejects_corrupted_document() -> None:
    row = build.build(b"x" * 32, "select", 1)[0]
    lines = row["state"].splitlines()
    bundle = json.loads(lines[1].split(": ", 1)[1])
    bundle["notices"] = []
    lines[1] = "Document A, signed requirement ledger and notices: " + json.dumps(
        bundle
    )
    assert (
        build._oracle_two(
            "\n".join(lines),
            row["audit_metadata"]["case_id"],
            row["audit_metadata"]["review_day"],
        )
        != row["label"]
    )
