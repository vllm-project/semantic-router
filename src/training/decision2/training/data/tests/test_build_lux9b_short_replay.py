"""Leakage checks for the short-replay 9B data arm."""

import pytest

from training.data.build_lux9b_short_replay import quarantine, selected_rows


def _row(identifier: str, group: str, state: str) -> dict:
    return {
        "id": identifier,
        "group_id": group,
        "input_sha256": identifier.zfill(64),
        "state": state,
    }


def test_group_incompleteness_rejected() -> None:
    pool = [_row("a", "paired", "one"), _row("b", "paired", "two")]
    with pytest.raises(ValueError, match="incomplete"):
        selected_rows(pool, {"a"})


def test_collision_removes_entire_pair() -> None:
    pool = [
        _row("a", "paired", "A unique situation"),
        _row("b", "paired", "A second unique situation"),
        _row("c", "safe", "An unrelated third situation"),
    ]
    protected = [_row("heldout", "other", "A second unique situation")]
    kept, audit = quarantine(pool, protected)
    assert [row["id"] for row in kept] == ["c"]
    assert audit["excluded_group_count"] == 1
