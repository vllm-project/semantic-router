"""Fail-closed checks for the Score v7p candidate admission path."""

from __future__ import annotations

import copy

import pytest

from training.data import score_v7p_admit as admit
from training.data.score_v7p_build import build


def test_triplets_recheck_rendered_oracle_and_group_identity() -> None:
    rows = build(b"x" * 32, "select", 2)
    admit._new_groups(rows, "select", 2)
    changed = copy.deepcopy(rows)
    changed[0]["audit_metadata"]["case_id"] = "K-WRONG"
    with pytest.raises(ValueError, match="triplet entity"):
        admit._new_groups(changed, "select", 2)
    changed = copy.deepcopy(rows)
    changed[0]["state"] = changed[0]["state"].replace("current veto", "former veto")
    with pytest.raises(ValueError, match="rendered document"):
        admit._new_groups(changed, "select", 2)


def test_score_control_keeps_complete_groups_and_token_budget() -> None:
    parent = []
    lengths = {}
    for index in range(130):
        for row_index in range(2):
            row = {
                "id": f"p{index}_{row_index}",
                "group_id": f"g{index}",
                "task_type": "score",
                "language": "en",
            }
            parent.append(row)
            lengths[row["id"]] = 280 + (index % 10)
    control = admit._score_control(parent, lengths, 240 * 286)
    assert len(control) == 240
    assert len({row["group_id"] for row in control}) == 120
    assert (
        abs(sum(lengths[row["id"]] for row in control) - 240 * 286) <= 240 * 286 * 0.01
    )
    # A long survivor from an incomplete original group cannot be used.
    del lengths["p0_0"]
    control = admit._score_control(parent, lengths, 240 * 286)
    assert "g0" not in {row["group_id"] for row in control}


def test_replay_uses_singleton_groups_with_no_row_reuse() -> None:
    parent = []
    lengths = {}
    for kind in ("choice", "noul"):
        for index in range(1030):
            row = {
                "id": f"{kind}-{index}",
                "group_id": f"{kind}-g{index}",
                "task_type": kind,
                "language": "en",
            }
            parent.append(row)
            lengths[row["id"]] = 100
    replay = admit._replay(parent, lengths)
    assert len(replay) == 2048
    assert len({row["id"] for row in replay}) == 2048
    assert len({row["group_id"] for row in replay}) == 2048
    assert {row["task_type"] for row in replay} == {"choice", "noul"}
