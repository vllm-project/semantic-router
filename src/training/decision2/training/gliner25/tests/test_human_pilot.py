"""Group-integrity checks for the human-only GLiNER native pilot."""

from __future__ import annotations

import pytest

from training.gliner25.human_pilot import GOEMOTIONS, choose_human_rows


def _row(source: str, group: str, kind: str, suffix: str = "a") -> dict:
    return {
        "source": source,
        "group_id": group,
        "task_type": kind,
        "language": "en",
        "id": f"{source}:{group}:{kind}:{suffix}",
    }


def test_group_draw_keeps_complete_goemotions_pair_and_one_natural_row() -> None:
    source = "legacy:cosmos_qa"
    rows = [
        _row(GOEMOTIONS, "g1", "choice"),
        _row(GOEMOTIONS, "g1", "noul"),
        _row(GOEMOTIONS, "g2", "choice"),
        _row(GOEMOTIONS, "g2", "noul"),
        _row(source, "n1", "choice", "a"),
        _row(source, "n1", "choice", "b"),
        _row(source, "n2", "choice"),
    ]
    selected = choose_human_rows(rows, {GOEMOTIONS: 1, source: 1})
    assert len(selected) == 3
    grouped = {(row["source"], row["group_id"]) for row in selected}
    assert len(grouped) == 2
    assert {row["task_type"] for row in selected if row["source"] == GOEMOTIONS} == {
        "choice",
        "noul",
    }
    assert selected == choose_human_rows(
        list(reversed(rows)), {GOEMOTIONS: 1, source: 1}
    )


def test_incomplete_goemotions_pair_fails_closed() -> None:
    with pytest.raises(ValueError, match="intact Choice/Noul pair"):
        choose_human_rows([_row(GOEMOTIONS, "g1", "choice")], {GOEMOTIONS: 1})


def test_unknown_or_non_english_source_fails_closed() -> None:
    row = _row("legacy:snli", "g1", "choice")
    with pytest.raises(ValueError, match="unexpected source/language"):
        choose_human_rows([row], {GOEMOTIONS: 1})
    row["source"] = GOEMOTIONS
    row["language"] = "zh"
    with pytest.raises(ValueError, match="unexpected source/language"):
        choose_human_rows([row], {GOEMOTIONS: 1})
