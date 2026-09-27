"""Whole-group selection for the frozen Kai recovery micro-arm."""

import pytest

from training.data.build_kai06_recovery1024 import select_whole_groups


def test_select_preserves_components_and_exact_budget():
    rows = [
        {
            "id": f"{group}/{index}",
            "component_id": group,
            "source_id": "tweeteval_train:hate",
            "question": {"type": "Choice"},
        }
        for group, size in (("one", 2), ("two", 2), ("three", 1))
        for index in range(size)
    ]
    chosen = select_whole_groups(rows, 3, source="tweeteval_train:hate", kind="Choice")
    assert len(chosen) == 3
    assert all(
        sum(row["component_id"] == group for row in chosen)
        in (0, sum(row["component_id"] == group for row in rows))
        for group in {row["component_id"] for row in rows}
    )
    with pytest.raises(ValueError, match="Cannot select"):
        select_whole_groups(rows, 6, source="tweeteval_train:hate", kind="Choice")
