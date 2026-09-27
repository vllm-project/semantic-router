"""Contracts for the prospective 4B Score corpus, not model-performance tests."""

from __future__ import annotations

import copy
from collections import Counter

import pytest

from training.data import score_4b_fresh_ordinal as corpus


def test_train_and_select3_are_complete_independent_triplets() -> None:
    train, select3 = corpus.generate(bytes(range(32)))
    assert (len(train), len(select3)) == (288, 96)
    assert not {row["group_id"] for row in train} & {row["group_id"] for row in select3}
    assert not {row["family"] for row in train} & {row["family"] for row in select3}
    for rows, n_groups in ((train, 96), (select3, 32)):
        assert Counter(row["label"] for row in rows) == {
            0: n_groups,
            1: n_groups,
            2: n_groups,
        }
        assert Counter(row["language"] for row in rows) == {
            "en": n_groups * 3 * 3 // 4,
            "zh": n_groups * 3 // 4,
        }
        by_group: dict[str, set[int]] = {}
        for row in rows:
            by_group.setdefault(row["group_id"], set()).add(row["label"])
            assert (
                corpus._rendered_oracle(
                    row["family"][6:], row["state"], row["language"]
                )
                == row["label"]
            )
        assert all(levels == {0, 1, 2} for levels in by_group.values())


def test_displayed_oracle_rejects_missing_decisive_document() -> None:
    train, _ = corpus.generate(bytes(range(32)))
    row = copy.deepcopy(train[0])
    row["state"]["documents"] = [
        document
        for document in row["state"]["documents"]
        if document["kind"] != "current"
    ]
    with pytest.raises(ValueError, match="Document identity"):
        corpus._rendered_oracle(row["family"][6:], row["state"], row["language"])
