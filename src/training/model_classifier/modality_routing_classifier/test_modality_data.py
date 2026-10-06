"""Tests for the pure training-data helpers in modality_data.py."""

import json
import random

import pytest
from modality_data import (
    compute_class_stats,
    load_jsonl,
    oversample_minority_classes,
)


def rows_with_counts(counts: dict[int, int]) -> list[dict]:
    return [
        {"text": f"row {label}-{i}", "label": label}
        for label, n in counts.items()
        for i in range(n)
    ]


def test_load_jsonl_reads_utf8_and_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"text": "¿Qué es esto? 猫", "label": 0}, ensure_ascii=False)
        + "\n\n"
        + json.dumps({"text": "b", "label": 1})
        + "\n",
        encoding="utf-8",
    )
    assert [r["text"] for r in load_jsonl(str(path))] == ["¿Qué es esto? 猫", "b"]


def test_class_stats_of_the_real_split_sizes():
    stats = compute_class_stats([0] * 1400 + [1] * 1400 + [2] * 738, num_classes=3)
    assert stats.label_counts == {0: 1400, 1: 1400, 2: 738}
    assert stats.imbalance_ratio == pytest.approx(1400 / 738)
    assert stats.focal_gamma == 2.0  # ratio 1.9: above 1.5, not above 3.0
    assert stats.class_weights[2] > stats.class_weights[0]


@pytest.mark.parametrize(
    ("counts", "gamma"),
    [
        ({0: 100, 1: 100, 2: 100}, 1.5),  # balanced
        ({0: 100, 1: 100, 2: 60}, 2.0),  # ratio 1.67, mild
        ({0: 100, 1: 100, 2: 20}, 3.0),  # ratio 5, severe
    ],
)
def test_focal_gamma_grows_with_imbalance(counts, gamma):
    labels = [label for label, n in counts.items() for _ in range(n)]
    assert compute_class_stats(labels, 3).focal_gamma == gamma


def test_class_weights_are_clamped():
    stats = compute_class_stats([0] * 1000 + [1] * 1000 + [2] * 1, num_classes=3)
    assert stats.class_weights[2] == 3.0
    assert min(stats.class_weights) >= 0.5


def test_class_stats_reject_an_empty_training_set():
    with pytest.raises(ValueError, match="empty"):
        compute_class_stats([], num_classes=3)


def test_oversampling_balances_the_classes():
    rows = rows_with_counts({0: 40, 1: 40, 2: 10})
    counts = {0: 40, 1: 40, 2: 10}
    balanced = oversample_minority_classes(rows, counts, random.Random(1))
    assert [sum(r["label"] == c for r in balanced) for c in range(3)] == [40, 40, 40]


def test_oversampling_is_reproducible_from_the_seed():
    rows = rows_with_counts({0: 40, 1: 40, 2: 10})
    counts = {0: 40, 1: 40, 2: 10}
    a = oversample_minority_classes(rows, counts, random.Random(5))
    b = oversample_minority_classes(rows, counts, random.Random(5))
    c = oversample_minority_classes(rows, counts, random.Random(6))
    assert a == b
    assert a != c


def test_oversampling_leaves_a_mild_imbalance_alone():
    rows = rows_with_counts({0: 40, 1: 40, 2: 30})
    assert (
        oversample_minority_classes(rows, {0: 40, 1: 40, 2: 30}, random.Random(1))
        is rows
    )
