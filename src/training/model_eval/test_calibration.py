"""Tests for the calibration, abstention, and slice helpers."""

import pathlib
import sys

import numpy as np
import pytest

TEST_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))

from calibration import (  # noqa: E402
    abstention_curve,
    expected_calibration_error,
    length_slices,
    multiclass_brier_score,
    reliability_bins,
)


def test_perfectly_calibrated_predictions_have_zero_error():
    # Half the rows are answered at confidence 1.0 and correct, half at 0.5
    # and correct exactly half the time.
    confidences = np.array([1.0] * 100 + [0.5] * 100)
    correct = np.array([1.0] * 100 + [1.0] * 50 + [0.0] * 50)
    ece, mce = expected_calibration_error(reliability_bins(confidences, correct))
    assert ece == pytest.approx(0.0, abs=1e-12)
    assert mce == pytest.approx(0.0, abs=1e-12)


def test_overconfidence_is_reported_as_a_positive_gap():
    confidences = np.full(100, 0.95)
    correct = np.array([1.0] * 50 + [0.0] * 50)
    ece, mce = expected_calibration_error(reliability_bins(confidences, correct))
    assert ece == pytest.approx(0.45)
    assert mce == pytest.approx(0.45)


def test_confidence_of_one_lands_in_the_final_bin():
    rows = 2
    bins = reliability_bins(np.array([1.0, 1.0]), np.array([1.0, 0.0]), bin_count=10)
    assert bins[-1]["count"] == rows
    assert sum(entry["count"] for entry in bins) == rows


def test_empty_bins_report_no_confidence_instead_of_zero():
    bins = reliability_bins(np.array([0.95]), np.array([1.0]), bin_count=10)
    assert bins[0]["count"] == 0
    assert bins[0]["confidence"] is None
    assert bins[0]["accuracy"] is None


def test_calibration_error_ignores_empty_bins():
    ece, _ = expected_calibration_error(
        [
            {"count": 0, "confidence": None, "accuracy": None},
            {"count": 10, "confidence": 0.9, "accuracy": 0.9},
        ]
    )
    assert ece == pytest.approx(0.0)


def test_brier_score_is_zero_for_a_confident_correct_prediction():
    probabilities = np.array([[0.0, 1.0], [1.0, 0.0]])
    labels = np.array([1, 0])
    assert multiclass_brier_score(probabilities, labels) == pytest.approx(0.0)


def test_brier_score_is_one_for_a_confident_wrong_prediction():
    probabilities = np.array([[0.0, 1.0]])
    labels = np.array([0])
    assert multiclass_brier_score(probabilities, labels) == pytest.approx(1.0)


def test_brier_score_rejects_labels_outside_the_probability_columns():
    with pytest.raises(ValueError, match="outside the probability columns"):
        multiclass_brier_score(np.array([[0.5, 0.5]]), np.array([2]))


def test_abstention_curve_reports_full_coverage_at_zero():
    confidences = np.array([0.4, 0.8, 0.95])
    correct = np.array([0.0, 1.0, 1.0])
    curve = {
        point["threshold"]: point
        for point in abstention_curve(confidences, correct, thresholds=(0.0, 0.5, 0.9))
    }
    assert curve[0.0]["coverage"] == pytest.approx(1.0)
    assert curve[0.0]["selective_accuracy"] == pytest.approx(2 / 3)
    assert curve[0.5]["abstained"] == 1
    assert curve[0.5]["selective_accuracy"] == pytest.approx(1.0)
    assert curve[0.9]["coverage"] == pytest.approx(1 / 3)


def test_abstention_curve_reports_no_accuracy_when_everything_abstains():
    point = abstention_curve(np.array([0.4]), np.array([1.0]), thresholds=(0.9,))[0]
    assert point["coverage"] == pytest.approx(0.0)
    assert point["selective_accuracy"] is None


def test_abstention_curve_rejects_an_out_of_range_threshold():
    with pytest.raises(ValueError, match="outside"):
        abstention_curve(np.array([0.5]), np.array([1.0]), thresholds=(1.5,))


def test_length_slices_partition_every_row_exactly_once():
    texts = ["a" * n for n in (10, 100, 500, 5000)]
    masks = length_slices(texts, boundaries=(64, 256, 1024))
    stacked = np.vstack(list(masks.values()))
    assert stacked.sum(axis=0).tolist() == [1, 1, 1, 1]
    assert list(masks) == ["chars<64", "chars<256", "chars<1024", "chars>=1024"]


def test_confidences_outside_the_unit_interval_are_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        reliability_bins(np.array([1.2]), np.array([1.0]))
