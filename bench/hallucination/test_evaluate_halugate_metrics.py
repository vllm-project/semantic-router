import importlib
import sys
from unittest.mock import MagicMock, patch

_mock_modules = {
    name: MagicMock()
    for name in ("torch", "datasets", "transformers")
    if name not in sys.modules
}

with patch.dict(sys.modules, _mock_modules):
    _module = importlib.import_module("bench.hallucination.evaluate_halugate")
    flagged = _module.flagged
    metrics = _module.metrics

THRESHOLD = 0.82
MIN_SPAN = 2

EXPECTED_TP = 3
EXPECTED_FN = 0
EXPECTED_TN = 2
EXPECTED_SKIPPED_CAUGHT = 7
EXPECTED_SKIPPED_FAITHFUL = 5
EXPECTED_SKIPPED_ACCURACY = 2
EXPECTED_ACCURACY = 1.0
EXPECTED_EMPTY_ACCURACY = 0.0
EXPECTED_ALL_SKIPPED = 3


def test_sentinel_skipped_not_counted_as_false_negative():
    caught = [(False, True, [0.95, 0.95, 0.95])] * EXPECTED_TP
    skipped = [(False, False, [0.95, 0.95, 0.95])] * EXPECTED_SKIPPED_CAUGHT

    result = metrics(caught + skipped, threshold=THRESHOLD, min_span=MIN_SPAN)

    assert result["tp"] == EXPECTED_TP
    assert result["fn"] == EXPECTED_FN
    assert result["recall"] == EXPECTED_ACCURACY
    assert result["sentinel_skipped"] == EXPECTED_SKIPPED_CAUGHT


def test_skipped_faithful_rows_do_not_inflate_tn():
    kept = [(True, True, [0.1, 0.1, 0.1])] * EXPECTED_TN
    skipped = [(True, False, [0.1, 0.1, 0.1])] * EXPECTED_SKIPPED_FAITHFUL

    result = metrics(kept + skipped, threshold=THRESHOLD, min_span=MIN_SPAN)

    assert result["tn"] == EXPECTED_TN
    assert result["sentinel_skipped"] == EXPECTED_SKIPPED_FAITHFUL


def test_accuracy_uses_checked_denominator():
    rows = [
        (False, True, [0.95, 0.95, 0.95]),
        (True, True, [0.1, 0.1, 0.1]),
        (False, False, [0.95, 0.95, 0.95]),
        (False, False, [0.95, 0.95, 0.95]),
    ]
    result = metrics(rows, threshold=THRESHOLD, min_span=MIN_SPAN)

    assert result["accuracy"] == EXPECTED_ACCURACY
    assert result["sentinel_skipped"] == EXPECTED_SKIPPED_ACCURACY


def test_all_sentinel_skipped_rows_have_zero_detector_metrics():
    rows = [
        (False, False, [0.95, 0.95, 0.95]),
        (True, False, [0.1, 0.1, 0.1]),
        (False, False, [0.95, 0.95, 0.95]),
    ]

    result = metrics(rows, threshold=THRESHOLD, min_span=MIN_SPAN)

    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["tn"] == 0
    assert result["sentinel_skipped"] == EXPECTED_ALL_SKIPPED

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["accuracy"] == 0.0


def test_flagged_short_circuits_on_sentinel():
    assert flagged([0.95, 0.95, 0.95], THRESHOLD, MIN_SPAN) is True
    assert flagged([0.1, 0.1, 0.1], THRESHOLD, MIN_SPAN) is False


def test_empty_rows_does_not_divide_by_zero():
    result = metrics([], threshold=THRESHOLD, min_span=MIN_SPAN)

    assert result["accuracy"] == EXPECTED_EMPTY_ACCURACY
    assert result["sentinel_skipped"] == EXPECTED_FN
