"""Vela 2.0 calibration: temperatures, thresholds, the PII length rule and the sparse gate."""

from __future__ import annotations

import math

import numpy as np
import pytest
from vllm_srun.families.vela2.calibration import Calibration
from vllm_srun.testing.vela2 import PII_TYPES, calibration


@pytest.fixture
def cal() -> Calibration:
    return Calibration(calibration(decoder=True))


def test_temperatures_and_set_thresholds(cal: Calibration) -> None:
    assert cal.temperature("choice") == 1.3
    assert cal.temperature("missing") == 1.0
    assert cal.span_temperature("broad") == 1.0
    assert cal.set_threshold("anything") == 0.3


def test_pii_length_rule_is_log_linear_between_anchors(cal: Calibration) -> None:
    assert cal.pii_length_threshold(5) == pytest.approx(0.75)
    assert cal.pii_length_threshold(126) == pytest.approx(0.05)
    assert cal.pii_length_threshold(10**6) == pytest.approx(0.005919)
    fraction = (math.log(51) - math.log(21)) / (math.log(126) - math.log(21))
    expected = math.exp(math.log(0.75) + fraction * (math.log(0.05) - math.log(0.75)))
    assert cal.pii_length_threshold(51) == pytest.approx(expected)


def test_span_thresholds_by_question_and_labels(cal: Calibration) -> None:
    assert cal.span_threshold("pii", ["x"], 126) == (pytest.approx(0.05), "pii:length")
    assert cal.span_threshold("names", ["PERSON"], 126)[1] == "pii:length"
    assert cal.span_threshold("halu", ["unsupported"], 10) == (0.55, "halu")
    assert cal.span_threshold("other", ["x"], 10) == (0.5, "span:*")
    assert cal.span_threshold("other", ["x"], 10, head="broad") == (0.55, "broad")
    fallback = Calibration(
        {
            **calibration(False),
            "pii_buckets": [{"name": "short", "max_tokens": 9, "threshold": 0.7}],
        }
    )
    del fallback.raw["pii_length_rule"]
    assert fallback.span_threshold("pii", [], 5) == (0.7, "pii:short")


def test_sparse_gate_raises_the_threshold_for_sparse_documents(
    cal: Calibration,
) -> None:
    text = "Tom met Ann"
    offsets = np.array([[0, 3], [4, 7], [8, 11]], np.int32)
    labels = [PII_TYPES[0], "PERSON"]
    sparse = np.array([[0.0, 0.9], [0.0, 0.0], [0.0, 0.08]])
    assert cal.sparse_gate(sparse, offsets, labels, text, 0.01, "pii:length") == (
        0.1,
        "pii:length+sparse",
    )
    assert cal.sparse_gate(sparse, offsets, labels, text, 0.2, "pii:length") == (
        0.2,
        "pii:length",
    )


def test_noul_calibration_is_off_unless_enabled(cal: Calibration) -> None:
    assert cal.noul(0.7, enabled=False) == 0.7
    expected = 1 / (1 + math.exp(-(math.log(0.7 / 0.3) / 2.0 + 0.25)))
    assert cal.noul(0.7, enabled=True) == pytest.approx(expected)
    assert cal.noul_default() is False
