import math

import pytest
from vllm_srun.families.decision2.answers import (
    apply_score_bias,
    normalized_answer,
    product_answer,
)


def test_choice_answer_with_confidence():
    answer = product_answer("choice", ["a", "b"], [2.0, 0.0], 1.0, ["A", None])
    assert answer["choice"] == "a"
    p = 1 / (1 + math.exp(-2))
    assert answer["probabilities"] == pytest.approx({"a": p, "b": 1 - p}, abs=1e-15)
    entropy = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    assert answer["confidence"] == pytest.approx(1 - entropy / math.log(2))


def test_exact_ties_resolve_to_the_first_option():
    answer = product_answer(
        "choice", ["x", "y", "z"], [1.0, 1.0, 0.0], 1.0, ["", "", ""]
    )
    assert answer["choice"] == "x"
    assert answer["probabilities"]["x"] == answer["probabilities"]["y"]


def test_noul_is_probability_of_true():
    answer = product_answer("noul", ["false", "true"], [0.0, 1.0], 1.0, ["No", "Yes"])
    assert answer == {"type": "noul", "noul": 1 / (1 + math.exp(-1))}


def test_score_expected_level_and_legend():
    answer = product_answer(
        "score", ["0", "1", "2"], [0.0, 0.0, 0.0], 1.0, ["low", {"k": 1}, "high"]
    )
    assert answer["score"] == pytest.approx(1.0)
    assert answer["legend"] == {"0": "low", "1": '{"k":1}', "2": "high"}
    assert answer["confidence"] == pytest.approx(0.0, abs=1e-12)


def test_temperature_scales_logits():
    hot = normalized_answer("choice", ["a", "b"], [2.0, 0.0], 2.0)
    cold = normalized_answer("choice", ["a", "b"], [1.0, 0.0], 1.0)
    assert hot["probabilities"] == cold["probabilities"]


@pytest.mark.parametrize("logits", [[float("nan"), 0.0], [float("inf"), 0.0], [0.0]])
def test_invalid_model_output(logits):
    with pytest.raises(ValueError):
        product_answer("choice", ["a", "b"], logits, 1.0, ["", ""])


def test_score_bias_applies_only_to_its_level_count():
    offsets = {3: [0.1, 0.0, -0.1]}
    assert apply_score_bias(offsets, [1.0, 1.0, 1.0], 3) == [1.1, 1.0, 0.9]
    assert apply_score_bias(offsets, [1.0, 1.0], 2) == [1.0, 1.0]
    assert apply_score_bias(offsets, ["x", 1.0, 1.0], 3) == ["x", 1.0, 1.0]
