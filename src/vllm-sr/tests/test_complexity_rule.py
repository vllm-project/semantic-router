import pytest
from pydantic import ValidationError

from cli.models import ComplexityRule


def _banks():
    return {"easy": {"candidates": ["simple"]}, "hard": {"candidates": ["complex"]}}


def test_embedding_mode_default_accepts_hard_and_easy():
    rule = ComplexityRule(name="c", **_banks())
    assert rule.method is None
    assert rule.hard is not None and rule.easy is not None


def test_explicit_embedding_mode_accepts_hard_and_easy():
    rule = ComplexityRule(name="c", method="embedding", **_banks())
    assert rule.method == "embedding"


def test_embedding_mode_missing_banks_rejected():
    with pytest.raises(ValidationError, match="requires both 'hard' and 'easy'"):
        ComplexityRule(name="c")


def test_model_mode_without_banks_accepted():
    rule = ComplexityRule(name="c", method="model", threshold=0.6)
    assert rule.method == "model"
    assert rule.hard is None and rule.easy is None


def test_unknown_method_rejected():
    with pytest.raises(ValidationError, match="method must be"):
        ComplexityRule(name="c", method="modle", **_banks())
