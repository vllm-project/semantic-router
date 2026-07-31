"""Tests for router signal and decision extension schema."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.models import (  # noqa: E402
    ClassifierSignal,
    Condition,
    MetadataPredicate,
    MetadataRule,
)


def test_metadata_rule_exact_match():
    rule = MetadataRule(
        name="consent-denied",
        key="consent",
        predicate=MetadataPredicate(equals="denied"),
    )
    assert rule.predicate.equals == "denied"


def test_metadata_predicate_rejects_multiple_comparators():
    with pytest.raises(ValidationError):
        MetadataPredicate(equals="denied", exists=True)


def test_classifier_signal_llm_shape():
    signal = ClassifierSignal(
        name="risk",
        type="llm",
        model="risk-judge",
        labels=["SAFE", "RISKY"],
        instructions="Choose a label.",
    )
    assert signal.labels == ["SAFE", "RISKY"]


def test_classifier_signal_local_shape():
    signal = ClassifierSignal(
        name="risk",
        type="local",
        model_path="models/risk",
        labels=["SAFE", "RISKY"],
    )
    assert signal.model_path == "models/risk"


def test_classifier_signal_rejects_backend_mixed_fields():
    with pytest.raises(ValidationError):
        ClassifierSignal(
            name="risk",
            type="local",
            model="risk-judge",
            model_path="models/risk",
            labels=["SAFE", "RISKY"],
        )


def test_classifier_condition_score_shape():
    condition = Condition(
        type="classifier",
        name="risk",
        label="RISKY",
        predicate={"gte": 0.5},
        on_error="no_match",
    )
    assert condition.predicate.gte == 0.5


def test_classifier_condition_requires_label_and_predicate():
    with pytest.raises(ValidationError):
        Condition(type="classifier", name="risk")


def test_non_classifier_condition_rejects_label():
    with pytest.raises(ValidationError):
        Condition(type="embedding", name="risk", label="RISKY")
