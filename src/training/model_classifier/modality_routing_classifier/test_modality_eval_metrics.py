"""Tests for the metrics and report assembly in modality_eval_metrics.py."""

import os
from datetime import datetime, timezone

import modality_eval_metrics
import numpy as np
import pytest
from modality_eval_metrics import (
    build_per_example_records,
    build_report,
    compute_classification_metrics,
    compute_routing_agreement,
    evaluate_subset,
    flag_test_contamination,
    normalize_text,
    portable_path,
    sha256_of,
)


def test_normalize_text_ignores_case_and_whitespace():
    assert normalize_text("  Draw   A\tCat ") == "draw a cat"


def test_contamination_flags_exact_and_normalized_matches():
    test_rows = [
        {"text": "draw a cat", "label_name": "DIFFUSION"},
        {"text": "Explain  TCP", "label_name": "AR"},
        {"text": "something new", "label_name": "AR"},
    ]
    indices, rate = flag_test_contamination(["draw a cat"], ["explain tcp"], test_rows)
    assert indices == [0, 1]
    assert rate == {"DIFFUSION": 1.0, "AR": 0.5}


def test_classification_metrics_on_a_known_case():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert metrics["per_class"]["AR"]["recall"] == 0.5
    assert metrics["per_class"]["DIFFUSION"]["precision"] == 0.5


def test_routing_agreement_splits_the_disagreements_by_who_was_right():
    y_true = np.array([0, 0, 1, 2, 2])
    a = np.array([0, 1, 1, 2, 0])
    b = np.array([0, 0, 1, 1, 1])
    result = compute_routing_agreement(y_true, a, b)
    assert result["num_agree"] == 2
    assert result["num_disagree"] == 3
    breakdown = result["disagreement_breakdown"]
    assert breakdown["a_matched_truth_b_wrong"]["count"] == 1
    assert breakdown["b_matched_truth_a_wrong"]["count"] == 1
    assert breakdown["both_wrong_different_labels"]["count"] == 1


def test_agreement_of_identical_models_has_no_disagreements():
    y = np.array([0, 1, 2])
    result = compute_routing_agreement(y, y, y)
    assert result["agreement_rate"] == 1.0
    assert (
        result["disagreement_breakdown"]["a_matched_truth_b_wrong"][
            "rate_of_disagreements"
        ]
        is None
    )


def test_portable_path_hides_the_machine():
    here = os.path.dirname(os.path.abspath(modality_eval_metrics.__file__))
    assert portable_path(os.path.join(here, "exported", "test.jsonl")) == os.path.join(
        "exported", "test.jsonl"
    )
    assert portable_path("/home/someone/runs/model_a/") == "model_a"
    assert portable_path("org/model-name") == "org/model-name"
    assert portable_path("models/local") == "models/local"


ROWS = [
    {"text": "draw a cat", "label": 1, "label_name": "DIFFUSION"},
    {"text": "explain tcp", "label": 0, "label_name": "AR"},
    {"text": "explain and show tcp", "label": 2, "label_name": "BOTH"},
    {"text": "another row", "label": 0, "label_name": "AR"},
]
PREDS = {
    "published_baseline": np.array([1, 0, 2, 0]),
    "clean_baseline": np.array([1, 0, 2, 1]),
    "candidate": np.array([1, 2, 2, 0]),
}


def test_per_example_records_carry_a_hash_and_optionally_the_text():
    plain = build_per_example_records(ROWS, PREDS, {1}, include_full_text=False)
    assert plain[0]["input_hash_sha256"] == sha256_of("draw a cat")
    assert plain[1]["contaminated"] is True
    assert plain[3]["clean_baseline_pred"] == "DIFFUSION"
    assert "text" not in plain[0]
    with_text = build_per_example_records(ROWS, PREDS, set(), include_full_text=True)
    assert with_text[0]["text"] == "draw a cat"


def test_build_report_is_deterministic_and_filters_contamination():
    contamination = ([1], {"AR": 0.5})
    kwargs = {
        "paths": {"test_file": "/abs/exported/test.jsonl"},
        "model_paths": {"candidate": "/abs/runs/cand", "clean_baseline": "org/x"},
        "generated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
    }
    report = build_report(ROWS, PREDS, contamination, **kwargs)
    assert report == build_report(ROWS, PREDS, contamination, **kwargs)
    assert report["metadata"]["num_test_examples"] == 4
    assert report["metadata"]["generated_at_utc"] == "2026-01-02T00:00:00+00:00"
    assert report["metadata"]["model_paths"] == {
        "candidate": "cand",
        "clean_baseline": "org/x",
    }
    assert report["contamination_check"]["num_contaminated"] == 1
    assert report["contamination_check"]["contamination_rate"] == 0.25
    assert report["full_test_set"]["metrics"]["candidate"]["accuracy"] == 0.75
    assert report["contamination_filtered_test_set"] is not None
    filtered = report["contamination_filtered_test_set"]["metrics"]["candidate"]
    assert filtered["accuracy"] == 1.0
    assert set(report["full_test_set"]["routing_agreement"]) == {
        "candidate_vs_clean_baseline",
        "candidate_vs_published_baseline",
        "clean_baseline_vs_published_baseline",
    }


def test_report_skips_the_filtered_set_when_every_row_is_contaminated():
    report = build_report(
        ROWS,
        PREDS,
        ([0, 1, 2, 3], {}),
        paths={},
        model_paths={},
        generated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    assert report["contamination_filtered_test_set"] is None


def test_evaluate_subset_scores_every_model():
    y_true = np.array([r["label"] for r in ROWS])
    result = evaluate_subset(y_true, PREDS)
    assert set(result["metrics"]) == set(PREDS)
    assert result["metrics"]["published_baseline"]["accuracy"] == pytest.approx(1.0)
