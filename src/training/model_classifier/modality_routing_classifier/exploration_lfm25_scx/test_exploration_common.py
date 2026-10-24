"""Tests for the helpers shared by the exploration scripts."""

import hashlib
import json
import re

import exploration_common as common
import pytest


def test_examples_have_one_true_label_and_all_labels():
    rows = [{"text": "draw a cat", "label_name": "DIFFUSION"}]
    assert common.to_gliclass_examples(rows) == [
        {
            "text": "draw a cat",
            "all_labels": ["AR", "DIFFUSION", "BOTH"],
            "true_labels": ["DIFFUSION"],
        }
    ]


def test_label_descriptions_are_prefixed_with_the_label_name():
    descriptions = common.label2description()
    assert descriptions["AR"] == "AR: a text-only response"
    assert set(descriptions) == {"AR", "DIFFUSION", "BOTH"}


def test_an_explicit_device_wins():
    assert common.pick_device("cpu") == "cpu"
    assert common.pick_device(None) in {"cpu", "cuda"}


@pytest.mark.parametrize("revision", [common.SCX_REVISION, common.LFM25_REVISION])
def test_hub_revisions_are_pinned_to_a_full_commit_hash(revision):
    assert re.fullmatch(r"[0-9a-f]{40}", revision)


TRUTH = ["AR", "AR", "DIFFUSION", "BOTH"]
PREDS = ["AR", "BOTH", "DIFFUSION", "BOTH"]


def test_summary_counts_correct_predictions_and_the_confusion_matrix():
    metrics = common.summarize_predictions(TRUTH, PREDS)
    assert metrics["accuracy"] == 0.75
    assert metrics["num_correct"] == 3 and metrics["num_rows"] == 4
    assert metrics["confusion_matrix"] == [[1, 0, 1], [0, 1, 0], [0, 0, 1]]
    assert metrics["per_class"]["BOTH"]["precision"] == 0.5


def test_summary_rejects_an_unknown_label():
    with pytest.raises(KeyError):
        common.summarize_predictions(["AR"], ["MAYBE"])


def test_summary_lines_name_the_model_and_show_accuracy():
    lines = common.format_summary("demo", common.summarize_predictions(TRUTH, PREDS))
    assert lines[0] == "MODEL: demo"
    assert lines[1] == "ACCURACY: 0.7500 (3/4)"
    assert lines[3].startswith("  AR") and "[1, 0, 1]" in lines[3]
    assert any("BOTH" in line and "precision=0.5000" in line for line in lines)


def test_saved_predictions_use_the_format_the_audit_report_reads(tmp_path):
    path = tmp_path / "nested" / "preds.json"
    rows = [{"text": f"prompt {i}"} for i in range(len(PREDS))]
    common.save_predictions(path, "demo", 0.75, PREDS, rows)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["preds"] == PREDS and saved["model"] == "demo"
    assert saved["input_hashes"] == [
        hashlib.sha256(row["text"].encode("utf-8")).hexdigest() for row in rows
    ]
