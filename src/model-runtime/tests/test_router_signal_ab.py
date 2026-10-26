"""The router signal A/B tool's join, scoring and verdict rules (``tools/router_signal_ab.py``)."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "router_signal_ab.py"


@pytest.fixture(scope="module")
def ab():
    spec = importlib.util.spec_from_file_location("router_signal_ab", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_auc_counts_ties_half(ab) -> None:
    y = np.array([True, False, True, False])
    assert ab.auc(np.array([0.9, 0.1, 0.8, 0.2]), y) == 1.0
    assert ab.auc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5
    assert np.isnan(ab.auc(np.array([0.1, 0.2]), np.array([True, True])))


def test_recorded_answers_keep_a_decisions_answer_with_its_question_set(
    ab, tmp_path
) -> None:
    def bundle(questions: list[str], domain: float) -> dict:
        answers = {
            q: {"type": "choice", "probabilities": {"a": domain, "b": 1 - domain}}
            for q in questions
        }
        return {"path": "/v1/bundle",
                "request": {"tasks": [{"id": "1", "decisions": {"state": "hi", "questions": {q: {} for q in questions}}}]},
                "response": {"results": [{"id": "1", "decisions": {"answers": answers}}]}}  # fmt: skip

    log = tmp_path / "cpu.sock.jsonl"
    entries = [
        bundle(["domain_classifier:domain", "prompt_guard:attack"], 0.6),
        bundle(["domain_classifier:domain", "feedback_detector:feedback"], 0.7),
        {"path": "/v1/bundle", "request": {"tasks": [{"id": "1", "classify": {"model": "@prompt_guard", "input": [{"text": "hi"}]}}]},
         "response": {"results": [{"id": "1", "classify": {"labels": ["benign", "jailbreak"], "results": [
             {"probabilities": [0.4, 0.6], "windows": [{"probabilities": [0.9, 0.1]}, {"probabilities": [0.2, 0.8]}]}]}}]}},
    ]  # fmt: skip
    log.write_text("".join(json.dumps(e) + "\n" for e in entries))
    answers = ab.recorded_answers([str(tmp_path)])
    assert answers[("hi", "request")]["domain_classifier"]["probabilities"]["a"] == 0.6
    assert answers[("hi", "feedback")]["domain_classifier"]["probabilities"]["a"] == 0.7
    # A windowed Vela 1.0 head is read at its riskiest window, as the Router reads it.
    assert ab.vela1_scores("jailbreak", answers[("hi", "any")]) == {
        "benign": 0.2,
        "jailbreak": 0.8,
    }


def test_a_row_finds_the_trimmed_and_the_sampled_text_its_consumers_received(
    ab, tmp_path
) -> None:
    def classify(model: str, text: str, p: float) -> dict:
        return {"path": "/v1/bundle", "request": {"tasks": [{"id": "1", "classify": {"model": model, "input": [{"text": text}]}}]},
                "response": {"results": [{"id": "1", "classify": {"labels": ["a", "b"], "results": [{"probabilities": [p, 1 - p]}]}}]}}  # fmt: skip

    head, middle, tail = " head " + "h" * 80, "m" * 80, "t" * 80 + " tail "
    text = head + "x" * 100 + middle + "y" * 100 + tail
    sampled = ab.SAMPLED.join([head, middle, tail])
    entries = [classify("@domain_classifier", sampled, 0.1), classify("@pii_classifier", text.strip(), 0.2),
               classify("@prompt_guard", text, 0.3), classify("@feedback_detector", ab.SAMPLED.join(["a", "b", "c"]), 0.4)]  # fmt: skip
    (tmp_path / "cpu.sock.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in entries)
    )
    outputs = ab.Answers([str(tmp_path)]).outputs(text, "request")
    assert {model: item["probabilities"][0] for model, item in outputs.items()} == {
        "@domain_classifier": 0.1,
        "@pii_classifier": 0.2,
        "@prompt_guard": 0.3,
    }


def test_pii_scores_count_only_sensitive_identifiers(ab) -> None:
    spans = [
        {"label": "GPE", "probability": 0.99},
        {"label": "PERSON", "probability": 0.7},
    ]
    assert ab.vela2_scores("pii", {"pii_classifier": {"spans": spans}}) == {
        "yes": 0.7,
        "no": pytest.approx(0.3),
    }
    assert ab.vela2_scores("pii", {"pii_classifier": {"spans": spans[:1]}}) == {
        "yes": 0.0,
        "no": 1.0,
    }


def test_verdicts_read_the_routers_matched_rules(ab) -> None:
    assert ab.verdict("domain", {"label": "math"}, {"matched": {"domains": ["math"]}})
    assert not ab.verdict("domain", {"label": "math"}, {"matched": {}})
    assert ab.verdict("feedback", {"label": "NO_FEEDBACK"}, {"matched": {}})
    assert ab.verdict(
        "modality", {"label": "DIFFUSION"}, {"matched": {"modality": ["BOTH"]}}
    )
    assert ab.verdict("jailbreak", {"label": "benign"}, {"matched": {}})
    assert ab.verdict(
        "hallucination", {"label": "hallucinated"}, {"spans": [{"start": 0}]}
    )


def test_a_matched_threshold_keeps_the_share_of_values_it_admits(ab) -> None:
    values = [0.1, 0.2, 0.3, 0.4]
    assert ab.matched_threshold(0.25, values, above=True) == pytest.approx(0.35)
    assert ab.matched_threshold(0.5, values, above=False) == pytest.approx(0.25)
    assert ab.matched_threshold(0.0, values, above=True) == pytest.approx(0.7)
    assert ab.matched_threshold(1.0, values, above=True) == pytest.approx(0.05)
    assert ab.matched_threshold(0.0, values, above=False) == pytest.approx(0.05)
    # A span model scores 0 where it finds no span: a threshold never splits the zeros.
    spans = [0.0] * 8 + [0.6, 0.9]
    assert ab.matched_threshold(0.45, spans, above=True) == pytest.approx(0.3)


def test_a_confidence_floor_falls_back_below_its_threshold(ab) -> None:
    rows = [{"id": "1", "label": "math"}, {"id": "2", "label": "other"}]
    preds = {
        "1": {"scores": {"math": 0.6, "other": 0.4}},
        "2": {"scores": {"math": 0.55, "other": 0.45}},
    }
    assert ab.floor_point("domain", rows, preds, 0.5) == {
        "below": 0.0,
        "balanced_accuracy": 0.5,
    }
    assert ab.floor_point("domain", rows, preds, 0.58) == {
        "below": 0.5,
        "balanced_accuracy": 1.0,
    }


def test_the_arms_differ_only_in_their_model_catalog(ab) -> None:
    base = ab.full_config(18899)
    vela2, vela1 = ab.arm_config(base, "vela2"), ab.arm_config(base, "vela1")
    for arm in (vela2, vela1):
        assert {k: v for k, v in arm.items() if k != "global"} == {
            k: v for k, v in base.items() if k != "global"
        }
    modules = vela2["global"]["model_catalog"]["modules"]
    assert modules["modality_detector"]["classifier"] == {"use_cpu": True}
    assert "system" not in vela2["global"]["model_catalog"]
    catalog = vela1["global"]["model_catalog"]
    assert catalog["system"] == ab.VELA1_SYSTEM
    assert catalog["modules"] == base["global"]["model_catalog"]["modules"]


def test_a_threshold_keeps_the_decimals_its_operating_point_needs(ab) -> None:
    spread = [i / 1000 for i in range(1000)]
    assert ab.shipped_threshold(0.4512, spread, above=True) == 0.45
    # Scores piled up near 1: two decimals would round 0.9984 to 1.0 and flag
    # nothing, and three to 0.998, which flags a tenth more of these scores.
    piled = [
        0.9970,
        0.9975,
        0.9979,
        0.9982,
        0.9986,
        0.9990,
        0.9993,
        0.9996,
        0.9998,
        0.9999,
    ]
    assert ab.shipped_threshold(0.9984, piled, above=True) == 0.9984
    assert ab.shipped_threshold(0.9987, piled, above=True) == 0.999


def test_vela1_scores_drop_shape_shifted_responses(ab) -> None:
    labels = ["domain_a", "domain_b"]
    assert (
        ab.vela1_scores(
            "domain", {"@domain_classifier": {"labels": labels, "probabilities": [0.9]}}
        )
        is None
    )
    assert (
        ab.vela1_scores(
            "domain", {"@domain_classifier": {"labels": [], "probabilities": [0.9]}}
        )
        is None
    )
    assert ab.vela1_scores(
        "domain",
        {"@domain_classifier": {"labels": labels, "probabilities": [0.9, 0.1]}},
    ) == {"domain_a": 0.9, "domain_b": 0.1}


def test_vela1_scores_ignore_windows_shorter_than_the_label_list(ab) -> None:
    outputs = {
        "@prompt_guard": {
            "labels": ["safe", "jailbreak"],
            "windows": [{"probabilities": [0.8]}, {"probabilities": [0.6, 0.9]}],
        }
    }
    # The first window cannot index the jailbreak label; the second decides.
    assert ab.vela1_scores("jailbreak", outputs) == {"safe": 0.6, "jailbreak": 0.9}
    short_only = {
        "@prompt_guard": {
            "labels": ["safe", "jailbreak"],
            "windows": [{"probabilities": [0.8]}],
        }
    }
    assert ab.vela1_scores("jailbreak", short_only) is None


def test_a_matched_threshold_without_values_is_none(ab) -> None:
    assert ab.matched_threshold(0.5, [], above=False) is None
    assert ab.matched_threshold(0.5, [0.2, 0.8], above=False) == pytest.approx(0.5)


def test_join_skips_rows_missing_from_the_rows_file(
    ab, tmp_path, capsys, monkeypatch
) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        json.dumps(
            {
                "id": "panel/1",
                "task": "domain",
                "file": "suite-a",
                "body": {"text": "hello"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run = tmp_path / "run.jsonl"
    run.write_text(
        json.dumps({"id": "panel/1", "status": 200, "ms": 1.0, "response": {}})
        + "\n"
        + json.dumps({"id": "ghost", "status": 200, "ms": 1.0, "response": {}})
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "logs").mkdir()
    out = tmp_path / "out"

    class FakeAnswers:
        def __init__(self, log_dirs):
            pass

        def outputs(self, text, kind):
            return {
                "@domain_classifier": {
                    "labels": ["domain_a", "domain_b"],
                    "probabilities": [0.7, 0.3],
                }
            }

    monkeypatch.setattr(ab, "Answers", FakeAnswers)
    args = argparse.Namespace(
        rows=str(rows),
        run=[str(run)],
        log_dir=[str(tmp_path / "logs")],
        arm="vela1",
        out=str(out),
    )
    ab.cmd_join(args)
    pred = out / "domain" / "suite-a.pred.jsonl"
    entries = [
        json.loads(line) for line in pred.read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["id"] for entry in entries] == ["panel/1"]
    assert "rows without a matching row: 1" in capsys.readouterr().out
