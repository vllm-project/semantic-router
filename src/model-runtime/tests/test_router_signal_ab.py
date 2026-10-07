"""The router signal A/B tool's join, scoring and verdict rules (``tools/router_signal_ab.py``)."""

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


def test_the_vela2_arm_differs_only_in_its_model_catalog(ab, tmp_path) -> None:
    base = ab.full_config(18899)
    bound = ab.bind_vela2(base, {"domain_classifier": ab.LABELS})
    assert {k: v for k, v in bound.items() if k != "global"} == {
        k: v for k, v in base.items() if k != "global"
    }
    catalog = bound["global"]["model_catalog"]
    assert catalog["deployments"]["vela2"]["artifact"] == "vllm-sr/Vela-2.0-0.3B"
    assert catalog["bindings"] == {
        "domain_classifier": {"contract": ab.LABELS, "deployment": "vela2"}
    }
    assert catalog["modules"] == base["global"]["model_catalog"]["modules"]
