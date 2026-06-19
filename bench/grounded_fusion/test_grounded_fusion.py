"""Unit tests for the grounded-fusion harness pure logic (no model/router needed).

Run: .venv-bench/bin/python -m pytest bench/grounded_fusion/test_grounded_fusion.py -q
"""

import json
import os
import re

from bench.grounded_fusion.datasets import load_draco
from bench.grounded_fusion.llm_client import ChatResult
from bench.grounded_fusion.rubric_judge import RubricJudge

FIXTURE = os.path.join(os.path.dirname(__file__), "testdata", "draco_fixture.json")


# ---- dataset loader --------------------------------------------------------


def test_loader_parses_fixture():
    samples = load_draco(FIXTURE)
    assert len(samples) == 2
    by_id = {s.id: s for s in samples}
    med = by_id["fixture-med-1"]
    assert med.domain == "Medicine"
    assert len(med.rubric.criteria) == 3
    assert med.rubric.max_positive_score == 15  # 10 + 5
    assert med.rubric.min_negative_score == -100
    assert len(med.rubric.negative_criteria) == 1


def test_loader_domain_filter_and_cap():
    only_fin = load_draco(FIXTURE, domains=["finance"])
    assert {s.domain for s in only_fin} == {"Finance"}
    capped = load_draco(FIXTURE, max_samples=1)
    assert len(capped) == 1


# ---- rubric grader math ----------------------------------------------------


class _FakeClient:
    """Returns matched=True only for the criterion ids in ``matched``."""

    def __init__(self, matched):
        self.matched = set(matched)

    def chat(self, messages, **kwargs):
        user = messages[-1]["content"]
        ids = re.findall(r'id="([^"]+)"', user)
        arr = [{"id": i, "matched": i in self.matched, "evidence": ""} for i in ids]
        return ChatResult(content=json.dumps(arr))


def _med_rubric():
    return {s.id: s for s in load_draco(FIXTURE)}["fixture-med-1"].rubric


def test_grader_rewards_positive_only():
    rubric = _med_rubric()
    judge = RubricJudge(
        _FakeClient(["call-emergency", "mention-aspirin"]), batch_size=2
    )
    score = judge.score("q", "good safe answer", rubric)
    assert score.total == 15
    assert score.positive_earned == 15
    assert score.negative_penalty == 0
    assert score.n_negative_triggered == 0
    assert score.normalized == 1.0


def test_grader_applies_negative_penalty():
    rubric = _med_rubric()
    # Answer triggers the unsafe "wait at home" negative criterion.
    judge = RubricJudge(_FakeClient(["call-emergency", "wait-home"]), batch_size=10)
    score = judge.score("q", "bad answer", rubric)
    assert score.total == 10 + (-100)
    assert score.positive_earned == 10
    assert score.negative_penalty == -100
    assert score.n_negative_triggered == 1
    assert score.normalized == 0.0  # clamped


def test_grader_handles_no_matches():
    rubric = _med_rubric()
    judge = RubricJudge(_FakeClient([]), batch_size=3)
    score = judge.score("q", "irrelevant", rubric)
    assert score.total == 0
    assert score.normalized == 0.0


# ---- JSON parsing robustness ----------------------------------------------


def test_parse_tolerates_fences_and_prose():
    parsed = RubricJudge._parse(
        'Here you go:\n```json\n[{"id":"a","matched":true}]\n```'
    )
    assert parsed["a"]["matched"] is True
    assert RubricJudge._parse("garbage") == {}
