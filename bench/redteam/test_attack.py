"""Attack-logic tests. A stub scorer keeps these independent of any model."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from bench.redteam.attack import (
    AttackResult,
    greedy_suffix_attack,
    summarize,
)
from bench.redteam.datasets import load_local


class WordCountScorer:
    """Scores 0.9 until ``drop_after`` benign words are appended, then 0.1."""

    def __init__(self, drop_after: int) -> None:
        self.drop_after = drop_after
        self.calls = 0

    def score(self, texts: Sequence[str]) -> list[float]:
        self.calls += len(texts)
        scores: list[float] = []
        for text in texts:
            appended = len(text.split()) - 1
            scores.append(0.1 if appended >= self.drop_after else 0.9 - appended * 0.01)
        return scores


class ConstantScorer:
    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, texts: Sequence[str]) -> list[float]:
        return [self.value] * len(texts)


def test_undetected_prompt_is_not_attacked() -> None:
    scorer = ConstantScorer(0.2)
    result = greedy_suffix_attack("harmless", scorer, threshold=0.7)
    assert not result.detected
    assert not result.flipped
    assert result.queries == 1
    assert result.suffix == []


def test_detected_prompt_flips_with_two_words() -> None:
    scorer = WordCountScorer(drop_after=2)
    result = greedy_suffix_attack("bad", scorer, threshold=0.7, max_words=5)
    assert result.detected
    assert result.flipped
    assert result.suffix_length == 2
    assert result.final_score < 0.7


def test_flat_score_stops_instead_of_exhausting_the_pool() -> None:
    scorer = ConstantScorer(0.95)
    result = greedy_suffix_attack(
        "bad", scorer, threshold=0.7, word_pool=("a", "b", "c"), max_words=5
    )
    assert result.detected
    assert not result.flipped
    # One baseline query plus a single round that fails to improve.
    assert result.queries == 4


def test_max_words_bounds_the_suffix() -> None:
    scorer = WordCountScorer(drop_after=99)
    result = greedy_suffix_attack(
        "bad", scorer, threshold=0.7, word_pool=tuple("abcdefgh"), max_words=3
    )
    assert not result.flipped
    assert result.suffix_length == 3


def test_a_word_is_never_reused() -> None:
    scorer = WordCountScorer(drop_after=99)
    result = greedy_suffix_attack(
        "bad", scorer, threshold=0.7, word_pool=("x", "y", "z"), max_words=3
    )
    assert sorted(result.suffix) == ["x", "y", "z"]


def test_summary_reports_recall_and_flip_rate() -> None:
    results = [
        AttackResult("a", 0.9, True, True, 0.1, ["p", "q"], 10),
        AttackResult("b", 0.8, True, False, 0.75, ["p"], 20),
        AttackResult("c", 0.2, False, False, 0.2, [], 1),
    ]
    report = summarize(results).as_dict()
    assert report["total"] == 3
    assert report["detected"] == 2
    assert report["baseline_recall"] == pytest.approx(0.6667, abs=1e-4)
    assert report["flipped"] == 1
    assert report["flip_rate"] == pytest.approx(0.5)
    assert report["suffix_max"] == 2


def test_empty_summary_does_not_divide_by_zero() -> None:
    report = summarize([]).as_dict()
    assert report["baseline_recall"] == 0.0
    assert report["flip_rate"] == 0.0


def test_local_dataset_reads_goal_field(tmp_path: Path) -> None:
    path = tmp_path / "behaviors.json"
    path.write_text(json.dumps([{"goal": "first"}, {"prompt": "second"}]))
    assert load_local(path) == ["first", "second"]


def test_local_dataset_rejects_a_file_with_no_prompts(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps([{"unrelated": "value"}]))
    with pytest.raises(ValueError, match="no prompts found"):
        load_local(path)
