"""Offline tests for the confidence result collector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tuning import collect_confidence_results as collector
from tuning.collect_confidence_results import (
    _extract_answer,
    _extract_avg_logprob,
    _read_questions,
    _result_from_response,
)


def test_extract_avg_logprob_accepts_dashscope_message_layout():
    choice = {
        "message": {
            "logprobs": {
                "content": [
                    {"token": "A", "logprob": -1.0},
                    {"token": "!", "logprob": -2.0},
                ]
            }
        }
    }

    average, count = _extract_avg_logprob(choice, choice["message"])

    assert average == pytest.approx(-1.5)
    assert count == 2


def test_extract_avg_logprob_accepts_openai_choice_layout():
    choice = {
        "logprobs": {
            "content": [
                {"token": "A", "logprob": -0.25},
            ]
        }
    }

    average, count = _extract_avg_logprob(choice, {})

    assert average == pytest.approx(-0.25)
    assert count == 1


def test_result_parser_extracts_answer_and_correctness():
    response = {
        "choices": [
            {
                "message": {
                    "content": "Answer: [B]",
                    "role": "assistant",
                    "logprobs": {"content": [{"token": "B", "logprob": -0.5}]},
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 3},
    }
    item = {
        "question_id": "q1",
        "category": "math",
        "prompt": "Choose an answer.",
        "correct_answer": "B",
        "split": "calibration",
    }

    result = _result_from_response(response, item, "qwen3-8b", True)

    assert result["predicted"] == "B"
    assert result["correct"] is True
    assert result["avg_logprob"] == pytest.approx(-0.5)
    assert result["usage"]["total_tokens"] == 3


def test_answer_parser_uses_last_explicit_answer():
    assert _extract_answer("The options are A and B. Answer: C") == "C"


def test_dataset_requires_all_splits(tmp_path: Path):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "category": "math",
                    "prompt": "Choose A.",
                    "correct_answer": "A",
                    "split": "calibration",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing non-empty splits"):
        _read_questions(path)


def test_collection_stops_submitting_after_first_failure(monkeypatch):
    calls = []

    def fail_once(**kwargs):
        calls.append(kwargs["item"]["question_id"])
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(collector, "_call_model", fail_once)

    items = [
        {
            "question_id": question_id,
            "category": "math",
            "prompt": "Choose an answer.",
            "correct_answer": "A",
            "split": "calibration",
        }
        for question_id in ("q1", "q2", "q3")
    ]

    with pytest.raises(RuntimeError, match="synthetic failure"):
        collector._collect_model(
            items,
            stage="test",
            endpoint="https://example.test",
            api_key="test-key",
            model="test-model",
            include_logprobs=False,
            max_tokens=16,
            timeout=1,
            retries=0,
            max_concurrency=1,
        )

    assert calls == ["q1"]
