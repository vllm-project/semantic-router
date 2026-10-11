"""Raw logits to the Vela 2.0 response: answers, sets, spans and thresholds."""

from __future__ import annotations

import math

import numpy as np
from vllm_srun.errors import INVALID_MODEL_OUTPUT
from vllm_srun.families.vela2.answers import Answerer
from vllm_srun.families.vela2.calibration import Calibration, sigmoid, softmax
from vllm_srun.families.vela2.raw import RawRow, RawSpan
from vllm_srun.families.vela2.request import QuestionReader
from vllm_srun.testing.vela2 import calibration

CAL = Calibration(calibration(decoder=True))
TEXT = "mail tom.b@ex.com now"
QUESTIONS = {
    "domain": {
        "type": "choice",
        "instructions": "Which subject?",
        "criteria": {"math": "math", "law": "law", "other": "anything else"},
    },
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "level": {
        "type": "score",
        "instructions": "How hard is it?",
        "criteria": ["easy", "medium", {"name": "hard"}],
    },
    "topics": {
        "type": "set",
        "instructions": "Which topics?",
        "criteria": {"billing": "money", "travel": "trips"},
    },
    "pii": {
        "type": "span",
        "instructions": "Find personal data.",
        "criteria": {"EMAIL_ADDRESS": "an email", "PERSON": "a name"},
        "threshold": 0.5,
    },
}


def answer(raw: RawRow, report_heads: bool = False) -> dict:
    plan = QuestionReader(CAL, broad_head=True).read(TEXT, QUESTIONS)
    answerer = Answerer(CAL, report_heads=report_heads)
    response: dict = {"answers": {}}
    for question in plan.questions:
        answerer.answer(question, raw, plan.state, response)
    return response


def logits(question_id: str, *values: float) -> np.ndarray:
    plan = QuestionReader(CAL, broad_head=True).read(TEXT, QUESTIONS)
    (question,) = (q for q in plan.questions if q.id == question_id)
    return np.array(values[: len(question.names) + int(question.abstain)])


RAW = RawRow(
    logits={
        "domain": logits("domain", 2.0, 0.5, -1.0, 0.3),
        "urgent": logits("urgent", -0.4, 1.2, 0.1),
        "level": logits("level", 0.2, 1.5, -0.7, 0.0),
        "topics": logits("topics", 1.4, -2.0),
    },
    span=RawSpan(
        labels=["EMAIL_ADDRESS", "PERSON"],
        offsets=np.array([[0, 4], [5, 17], [18, 21]]),
        logits=np.array([[-5.0, -5.0], [4.0, -5.0], [-5.0, -5.0]]),
        alias={"PERSON": "NAME"},
    ),
)


def test_choice_noul_and_score_answers() -> None:
    answers = answer(RAW)["answers"]
    choice = softmax(RAW.logits["domain"][:3] / 1.3)
    assert answers["domain"]["choice"] == "math"
    assert answers["domain"]["confidence"] == choice[0] - max(choice[1:])
    full = softmax(RAW.logits["domain"] / 1.3)
    assert answers["domain"]["abstain_probability"] == float(full[-1])
    assert "abstain_probability" not in answers["level"]
    assert answers["urgent"] == {
        "type": "noul",
        "noul": float(softmax(RAW.logits["urgent"][:2] / 1.3)[1]),
    }
    levels = softmax(RAW.logits["level"][:3] / 1.1)
    mean = math.fsum(i * p for i, p in enumerate(levels))
    variance = math.fsum(p * (i - mean) ** 2 for i, p in enumerate(levels))
    assert answers["level"]["score"] == mean
    assert answers["level"]["confidence"] == 1.0 - variance / (8 / 12)
    assert answers["level"]["legend"] == {
        "0": "easy",
        "1": "medium",
        "2": '{"name":"hard"}',
    }


def test_repeated_score_levels_answer_per_level() -> None:
    question = {"type": "score", "instructions": "How?", "criteria": ["a", "b", "a"]}
    plan = QuestionReader(CAL, broad_head=True).read(TEXT, {"s": question})
    values = np.array([0.4, -1.0, 1.6])
    response: dict = {"answers": {}}
    Answerer(CAL).answer(
        plan.questions[0], RawRow(logits={"s": values}), plan.state, response
    )
    levels = softmax(values / 1.1)
    answer = response["answers"]["s"]
    assert answer["probabilities"] == {str(i): float(p) for i, p in enumerate(levels)}
    assert answer["score"] == math.fsum(i * p for i, p in enumerate(levels))
    assert answer["legend"] == {"0": "a", "1": "b", "2": "a"}


def test_set_answers_select_above_the_threshold_with_label_views() -> None:
    response = answer(RAW)
    scores = sigmoid(RAW.logits["topics"] / 1.05)
    assert response["sets"]["topics"]["selected"] == ["billing"]
    assert response["thresholds"]["topics"] == 0.3
    assert response["answers"]["topics.travel"] == {"type": "noul", "noul": scores[1]}


def test_span_answers_decode_words_and_report_heads() -> None:
    response = answer(RAW, report_heads=True)
    probability = float(sigmoid(np.float64(4.0) / 0.4))
    assert response["spans"]["pii"] == [
        {
            "label": "EMAIL_ADDRESS",
            "start": 5,
            "end": 17,
            "text": "tom.b@ex.com",
            "probability": float(np.float32(probability)),
        }
    ]
    assert response["thresholds"]["pii"] == 0.5
    assert response["answers"]["pii"] == {"type": "noul", "noul": probability}
    assert response["span_heads"] == {"pii": "router"}
    assert "span_heads" not in answer(RAW)


def test_span_labels_come_back_under_the_callers_names() -> None:
    span = RawSpan(
        labels=["EMAIL_ADDRESS", "PERSON"],
        offsets=np.array([[0, 4], [5, 17], [18, 21]]),
        logits=np.array([[3.0, -5.0], [-5.0, 3.0], [-5.0, -5.0]]),
        alias={"PERSON": "NAME"},
    )
    raw = RawRow(logits=RAW.logits, span=span)
    labels = [entry["label"] for entry in answer(raw)["spans"]["pii"]]
    assert labels == ["EMAIL_ADDRESS", "NAME"]


def test_malformed_outputs_fail_only_their_question() -> None:
    raw = RawRow(
        logits={**RAW.logits, "domain": np.array([np.nan, 0.0, 0.0, 0.0])},
        span=RAW.span,
    )
    del raw.logits["level"]
    answers = answer(raw)["answers"]
    assert answers["domain"] == {"type": "choice", "error": INVALID_MODEL_OUTPUT}
    assert answers["level"] == {"type": "score", "error": INVALID_MODEL_OUTPUT}
    assert answers["urgent"]["type"] == "noul"
