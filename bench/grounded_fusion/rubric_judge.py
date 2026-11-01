"""DRACO-style rubric grader.

Scores a free-text answer against a DRACO ``DracoRubric`` by asking an LLM, for
each weighted criterion, whether the answer exhibits/satisfies the requirement.

Scoring (matches DRACO semantics):
    score = sum(criterion.weight for each criterion the answer matches)
Positive criteria add their weight when matched (the answer did the good thing);
negative criteria subtract (their weight is < 0) when matched (the answer
committed the penalized behavior -- e.g. unsafe medical advice, citing Reddit).

``negative_penalty`` is reported separately because reducing it is the headline
claim for grounding-aware synthesis (keep confident-but-wrong panel responses out
of the final answer).
"""

from __future__ import annotations

import json

from cli.sr_bench.draco_rubric import (
    DEFAULT_BATCH_SIZE,
    MAX_ANSWER_CHARS,
    CriterionVerdict,
    RubricScore,
    aggregate,
    decode_verdicts,
    judge_messages,
)

from .datasets import DracoCriterion, DracoRubric
from .llm_client import ChatClient


class RubricJudge:
    def __init__(
        self,
        client: ChatClient,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_answer_chars: int = MAX_ANSWER_CHARS,
    ) -> None:
        self.client = client
        self.batch_size = batch_size
        self.max_answer_chars = max_answer_chars

    def score(self, problem: str, answer: str, rubric: DracoRubric) -> RubricScore:
        answer = (answer or "")[: self.max_answer_chars]
        verdicts: list[CriterionVerdict] = []
        crits = rubric.criteria
        for i in range(0, len(crits), self.batch_size):
            batch = crits[i : i + self.batch_size]
            matched_map = self._judge_batch(problem, answer, batch)
            for c in batch:
                m = matched_map.get(c.id, {})
                verdicts.append(
                    CriterionVerdict(
                        id=c.id,
                        matched=bool(m.get("matched", False)),
                        weight=c.weight,
                        section_title=c.section_title,
                        is_negative=c.is_negative,
                        evidence=str(m.get("evidence", ""))[:120],
                    )
                )
        return self._aggregate(rubric, verdicts)

    def _judge_batch(
        self, problem: str, answer: str, batch: list[DracoCriterion]
    ) -> dict[str, dict]:
        res = self.client.chat(
            messages=judge_messages(problem=problem, answer=answer, batch=batch),
            extra_body={
                "think": False
            },  # ask Qwen3 to skip thinking; ignored by others
        )
        return self._parse(res.content)

    @staticmethod
    def _parse(content: str) -> dict[str, dict]:
        if not content:
            return {}
        try:
            arr = decode_verdicts(content)
        except json.JSONDecodeError:
            return {}
        out: dict[str, dict] = {}
        for item in arr if isinstance(arr, list) else []:
            if isinstance(item, dict) and "id" in item:
                out[str(item["id"])] = item
        return out

    @staticmethod
    def _aggregate(
        rubric: DracoRubric, verdicts: list[CriterionVerdict]
    ) -> RubricScore:
        return aggregate(rubric=rubric, verdicts=verdicts)
