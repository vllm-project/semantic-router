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
import re
from dataclasses import dataclass, field

from .datasets import DracoCriterion, DracoRubric
from .llm_client import ChatClient

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


@dataclass
class CriterionVerdict:
    id: str
    matched: bool
    weight: float
    section_title: str
    is_negative: bool
    evidence: str = ""


@dataclass
class RubricScore:
    total: float
    max_positive: float
    positive_earned: float
    negative_penalty: float  # <= 0; sum of negative weights that were triggered
    per_section: dict[str, float] = field(default_factory=dict)
    verdicts: list[CriterionVerdict] = field(default_factory=list)

    @property
    def normalized(self) -> float:
        """Total clamped to [0,1] against the positive ceiling (penalties can push <0)."""
        if self.max_positive <= 0:
            return 0.0
        return max(0.0, min(1.0, self.total / self.max_positive))

    @property
    def n_negative_triggered(self) -> int:
        return sum(1 for v in self.verdicts if v.is_negative and v.matched)


_SYSTEM = (
    "You are a strict grader. You are given a QUESTION, an ANSWER, and a list of "
    "RUBRIC CRITERIA. For each criterion decide whether the ANSWER exhibits or "
    "satisfies what the criterion's requirement describes. A criterion may describe "
    "a GOOD property (reward) or a BAD behavior (penalty); in BOTH cases return "
    "matched=true if and only if the ANSWER actually does what the requirement "
    "describes. Judge only what the answer states; do not reward omissions. Return "
    "ONLY a JSON array, one object per criterion: "
    '[{"id": "<criterion id>", "matched": true|false, "evidence": "<=15 words"}]. '
    "No prose, no markdown fences."
)


class RubricJudge:
    def __init__(
        self,
        client: ChatClient,
        batch_size: int = 8,
        max_answer_chars: int = 24000,
    ):
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
        crit_lines = "\n".join(f'- id="{c.id}": {c.requirement}' for c in batch)
        user = (
            f"QUESTION:\n{problem}\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"RUBRIC CRITERIA ({len(batch)}):\n{crit_lines}\n\n"
            "Return the JSON array now."
        )
        res = self.client.chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            extra_body={
                "think": False
            },  # ask Qwen3 to skip thinking; ignored by others
        )
        return self._parse(res.content)

    @staticmethod
    def _parse(content: str) -> dict[str, dict]:
        if not content:
            return {}
        text = content.strip()
        if not text.startswith("["):
            m = _JSON_ARRAY_RE.search(text)
            text = m.group(0) if m else "[]"
        try:
            arr = json.loads(text)
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
        total = sum(v.weight for v in verdicts if v.matched)
        positive_earned = sum(
            v.weight for v in verdicts if v.matched and not v.is_negative
        )
        negative_penalty = sum(
            v.weight for v in verdicts if v.matched and v.is_negative
        )
        per_section: dict[str, float] = {}
        for v in verdicts:
            if v.matched:
                per_section[v.section_title] = (
                    per_section.get(v.section_title, 0.0) + v.weight
                )
        return RubricScore(
            total=total,
            max_positive=rubric.max_positive_score,
            positive_earned=positive_earned,
            negative_penalty=negative_penalty,
            per_section=per_section,
            verdicts=verdicts,
        )
