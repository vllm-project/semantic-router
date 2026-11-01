from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field

MAX_ANSWER_CHARS: int = 24000
DEFAULT_BATCH_SIZE: int = 8
_JSON_ARRAY_RE: re.Pattern[str] = re.compile(r"\[.*\]", re.DOTALL)
_SYSTEM: str = (
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


@dataclass
class DracoCriterion:
    id: str
    weight: float
    requirement: str
    section_id: str
    section_title: str

    @property
    def is_negative(self) -> bool:
        return self.weight < 0


@dataclass
class DracoSection:
    id: str
    title: str
    criteria: list[DracoCriterion]


@dataclass
class DracoRubric:
    id: str
    sections: list[DracoSection]

    @property
    def criteria(self) -> list[DracoCriterion]:
        return [c for s in self.sections for c in s.criteria]

    @property
    def positive_criteria(self) -> list[DracoCriterion]:
        return [c for c in self.criteria if not c.is_negative]

    @property
    def negative_criteria(self) -> list[DracoCriterion]:
        return [c for c in self.criteria if c.is_negative]

    @property
    def max_positive_score(self) -> float:
        return sum(c.weight for c in self.positive_criteria)

    @property
    def min_negative_score(self) -> float:
        return sum(c.weight for c in self.negative_criteria)


@dataclass
class DracoSample:
    id: str
    problem: str
    domain: str
    rubric: DracoRubric
    metadata: dict[str, object] = field(default_factory=dict)


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
    negative_penalty: float
    per_section: dict[str, float] = field(default_factory=dict)
    verdicts: list[CriterionVerdict] = field(default_factory=list)

    @property
    def normalized(self) -> float:
        if self.max_positive <= 0:
            return 0.0
        return max(0.0, min(1.0, self.total / self.max_positive))

    @property
    def n_negative_triggered(self) -> int:
        return sum(1 for v in self.verdicts if v.is_negative and v.matched)


def _parse_rubric(rubric_id: str, raw: object) -> DracoRubric:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise ValueError(f"rubric for {rubric_id!r} is not an object: {type(raw)}")
    sections: list[DracoSection] = []
    for section in raw.get("sections", []):
        section_id = str(section.get("id", ""))
        title = str(section.get("title", section_id))
        criteria = [
            DracoCriterion(
                id=str(c.get("id", f"{section_id}-{i}")),
                weight=float(c.get("weight", 0)),
                requirement=str(c.get("requirement", "")),
                section_id=section_id,
                section_title=title,
            )
            for i, c in enumerate(section.get("criteria", []))
        ]
        sections.append(DracoSection(id=section_id, title=title, criteria=criteria))
    return DracoRubric(id=str(raw.get("id", rubric_id)), sections=sections)


def parse_samples(data: object) -> list[DracoSample]:
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    if not isinstance(rows, list):
        raise ValueError("DRACO source must contain a list of rows")
    samples: list[DracoSample] = []
    for entry in rows:
        row = entry.get("row", entry) if isinstance(entry, dict) else entry
        if not isinstance(row, dict):
            raise ValueError("DRACO rows must be objects")
        samples.append(
            DracoSample(
                id=str(row["id"]),
                problem=str(row["problem"]),
                domain=str(row.get("domain", "unknown")),
                rubric=_parse_rubric(rubric_id=str(row["id"]), raw=row["answer"]),
            )
        )
    return samples


def select_samples(
    *,
    samples: list[DracoSample],
    domains: list[str] | None,
    seed: int,
    max_samples: int | None,
) -> list[DracoSample]:
    selected = list(samples)
    if domains:
        wanted = {domain.lower() for domain in domains}
        selected = [sample for sample in selected if sample.domain.lower() in wanted]
    random.Random(seed).shuffle(selected)
    return selected[:max_samples] if max_samples is not None else selected


def judge_messages(
    *, problem: str, answer: str, batch: list[DracoCriterion]
) -> list[dict[str, str]]:
    criteria = "\n".join(f'- id="{c.id}": {c.requirement}' for c in batch)
    user = (
        f"QUESTION:\n{problem}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"RUBRIC CRITERIA ({len(batch)}):\n{criteria}\n\n"
        "Return the JSON array now."
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


def decode_verdicts(content: str) -> object:
    text = content.strip()
    if not text.startswith("["):
        match = _JSON_ARRAY_RE.search(text)
        text = match.group(0) if match else "[]"
    return json.loads(text)


def aggregate(*, rubric: DracoRubric, verdicts: list[CriterionVerdict]) -> RubricScore:
    per_section: dict[str, float] = {}
    for verdict in verdicts:
        if verdict.matched:
            per_section[verdict.section_title] = (
                per_section.get(verdict.section_title, 0.0) + verdict.weight
            )
    return RubricScore(
        total=sum(v.weight for v in verdicts if v.matched),
        max_positive=rubric.max_positive_score,
        positive_earned=sum(
            v.weight for v in verdicts if v.matched and not v.is_negative
        ),
        negative_penalty=sum(v.weight for v in verdicts if v.matched and v.is_negative),
        per_section=per_section,
        verdicts=verdicts,
    )
