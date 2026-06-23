"""DRACO dataset loader for the grounding-aware fusion benchmark.

DRACO (perplexity-ai/draco) is a *rubric-graded* deep-research benchmark, not
reference-answer QA. Each item has:

- ``problem``  -- a long, open-ended expert prompt.
- ``answer``   -- a JSON **scoring rubric** (NOT a reference answer): a list of
  sections, each with weighted criteria. Positive weights reward satisfied
  requirements; negative weights penalize confident-but-wrong / unsafe / badly
  sourced claims. Score = sum(weight for each satisfied criterion).
- ``domain``   -- topical domain (Medicine, Law, Finance, ...).

This module parses the HuggingFace datasets-server JSON export
(``{"rows": [{"row": {...}}, ...]}``) into typed objects the harness scores
against. It does no network I/O -- point it at a local file.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DracoCriterion:
    """A single weighted rubric requirement."""

    id: str
    weight: float
    requirement: str
    section_id: str
    section_title: str

    @property
    def is_negative(self) -> bool:
        """Negative-weight criteria penalize bad behavior (the confident-wrong axis)."""
        return self.weight < 0


@dataclass
class DracoSection:
    """A rubric section (e.g. 'Factual Accuracy', 'Citation Quality')."""

    id: str
    title: str
    criteria: list[DracoCriterion]


@dataclass
class DracoRubric:
    """The full weighted rubric for one DRACO problem."""

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
        """Sum of positive weights -- the ceiling used to normalize scores to [0,1]."""
        return sum(c.weight for c in self.positive_criteria)

    @property
    def min_negative_score(self) -> float:
        """Sum of negative weights -- the floor of the penalty subtotal."""
        return sum(c.weight for c in self.negative_criteria)


@dataclass
class DracoSample:
    """One DRACO problem + its rubric."""

    id: str
    problem: str
    domain: str
    rubric: DracoRubric
    metadata: dict[str, object] = field(default_factory=dict)


def _parse_rubric(rubric_id: str, raw: object) -> DracoRubric:
    """Parse the ``answer`` field (a JSON string or already-decoded dict)."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise ValueError(f"rubric for {rubric_id!r} is not an object: {type(raw)}")

    sections: list[DracoSection] = []
    for s in raw.get("sections", []):
        sec_id = str(s.get("id", ""))
        sec_title = str(s.get("title", sec_id))
        criteria = [
            DracoCriterion(
                id=str(c.get("id", f"{sec_id}-{i}")),
                weight=float(c.get("weight", 0)),
                requirement=str(c.get("requirement", "")),
                section_id=sec_id,
                section_title=sec_title,
            )
            for i, c in enumerate(s.get("criteria", []))
        ]
        sections.append(DracoSection(id=sec_id, title=sec_title, criteria=criteria))
    return DracoRubric(id=str(raw.get("id", rubric_id)), sections=sections)


def load_draco(
    path: str,
    max_samples: int | None = None,
    domains: list[str] | None = None,
    seed: int = 42,
) -> list[DracoSample]:
    """Load DRACO samples from a datasets-server JSON export.

    Args:
        path: Path to ``draco.json``.
        max_samples: Cap the number of samples (after domain filtering + shuffle).
        domains: If given, keep only these domains (case-insensitive).
        seed: Shuffle seed for reproducible subsetting.

    Returns:
        List of ``DracoSample``.
    """
    data = json.loads(Path(path).expanduser().read_text())
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data

    samples: list[DracoSample] = []
    for entry in rows:
        row = entry.get("row", entry) if isinstance(entry, dict) else entry
        sample = DracoSample(
            id=str(row["id"]),
            problem=str(row["problem"]),
            domain=str(row.get("domain", "unknown")),
            rubric=_parse_rubric(str(row["id"]), row["answer"]),
        )
        samples.append(sample)

    if domains:
        wanted = {d.lower() for d in domains}
        samples = [s for s in samples if s.domain.lower() in wanted]

    rng = random.Random(seed)
    rng.shuffle(samples)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def load_jsonl(
    path: str,
    max_samples: int | None = None,
    domains: list[str] | None = None,
    seed: int = 42,
) -> list[DracoSample]:
    """Load a generic rubric-graded dataset from JSONL (one row per line).

    Each line mirrors a DRACO row -- ``{id, problem, domain, answer}`` where
    ``answer`` is the weighted rubric -- plus an optional ``context`` (gold passage)
    surfaced in ``metadata["context"]``. This is the seam for context-mode grounding:
    a context-grounded benchmark (RAGTruth / HotpotQA-with-gold-passages reshaped to
    this row format) is scored against its real source rather than panel agreement.
    """
    samples: list[DracoSample] = []
    for raw_line in Path(path).expanduser().read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        samples.append(
            DracoSample(
                id=str(row["id"]),
                problem=str(row["problem"]),
                domain=str(row.get("domain", "unknown")),
                rubric=_parse_rubric(str(row["id"]), row["answer"]),
                metadata={"context": str(row.get("context", "") or "")},
            )
        )

    if domains:
        wanted = {d.lower() for d in domains}
        samples = [s for s in samples if s.domain.lower() in wanted]
    rng = random.Random(seed)
    rng.shuffle(samples)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def get_dataset(name: str, **kwargs) -> list[DracoSample]:
    """Dispatch a dataset by name (``draco`` or a generic rubric-graded ``jsonl``)."""
    name = name.lower()
    if name == "draco":
        path = kwargs.pop("path", None) or kwargs.pop("draco_path", None)
        if not path:
            raise ValueError("draco dataset requires path=/path/to/draco.json")
        return load_draco(path, **kwargs)
    if name == "jsonl":
        path = kwargs.pop("path", None) or kwargs.pop("draco_path", None)
        if not path:
            raise ValueError("jsonl dataset requires path=/path/to/dataset.jsonl")
        return load_jsonl(path, **kwargs)
    raise ValueError(f"unknown dataset: {name!r}")
