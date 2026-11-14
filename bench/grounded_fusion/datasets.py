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
from pathlib import Path

from cli.sr_bench.draco_rubric import (
    DracoCriterion,
    DracoRubric,
    DracoSample,
    DracoSection,
    _parse_rubric,
    parse_samples,
    select_samples,
)

__all__: list[str] = [
    "DracoCriterion",
    "DracoRubric",
    "DracoSample",
    "DracoSection",
    "get_dataset",
    "load_draco",
    "load_jsonl",
]


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
    return select_samples(
        samples=parse_samples(data),
        domains=domains,
        seed=seed,
        max_samples=max_samples,
    )


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
