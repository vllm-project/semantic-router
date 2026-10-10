"""Training rows: loading, validation, image path resolution and exact token costs.

Rows follow the Omni training-row contract (``vision_format`` docstring): the Vega row plus
``images`` relative to the directory of the row file. A row's cost is its exact input token count
(rendered prompt plus merged image patches at the 1.6 MP cap), computed without decoding pixels.
Rows that cannot be trained on as written are dropped with a reason, never truncated.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from d25.omni.common import vision_format
from d25.omni.eval.shards import read_jsonl
from d25.omni.model import inputs


@dataclass
class TrainRow:
    id: str
    corpus: str
    source: str
    family: str
    state: Any
    question: dict[str, Any]
    target: list[float]
    weight: float = 1.0
    images: list[str] = field(default_factory=list)
    tokens: int = 0

    @property
    def n_options(self) -> int:
        return len(self.target)


def resolve(ref: str, root: Path) -> str:
    if ref.startswith("data:") or Path(ref).is_absolute():
        return ref
    return str(root / ref)


def read_rows(
    paths: Sequence[str | Path], corpus: str
) -> tuple[list[TrainRow], collections.Counter]:
    """Validated rows of ``paths``; returns the rows and drop counts by reason."""
    rows: list[TrainRow] = []
    dropped: collections.Counter = collections.Counter()
    for name in paths:
        path = Path(name)
        for raw in read_jsonl(path):
            try:
                vision_format.validate_row(raw)
            except (KeyError, TypeError, ValueError) as error:
                dropped[f"invalid: {type(error).__name__}"] += 1
                continue
            weight = float(raw.get("weight", 1.0))
            if not weight > 0:
                dropped["non-positive weight"] += 1
                continue
            rows.append(
                TrainRow(
                    id=raw["id"],
                    corpus=corpus,
                    source=raw["source"],
                    family=raw["family"],
                    state=raw["state"],
                    question=raw["question"],
                    target=[float(v) for v in raw["target"]],
                    weight=weight,
                    images=[
                        resolve(ref, path.parent) for ref in raw.get("images") or []
                    ],
                )
            )
    return rows, dropped


def measure(
    rows: Sequence[TrainRow],
    processor,
    codes: Sequence[str],
    prompt: str,
    max_length: int,
) -> list[tuple[int, str | None]]:
    """``(tokens, drop reason or None)`` per row."""
    results: list[tuple[int, str | None]] = []
    for row in rows:
        try:
            text = inputs.render(
                processor, prompt, row.state, row.question, codes, len(row.images)
            )
        except (TypeError, ValueError):
            results.append((0, "prompt rendering failed"))
            continue
        if inputs.placeholder_conflict(text, len(row.images)):
            results.append((0, "literal image or video placeholder in the row text"))
            continue
        try:
            sizes = [inputs.image_size(ref) for ref in row.images]
            tokens = inputs.cost(processor, text, sizes).tokens
        except (OSError, ValueError) as error:
            results.append((0, f"image rejected: {type(error).__name__}"))
            continue
        results.append(
            (tokens, f"over max_length {max_length}" if tokens > max_length else None)
        )
    return results


def unique_ids(rows: Sequence[TrainRow]) -> None:
    seen: set[str] = set()
    for row in rows:
        if row.id in seen:
            raise ValueError(f"duplicate training row id {row.id!r}")
        seen.add(row.id)
