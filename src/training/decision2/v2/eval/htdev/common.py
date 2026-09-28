"""Shared pieces of the HT-DEV converters (`v2/eval/htdev/sources/<key>.py`).

A converter exports `SPEC: SourceSpec` and `candidates(root) -> Iterator[HtCandidate]`
over its pinned snapshot, exactly like the C1 converters (`v2/eval/sealed/schema.py`
rules: one frozen template per task, label-blind states, no model-generated columns).
On top of the C1 candidate an HT-DEV candidate carries its split and a provenance
pointer (file relpath + row index) and a bootstrap cluster (the group unless the source
names a coarser one); all stay on the pool / gold side, never in a prompt.

Choice option order is `display_order(item id, n, salt="ht-dev-display")` for every
Choice task. For per-item option sets (two captions, two arguments ...) the options are
keyed A, B ... by display position and their texts are kept in `option_texts` (pool side)
so the length-only baseline can see them.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from v2.eval.sealed.schema import (
    MAX_INPUT_CHARS,
    Candidate,
    SourceSpec,
    display_order,
    input_chars,
    letters,
)

SALT = "ht-dev-display"
BODY_CHARS = 2000
SENTENCE_END = re.compile(r"[.!?][\"'\u201d\u2019)\]]*(?=\s)")


@dataclass
class HtCandidate(Candidate):
    split: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    option_texts: list[str] | None = None
    cluster_id: str = ""


def spec(
    key: str,
    dataset_id: str,
    revision: str,
    licence: str,
    evidence: str,
    label_provenance: str,
    tasks: tuple[str, ...],
    notes: str = "",
) -> SourceSpec:
    return SourceSpec(
        key=key,
        dataset_id=dataset_id,
        revision=revision,
        licence=licence,
        licence_flag=None,
        first_release="n/a (development panel; no date cutoff)",
        evidence=evidence,
        label_provenance=label_provenance,
        languages=("en",),
        tasks=tasks,
        notes=notes,
    )


def read_parquet(path: Path, columns: list[str] | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns).to_pylist()


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    csv.field_size_limit(1 << 30)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def clean(text: Any) -> str:
    return " ".join(str(text or "").split()) if text is not None else ""


def cut_at_sentence(text: str, limit: int = BODY_CHARS) -> str:
    """The first `limit` characters, cut after the last sentence end inside them."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    ends = [m.end() for m in SENTENCE_END.finditer(head + " ")]
    if ends and ends[-1] >= limit // 4:
        return head[: ends[-1]].rstrip()
    space = head.rfind(" ")
    return head[: space if space > 0 else limit].rstrip()


def level(value: float, cuts: Iterable[float]) -> int:
    """Level index for half-open bins [cut_i, cut_i+1)."""
    return sum(value >= cut for cut in cuts)


def choice(
    item_id: str, instructions: str, options: list[tuple[str, str]]
) -> dict[str, Any]:
    """A Choice question over fixed (key, description) options in display order."""
    order = display_order(item_id, len(options), SALT)
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {options[i][0]: options[i][1] for i in order},
    }


def pairwise(
    item_id: str, texts: list[str], gold_index: int
) -> tuple[list[str], list[str], str]:
    """Per-item options in display order: (display texts, keys, gold key)."""
    order = display_order(item_id, len(texts), SALT)
    keys = letters(len(texts))
    shown = [texts[i] for i in order]
    return shown, keys, keys[order.index(gold_index)]


def noul(instructions: str, criterion: str, negation: str) -> dict[str, Any]:
    return {
        "type": "noul",
        "instructions": f"{instructions} Criterion: {criterion}",
        "criteria": {"true": f"Yes: {criterion}", "false": f"No: {negation}"},
    }


def score(instructions: str, levels: list[str]) -> dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def make(
    spec_: SourceSpec,
    task: str,
    item_id: str,
    group: str,
    split: str,
    file: str,
    row: int | str,
    state: Any,
    question: dict[str, Any],
    gold: Any,
    overlap_texts: list[str] | None = None,
    option_texts: list[str] | None = None,
    cluster: str = "",
) -> HtCandidate | None:
    if input_chars(state, question) > MAX_INPUT_CHARS:
        return None
    return HtCandidate(
        source=spec_.key,
        task=task,
        source_item_id=item_id,
        group_id=group,
        balance_label=json.dumps(gold),
        language="en",
        state=state,
        question=question,
        gold=gold,
        overlap_texts=[t for t in (overlap_texts or []) if t],
        split=split,
        provenance={"file": file, "row": row},
        option_texts=option_texts,
        cluster_id=cluster or group,
    )


def present(items: Iterable[HtCandidate | None]) -> Iterator[HtCandidate]:
    return (item for item in items if item is not None)
