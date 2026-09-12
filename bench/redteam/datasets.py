"""Prompt sets for the red-team benchmark.

JailbreakBench ships the 100 harmful behaviors this benchmark reports against.
A local JSON or JSONL file is accepted so the benchmark can run offline.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

JAILBREAKBENCH_REPO = "JailbreakBench/JBB-Behaviors"
JAILBREAKBENCH_SUBSET = "behaviors"
GOAL_KEYS = ("goal", "prompt", "behavior", "text")


def _first_goal(record: dict[str, object]) -> str | None:
    for key in GOAL_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _iter_records(path: Path) -> Iterator[dict[str, object]]:
    if path.suffix == ".jsonl":
        for line in path.read_text().splitlines():
            if line.strip():
                yield json.loads(line)
        return
    payload = json.loads(path.read_text())
    records = payload if isinstance(payload, list) else payload.get("behaviors", [])
    for record in records:
        if isinstance(record, dict):
            yield record


def load_local(path: str | Path) -> list[str]:
    prompts = [_first_goal(record) for record in _iter_records(Path(path))]
    found = [prompt for prompt in prompts if prompt]
    if not found:
        raise ValueError(f"no prompts found in {path}; expected one of {GOAL_KEYS}")
    return found


def load_jailbreakbench() -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(JAILBREAKBENCH_REPO, JAILBREAKBENCH_SUBSET, split="harmful")
    prompts = [_first_goal(dict(record)) for record in dataset]
    return [prompt for prompt in prompts if prompt]


def load_prompts(source: str) -> list[str]:
    """``source`` is ``jailbreakbench`` or a path to a local JSON/JSONL file."""
    if source == "jailbreakbench":
        return load_jailbreakbench()
    return load_local(source)
