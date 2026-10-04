"""Reasoning-graph problems: one final System One question plus typed intermediate nodes.

A problem is a plain dict:

    {"pid", "family", "source", "render_template", "language", "licence",
     "state": text or JSON object,
     "final": {"task_type", "instructions", "options": [{"key", "description"}], "label"},
     "nodes": [{"id", "depends_on": [ids], "kind": "choice" | "noul" | "score",
                "question", "options" (choice / score), "label" (choice / score) | "answer" (noul),
                "statement" (true sentence), "false_statement" (or None), "verified": True}],
     "audit": {...}}

Program generators fill every node from the program, so ``verified`` is always true for them; teacher graphs keep
only nodes whose fresh re-ask agreed.
"""

from __future__ import annotations

import random
from typing import Any

KEY_STYLES = ("letters", "option_n", "kn", "numbers")


def option_keys(count: int, style: str) -> list[str]:
    if style == "letters":
        return [chr(65 + i) for i in range(count)]
    if style == "option_n":
        return [f"option_{i}" for i in range(count)]
    if style == "kn":
        return [f"K{i + 1}" for i in range(count)]
    if style == "numbers":
        return [str(i + 1) for i in range(count)]
    raise ValueError(style)


def choice(
    rng: random.Random, correct: str, wrong: list[str], *, style: str | None = None
) -> tuple[list[dict[str, Any]], int]:
    """Shuffled options with distinct descriptions; returns (options, gold index)."""
    seen = {correct}
    distractors = []
    for value in wrong:
        if value not in seen:
            seen.add(value)
            distractors.append(value)
    values = [correct, *distractors]
    rng.shuffle(values)
    keys = option_keys(len(values), style or rng.choice(KEY_STYLES))
    return [{"key": k, "description": v} for k, v in zip(keys, values)], values.index(
        correct
    )


def node(
    node_id: str,
    question: str,
    *,
    kind: str,
    depends_on: list[str] | None = None,
    options: list[dict[str, Any]] | None = None,
    label: int | None = None,
    answer: bool | None = None,
    statement: str,
    false_statement: str | None = None,
) -> dict[str, Any]:
    if kind == "noul":
        if answer is None:
            raise ValueError("noul node needs answer")
    elif kind in ("choice", "score"):
        if options is None or label is None:
            raise ValueError(f"{kind} node needs options and label")
    else:
        raise ValueError(kind)
    out: dict[str, Any] = {
        "id": node_id,
        "depends_on": list(depends_on or []),
        "kind": kind,
        "question": question,
        "statement": statement,
        "false_statement": false_statement,
        "verified": True,
    }
    if kind == "noul":
        out["answer"] = bool(answer)
    else:
        out["options"] = options
        out["label"] = label
    return out


def depths(nodes: list[dict[str, Any]]) -> dict[str, int]:
    """0 for nodes without dependencies, else one more than the deepest dependency."""
    by_id = {n["id"]: n for n in nodes}
    memo: dict[str, int] = {}

    def depth(node_id: str, stack: frozenset[str]) -> int:
        if node_id in memo:
            return memo[node_id]
        if node_id in stack or node_id not in by_id:
            return 0
        deps = [d for d in by_id[node_id]["depends_on"] if d in by_id]
        memo[node_id] = (
            1 + max(depth(d, stack | {node_id}) for d in deps) if deps else 0
        )
        return memo[node_id]

    return {n["id"]: depth(n["id"], frozenset()) for n in nodes}


def check_problem(problem: dict[str, Any]) -> None:
    """Structural invariants every generator must satisfy."""
    final = problem["final"]
    if final["task_type"] not in ("choice", "noul", "score"):
        raise ValueError("final task type")
    if not 0 <= final["label"] < len(final["options"]):
        raise ValueError("final label")
    descriptions = [str(o["description"]) for o in final["options"]]
    if len(set(descriptions)) != len(descriptions):
        raise ValueError(f"{problem['pid']}: duplicate final options")
    ids = set()
    for item in problem["nodes"]:
        if item["id"] in ids:
            raise ValueError("duplicate node id")
        if any(d not in ids for d in item["depends_on"]):
            raise ValueError(
                f"{problem['pid']}: node {item['id']} depends on a later or unknown node"
            )
        ids.add(item["id"])
        if item["kind"] in ("choice", "score"):
            texts = [str(o["description"]) for o in item["options"]]
            if len(set(texts)) != len(texts) or not 0 <= item["label"] < len(texts):
                raise ValueError(f"{problem['pid']}: node {item['id']} options")
