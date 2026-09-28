"""Disaggregated state/candidate texts for separately encoded CLM-style arms.

The joint arms use the unchanged native prompt from
``training.model.decision_model``. The disaggregated arms encode the state text
(context followed by the question) once and every candidate text on its own,
so a candidate vector depends only on its text and can be cached across
requests. Structured values are rendered as indented ``key: value`` prose
lines rather than JSON.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

DISAGGREGATED_RENDER_VERSION = "decision2-9b-disaggregated-prose-v1"


def to_text(value: Any, indent: int = 0) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    pad = " " * indent
    if isinstance(value, dict):
        parts = []
        for key, child in value.items():
            if isinstance(child, (dict, list)) and child:
                parts.append(f"{pad}{key}:\n{to_text(child, indent + 2)}")
            else:
                parts.append(f"{pad}{key}: {to_text(child)}")
        return ("\n\n" if indent == 0 else "\n").join(parts)
    if isinstance(value, (list, tuple)):
        parts = []
        for child in value:
            if isinstance(child, (dict, list)) and child:
                parts.append(f"{pad}-\n{to_text(child, indent + 2)}")
            else:
                parts.append(f"{pad}- {to_text(child)}")
        return "\n".join(parts)
    raise TypeError(f"Unsupported payload value {type(value).__name__}")


def state_text(row: dict[str, Any]) -> str:
    context = to_text(row["state"]).strip()
    question = to_text(row["instructions"]).strip()
    if context and question:
        return f"{context}\n\n{question}"
    text = context or question
    if not text:
        raise ValueError(f"{row.get('id')}: empty disaggregated state text")
    return text


def candidate_texts(row: dict[str, Any]) -> list[str]:
    """One text per offered option, in the row's option order."""
    kind = row["task_type"]
    texts = []
    question = to_text(row["instructions"]).strip()
    for option in row["options"]:
        key, description = option["key"], option.get("description")
        empty = description is None or description == "" or description == {}
        if kind == "noul":
            if empty:
                description = (
                    f"Yes. This is true: {question}"
                    if key == "true"
                    else f"No. This is false: {question}"
                )
            texts.append(f"{key}: {to_text(description)}")
        elif kind == "choice":
            texts.append(key if empty else to_text(description))
        elif kind == "score":
            texts.append(f"Level {key}" if empty else to_text(description))
        else:
            raise ValueError(f"{row.get('id')}: unsupported task type {kind}")
    if any(not text.strip() for text in texts):
        raise ValueError(f"{row.get('id')}: empty candidate text")
    return texts


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def score_level_order(row: dict[str, Any]) -> list[int]:
    """Option positions sorted by ordinal level for a Score row."""
    if row["task_type"] != "score":
        raise ValueError("Only Score rows have ordinal levels")
    levels = [int(option["key"]) for option in row["options"]]
    if sorted(levels) != list(range(len(levels))):
        raise ValueError(f"{row.get('id')}: Score keys must enumerate 0..K-1")
    return sorted(range(len(levels)), key=lambda index: levels[index])
