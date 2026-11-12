"""The segmented candidate-endpoint prompt of the decision decoders, exactly as their packages were scored.

Decision 2.0 (``decision2-segmented-options-global-query-v1``) and the
Decision 1.0 decoders (``structured-segmented-candidate-endpoints-global-query-v2``)
render the same prompt: a context and question prefix, one ``<option>``
segment per option, and a fixed query suffix. Each segment is tokenized on its
own without special tokens; the last token of an option segment is its
endpoint and the last token overall is the query position. Inputs longer than
the model limit are rejected, never truncated.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from ..errors import INVALID_QUESTION, MAX_LENGTH_EXCEEDED, QuestionError
from ..systemone import MAX_OPTIONS, MIN_OPTIONS, canonical

SUFFIX = "\n\nSelect the single option best supported by the context and instructions.\nDecision:"


def segments(
    state: Any, kind: str, instructions: Any, options: list[dict[str, Any]]
) -> tuple[str, list[str], str]:
    def payload(value: Any) -> str:
        return value if isinstance(value, str) else canonical(value)

    prefix = f"Context:\n{payload(state)}\n\nTask type: {kind}\nQuestion:\n{payload(instructions)}\nOptions:"
    rendered = [
        "\n<option>\n"
        + canonical({"key": option["key"], "description": option["description"]})
        + "\n</option>"
        for option in options
    ]
    return prefix, rendered, SUFFIX


def encode(
    question_id: str,
    state: Any,
    kind: str,
    instructions: Any,
    options: list[dict[str, Any]],
    encode_text: Callable[[str], list[int]],
    max_length: int,
) -> dict[str, Any]:
    prefix, rendered, suffix = segments(state, kind, instructions, options)
    ids = encode_text(prefix)
    endpoints = []
    for option in rendered:
        part = encode_text(option)
        if not part:
            raise QuestionError(
                INVALID_QUESTION, f"{question_id}: empty tokenized option"
            )
        ids.extend(part)
        endpoints.append(len(ids) - 1)
    tail = encode_text(suffix)
    if not tail:
        raise QuestionError(INVALID_QUESTION, f"{question_id}: empty tokenized query")
    ids.extend(tail)
    if len(ids) > max_length:
        raise QuestionError(
            MAX_LENGTH_EXCEEDED,
            f"{question_id}: {len(ids)} tokens exceeds max_length={max_length}; no truncation",
        )
    return {"ids": ids, "endpoints": endpoints, "query": len(ids) - 1}


def collate(items: list[Any], pad_id: int, multiple: int = 8) -> dict[str, Any]:
    """Pad rows to a multiple of ``multiple`` tokens and option endpoints to the widest row (the scored layout)."""
    import torch

    if not items:
        raise ValueError("cannot collate an empty batch")
    length = math.ceil(max(len(item.ids) for item in items) / multiple) * multiple
    width = max(len(item.keys) for item in items)
    input_ids = torch.full((len(items), length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    positions = torch.zeros((len(items), width), dtype=torch.long)
    candidate_mask = torch.zeros((len(items), width), dtype=torch.bool)
    for index, item in enumerate(items):
        count = len(item.keys)
        if not MIN_OPTIONS <= count <= MAX_OPTIONS or len(item.gather) != count:
            raise ValueError(f"{item.question_id}: candidate count mismatch")
        if len(set(item.gather)) != count or not all(
            0 <= p < item.query < len(item.ids) for p in item.gather
        ):
            raise ValueError(f"{item.question_id}: invalid option endpoints")
        input_ids[index, : len(item.ids)] = torch.tensor(item.ids, dtype=torch.long)
        attention_mask[index, : len(item.ids)] = 1
        positions[index, :count] = torch.tensor(item.gather, dtype=torch.long)
        candidate_mask[index, :count] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "candidate_positions": positions,
        "candidate_mask": candidate_mask,
        "query_positions": torch.tensor(
            [item.query for item in items], dtype=torch.long
        ),
    }
