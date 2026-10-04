"""System One questions to Decision 2.0 model inputs, exactly as the packages were scored.

Prompt ``decision2-segmented-options-global-query-v1``: a context and question
prefix, one ``<option>`` segment per option, and a fixed query suffix. Each
segment is tokenized on its own without special tokens; the last token of an
option segment is its endpoint and the last token overall is the query
position. Inputs longer than the model limit are rejected, never truncated.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...errors import INVALID_QUESTION, MAX_LENGTH_EXCEEDED, QuestionError
from .package import MAX_OPTIONS, MIN_OPTIONS

QUESTION_TYPES = ("choice", "noul", "score")
MIN_LEVELS, MAX_LEVELS = 2, 10
SUFFIX = "\n\nSelect the single option best supported by the context and instructions.\nDecision:"


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_payload(
    value: Any, *, nullable: bool = False, require_nonempty_text: bool = False
) -> bool:
    """Text, an object or an array of JSON values (Choice descriptions may be null)."""
    if value is None:
        return nullable
    if not isinstance(value, (str, dict, list)) or (
        require_nonempty_text and value == ""
    ):
        return False

    def json_value(part: Any) -> bool:
        if part is None or isinstance(part, (str, bool, int)):
            return True
        if isinstance(part, float):
            return math.isfinite(part)
        if isinstance(part, list):
            return all(json_value(child) for child in part)
        if isinstance(part, dict):
            return all(
                isinstance(key, str) and json_value(child)
                for key, child in part.items()
            )
        return False

    try:
        return json_value(value)
    except RecursionError:
        return False


def valid_state(state: Any) -> bool:
    if not json_payload(state):
        return False
    try:
        canonical(state)
    except (TypeError, ValueError):
        return False
    return True


def question_options(question: Any) -> tuple[str, Any, list[dict[str, Any]]]:
    """Validate one question; its type, instructions and ordered options.

    Accepts System One ``criteria`` and the superset fields ``choices``
    (ordered Choice / Noul options) and ``levels`` (ordered Score levels).
    """
    if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
        raise QuestionError(INVALID_QUESTION, "unsupported or malformed question type")
    kind = question["type"]
    instructions = question.get("instructions")
    if not json_payload(instructions, require_nonempty_text=True):
        raise QuestionError(INVALID_QUESTION, "missing question instructions")
    criteria = question.get("criteria")
    choices = question.get("choices")
    levels = question.get("levels")
    if kind == "score":
        if choices is not None:
            raise QuestionError(
                INVALID_QUESTION, "score questions take criteria or levels, not choices"
            )
        if levels is not None:
            if criteria is not None:
                raise QuestionError(
                    INVALID_QUESTION, "use criteria or levels, not both"
                )
            criteria = levels
        if (
            not isinstance(criteria, list)
            or not MIN_LEVELS <= len(criteria) <= MAX_LEVELS
        ):
            raise QuestionError(
                INVALID_QUESTION,
                "score criteria must be an ordered list of 2..10 levels",
            )
        if any(not json_payload(description) for description in criteria):
            raise QuestionError(
                INVALID_QUESTION,
                "score criteria descriptions must be text or structured data",
            )
        return (
            kind,
            instructions,
            [
                {"key": str(index), "description": description}
                for index, description in enumerate(criteria)
            ],
        )
    if levels is not None:
        raise QuestionError(INVALID_QUESTION, "levels apply only to score questions")
    if choices is not None:
        if criteria is not None:
            raise QuestionError(INVALID_QUESTION, "use criteria or choices, not both")
        criteria = _choices_to_criteria(choices)
    if kind == "noul":
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
            raise QuestionError(
                INVALID_QUESTION, "noul requires only false and true criteria"
            )
        if len(criteria) < MIN_OPTIONS:
            criteria = {
                "false": criteria.get("false", "No"),
                "true": criteria.get("true", "Yes"),
            }
    if (
        not isinstance(criteria, dict)
        or not MIN_OPTIONS <= len(criteria) <= MAX_OPTIONS
    ):
        raise QuestionError(
            INVALID_QUESTION,
            "choice/noul criteria must be an object with 2..255 options",
        )
    if any(
        not isinstance(key, str)
        or not key
        or not json_payload(description, nullable=kind == "choice")
        for key, description in criteria.items()
    ):
        raise QuestionError(
            INVALID_QUESTION,
            "choice/noul criteria need nonempty string keys and valid descriptions",
        )
    if kind == "noul" and set(criteria) != {"false", "true"}:
        raise QuestionError(INVALID_QUESTION, "noul requires false and true criteria")
    return (
        kind,
        instructions,
        [
            {"key": key, "description": description}
            for key, description in criteria.items()
        ],
    )


def _choices_to_criteria(choices: Any) -> dict[str, Any]:
    if not isinstance(choices, list):
        raise QuestionError(
            INVALID_QUESTION, "choices must be a list of {key, description}"
        )
    criteria: dict[str, Any] = {}
    for choice in choices:
        if (
            not isinstance(choice, dict)
            or set(choice) - {"key", "description"}
            or "key" not in choice
        ):
            raise QuestionError(INVALID_QUESTION, "each choice is {key, description}")
        key = choice["key"]
        if not isinstance(key, str) or not key or key in criteria:
            raise QuestionError(
                INVALID_QUESTION, "choice keys must be unique nonempty strings"
            )
        criteria[key] = choice.get("description")
    return criteria


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


class Tokenizer:
    """The package tokenizer through the ``tokenizers`` library (no special tokens added)."""

    def __init__(self, backend: Any, pad_id: int):
        self.backend = backend
        self.pad_id = pad_id

    @classmethod
    def from_package(cls, root: Path) -> Tokenizer:
        from tokenizers import Tokenizer as Backend

        backend = Backend.from_file(str(root / "tokenizer.json"))
        config_path = root / "tokenizer_config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        pad_id = None
        for name in ("pad_token", "eos_token"):
            token = config.get(name)
            if isinstance(token, dict):
                token = token.get("content")
            if isinstance(token, str):
                pad_id = backend.token_to_id(token)
                if pad_id is not None:
                    break
        if pad_id is None:
            raise ValueError("the tokenizer needs a pad or EOS token")
        return cls(backend, pad_id)

    def encode(self, text: str) -> list[int]:
        return list(self.backend.encode(text, add_special_tokens=False).ids)


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


def collate(items: list[Any], pad_id: int) -> dict[str, Any]:
    """Pad rows to a multiple of 8 and option endpoints to the widest row (the scored layout)."""
    import torch

    if not items:
        raise ValueError("cannot collate an empty batch")
    length = math.ceil(max(len(item.ids) for item in items) / 8) * 8
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
