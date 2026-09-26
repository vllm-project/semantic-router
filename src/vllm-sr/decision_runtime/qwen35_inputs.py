"""Tokenizer-side preparation for Qwen3.5 Decision rows.

No tensor framework is imported here.  Complete rows are encoded before queue
admission, so an oversized or malformed request cannot poison unrelated
callers that later share a physical batch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from .model_inputs import ModelInput, canonical_json, qwen_segments


class QwenTokenizer(Protocol):
    """Small tokenizer seam used by deterministic preparation tests."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def __call__(
        self, texts: list[str], **kwargs: Any
    ) -> dict[str, list[list[int]]]: ...


@dataclass(frozen=True, slots=True)
class EncodedQwenRow:
    """One complete pointer-v2 row ready for tensor collation."""

    question_id: str
    type: str
    input_ids: tuple[int, ...]
    candidate_positions: tuple[int, ...]
    query_position: int
    prompt_sha256: str

    @property
    def input_tokens(self) -> int:
        return len(self.input_ids)


def encode_qwen_rows(
    rows: tuple[ModelInput, ...],
    tokenizer: QwenTokenizer,
    *,
    max_length: int,
    max_cached_characters: int = 8_000_000,
    segment_batch_size: int = 64,
) -> tuple[EncodedQwenRow, ...]:
    """Encode exact segments once per request without truncating any row."""

    if not rows:
        raise ValueError("at least one Qwen row is required")
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length < 1
    ):
        raise ValueError("max_length must be a positive integer")
    if max_cached_characters < 0 or segment_batch_size < 1:
        raise ValueError("invalid tokenizer preparation resource bound")

    segmented = tuple(qwen_segments(row) for row in rows)
    unique: dict[str, tuple[int, ...] | None] = {}
    characters = 0
    for item in segmented:
        for segment in (item.prefix, *item.options, item.suffix):
            if segment not in unique:
                unique[segment] = None
                characters += len(segment)

    if characters <= max_cached_characters:
        strings = list(unique)
        for start in range(0, len(strings), segment_batch_size):
            batch = strings[start : start + segment_batch_size]
            tokenized = tokenizer(
                batch,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            ).get("input_ids")
            if not isinstance(tokenized, list) or len(tokenized) != len(batch):
                raise ValueError("batch tokenizer output count differs")
            for segment, token_ids in zip(batch, tokenized, strict=True):
                unique[segment] = _token_ids(token_ids)
    else:
        unique = {}

    return tuple(
        _encode_one(row, segments, tokenizer, unique, max_length=max_length)
        for row, segments in zip(rows, segmented, strict=True)
    )


def _encode_one(
    row: ModelInput,
    segments,
    tokenizer: QwenTokenizer,
    cached: dict[str, tuple[int, ...] | None],
    *,
    max_length: int,
) -> EncodedQwenRow:
    input_ids = list(_encode_segment(segments.prefix, tokenizer, cached))
    candidate_positions = []
    for option in segments.options:
        part = _encode_segment(option, tokenizer, cached)
        if not part:
            raise ValueError("empty tokenized candidate")
        input_ids.extend(part)
        candidate_positions.append(len(input_ids) - 1)
    suffix = _encode_segment(segments.suffix, tokenizer, cached)
    if not suffix:
        raise ValueError("empty tokenized Decision suffix")
    input_ids.extend(suffix)
    if len(input_ids) > max_length:
        raise ValueError(
            f"{row.question_id}: {len(input_ids)} tokens exceeds "
            f"max_length={max_length}; no truncation allowed"
        )
    query_position = len(input_ids) - 1
    if not all(0 <= position < query_position for position in candidate_positions):
        raise ValueError("candidate endpoints must precede the global query")
    if len(set(candidate_positions)) != len(candidate_positions):
        raise ValueError("duplicate candidate endpoint")
    return EncodedQwenRow(
        question_id=row.question_id,
        type=row.type,
        input_ids=tuple(input_ids),
        candidate_positions=tuple(candidate_positions),
        query_position=query_position,
        prompt_sha256=hashlib.sha256(segments.rendered.encode()).hexdigest(),
    )


def _encode_segment(
    segment: str,
    tokenizer: QwenTokenizer,
    cached: dict[str, tuple[int, ...] | None],
) -> tuple[int, ...]:
    if segment in cached:
        token_ids = cached[segment]
        if token_ids is None:  # pragma: no cover - internal invariant
            raise RuntimeError("tokenizer segment cache is incomplete")
        return token_ids
    return _token_ids(tokenizer.encode(segment, add_special_tokens=False))


def _token_ids(value: object) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        raise ValueError("tokenizer returned invalid token IDs")
    return tuple(value)


def token_ids_sha256(row: EncodedQwenRow) -> str:
    """Return the released canonical token-ID receipt for parity evidence."""

    return hashlib.sha256(canonical_json(list(row.input_ids)).encode()).hexdigest()
