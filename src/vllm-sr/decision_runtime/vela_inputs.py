"""Tokenizer-side preparation for Vela Decision marker rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .model_inputs import ModelInput, VelaTextInput, vela_text_input

MIN_VELA_INPUT_TOKENS = 8


class VelaTokenizer(Protocol):
    cls_token_id: int | None
    bos_token_id: int | None
    sep_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    mask_token_id: int | None

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, list[int]]: ...


@dataclass(frozen=True, slots=True)
class VelaSpecialTokens:
    bos: int
    sep: int
    pad: int
    marker: int


@dataclass(frozen=True, slots=True)
class EncodedVelaRow:
    """One complete Vela marker row ready for tensor collation."""

    question_id: str
    type: str
    input_ids: tuple[int, ...]
    marker_positions: tuple[int, ...]
    candidate_ids: tuple[str, ...]
    state_tokens: int

    @property
    def input_tokens(self) -> int:
        return len(self.input_ids)


def vela_special_tokens(tokenizer: VelaTokenizer) -> VelaSpecialTokens:
    bos = (
        tokenizer.cls_token_id
        if tokenizer.cls_token_id is not None
        else tokenizer.bos_token_id
    )
    sep = (
        tokenizer.sep_token_id
        if tokenizer.sep_token_id is not None
        else tokenizer.eos_token_id
    )
    values = (bos, sep, tokenizer.pad_token_id, tokenizer.mask_token_id)
    if any(value is None for value in values):
        raise ValueError(
            "Vela tokenizer must provide BOS/CLS, SEP/EOS, PAD, and MASK IDs"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError("Vela tokenizer special token IDs are invalid")
    return VelaSpecialTokens(
        bos=values[0],  # type: ignore[arg-type]
        sep=values[1],  # type: ignore[arg-type]
        pad=values[2],  # type: ignore[arg-type]
        marker=values[3],  # type: ignore[arg-type]
    )


def encode_vela_rows(
    rows: tuple[ModelInput, ...],
    tokenizer: VelaTokenizer,
    *,
    max_length: int,
    max_cached_characters: int = 8_000_000,
) -> tuple[EncodedVelaRow, ...]:
    """Encode complete rows, sharing exact repeated spans within one request."""

    if not rows:
        raise ValueError("at least one Vela row is required")
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length < MIN_VELA_INPUT_TOKENS
    ):
        raise ValueError("max_length must be an integer of at least eight")
    if type(max_cached_characters) is not int or max_cached_characters < 0:
        raise ValueError("max_cached_characters must be non-negative")
    special = vela_special_tokens(tokenizer)
    rendered = tuple(vela_text_input(row) for row in rows)
    unique: dict[str, tuple[int, ...]] = {}
    characters = 0
    for item in rendered:
        for span in (item.question, *item.candidates, item.state):
            if span not in unique:
                unique[span] = ()
                characters += len(span)
    if characters <= max_cached_characters:
        for span in unique:
            unique[span] = _tokens(tokenizer, span)
    else:
        unique = {}
    return tuple(
        _encode_one(row, item, tokenizer, special, unique, max_length=max_length)
        for row, item in zip(rows, rendered, strict=True)
    )


def _encode_one(
    row: ModelInput,
    rendered: VelaTextInput,
    tokenizer: VelaTokenizer,
    special: VelaSpecialTokens,
    cached: dict[str, tuple[int, ...]],
    *,
    max_length: int,
) -> EncodedVelaRow:
    question = _cached_tokens(tokenizer, rendered.question, cached)
    input_ids = [special.bos, *question, special.sep]
    marker_positions = []
    for candidate in rendered.candidates:
        description = _cached_tokens(tokenizer, candidate, cached)
        marker_positions.append(len(input_ids))
        input_ids.extend((special.marker, *description, special.sep))
    state = _cached_tokens(tokenizer, rendered.state, cached)
    room = max_length - len(input_ids) - 1
    if room < 1:
        raise ValueError("complete question and candidates leave no room for state")
    if len(state) > room:
        raise ValueError(
            f"{row.question_id}: complete input exceeds max_length={max_length}; "
            "no truncation allowed"
        )
    input_ids.extend((*state, special.sep))
    return EncodedVelaRow(
        question_id=row.question_id,
        type=row.type,
        input_ids=tuple(input_ids),
        marker_positions=tuple(marker_positions),
        candidate_ids=tuple(candidate.key for candidate in row.candidates),
        state_tokens=len(state),
    )


def _cached_tokens(
    tokenizer: VelaTokenizer, text: str, cached: dict[str, tuple[int, ...]]
) -> tuple[int, ...]:
    return cached[text] if text in cached else _tokens(tokenizer, text)


def _tokens(tokenizer: VelaTokenizer, text: str) -> tuple[int, ...]:
    value = tokenizer(text, add_special_tokens=False, truncation=False).get("input_ids")
    if not isinstance(value, list) or not value:
        raise ValueError("empty or invalid Vela token span")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        raise ValueError("Vela tokenizer returned invalid token IDs")
    return tuple(value)
