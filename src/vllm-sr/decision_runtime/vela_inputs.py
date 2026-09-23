"""Tokenizer-side preparation for Vela Decision marker rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .model_inputs import ModelInput, vela_text_input


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
) -> tuple[EncodedVelaRow, ...]:
    """Encode complete rows before scheduling; implicit truncation is forbidden."""

    if not rows:
        raise ValueError("at least one Vela row is required")
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length < 8
    ):
        raise ValueError("max_length must be an integer of at least eight")
    special = vela_special_tokens(tokenizer)
    return tuple(
        _encode_one(row, tokenizer, special, max_length=max_length) for row in rows
    )


def _encode_one(
    row: ModelInput,
    tokenizer: VelaTokenizer,
    special: VelaSpecialTokens,
    *,
    max_length: int,
) -> EncodedVelaRow:
    rendered = vela_text_input(row)
    question = _tokens(tokenizer, rendered.question)
    input_ids = [special.bos, *question, special.sep]
    marker_positions = []
    for candidate in rendered.candidates:
        description = _tokens(tokenizer, candidate)
        marker_positions.append(len(input_ids))
        input_ids.extend((special.marker, *description, special.sep))
    state = _tokens(tokenizer, rendered.state)
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
