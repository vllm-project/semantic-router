"""Masked-language-model collators for the mmBERT 32K foundation recipe."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: a8ef9416fb4ce6e6374e5d92f5fefb4dd27221e0.

from __future__ import annotations

import torch

ANCHOR_MASK_PROBABILITY = 0.5
_TOKEN_ID_ATTRIBUTES = (
    "pad_token_id",
    "cls_token_id",
    "sep_token_id",
    "unk_token_id",
    "mask_token_id",
)


def _special_token_ids(tokenizer, *, include_mask: bool) -> set[int]:
    attributes = _TOKEN_ID_ATTRIBUTES if include_mask else _TOKEN_ID_ATTRIBUTES[:-1]
    return {
        token_id
        for attribute in attributes
        if (token_id := getattr(tokenizer, attribute, None)) is not None
    }


def _stack_inputs(examples, pad_token_id: int | None):
    input_ids = torch.stack(
        [torch.tensor(example["input_ids"]) for example in examples]
    )
    if "attention_mask" in examples[0]:
        attention_mask = torch.stack(
            [torch.tensor(example["attention_mask"]) for example in examples]
        )
    else:
        attention_mask = (input_ids != (pad_token_id or 0)).long()
    return input_ids, attention_mask


def _standard_mask(
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    probability: float,
    special_token_ids: set[int],
) -> torch.Tensor:
    probability_matrix = torch.full(labels.shape, probability)
    special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
    for special_id in special_token_ids:
        special_tokens_mask |= labels == special_id
    probability_matrix.masked_fill_(special_tokens_mask, 0.0)
    probability_matrix.masked_fill_(attention_mask == 0, 0.0)
    return torch.bernoulli(probability_matrix).bool()


def _apply_replacements(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    masked_indices: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
) -> None:
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    input_ids[indices_replaced] = mask_token_id
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]


def _first_half_tokens(seq, half_point: int, special_token_ids: set[int]) -> set[int]:
    return {
        token
        for position in range(1, half_point)
        if (token := seq[position].item()) not in special_token_ids
    }


def _mask_long_range_copies(
    seq,
    output,
    batch_index: int,
    actual_length: int,
    half_point: int,
    min_distance: int,
    retrieval_probability: float,
    special_token_ids: set[int],
) -> None:
    first_half = _first_half_tokens(seq, half_point, special_token_ids)
    start = half_point + min_distance // 2
    for position in range(start, actual_length - 1):
        if seq[position].item() not in first_half:
            continue
        if torch.rand(1).item() < retrieval_probability:
            output[batch_index, position] = True


def _mask_anchor_references(
    seq,
    output,
    batch_index: int,
    actual_length: int,
    half_point: int,
    anchor_positions: list[int],
    special_token_ids: set[int],
) -> None:
    for anchor_position in anchor_positions:
        if anchor_position >= half_point:
            continue
        anchor_token = seq[anchor_position].item()
        if anchor_token in special_token_ids:
            continue
        for later_position in range(half_point, actual_length - 1):
            if seq[later_position].item() != anchor_token:
                continue
            if torch.rand(1).item() < ANCHOR_MASK_PROBABILITY:
                output[batch_index, later_position] = True
            break


class RetrievalMaskingCollator:
    """MLM collator that additionally masks long-range copies and anchors."""

    def __init__(
        self,
        tokenizer,
        mlm_probability: float = 0.30,
        retrieval_probability: float = 0.10,
        min_distance_for_retrieval: int = 512,
        anchor_positions: list[int] | None = None,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.retrieval_probability = retrieval_probability
        self.min_distance = min_distance_for_retrieval
        self.anchor_positions = anchor_positions or [10, 50, 100, 200, 500]
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        self.special_token_ids = _special_token_ids(tokenizer, include_mask=True)

    def _retrieval_mask(self, input_ids, attention_mask):
        output = torch.zeros_like(input_ids, dtype=torch.bool)
        for batch_index, (seq, attention) in enumerate(
            zip(input_ids, attention_mask, strict=True)
        ):
            actual_length = attention.sum().item()
            if actual_length < self.min_distance * 2:
                continue
            half_point = actual_length // 2
            _mask_long_range_copies(
                seq,
                output,
                batch_index,
                actual_length,
                half_point,
                self.min_distance,
                self.retrieval_probability,
                self.special_token_ids,
            )
            _mask_anchor_references(
                seq,
                output,
                batch_index,
                actual_length,
                half_point,
                self.anchor_positions,
                self.special_token_ids,
            )
        return output

    def __call__(self, examples):
        input_ids, attention_mask = _stack_inputs(examples, self.tokenizer.pad_token_id)
        labels = input_ids.clone()
        masked_indices = _standard_mask(
            labels,
            attention_mask,
            self.mlm_probability,
            self.special_token_ids,
        )
        retrieval_masked = self._retrieval_mask(input_ids, attention_mask)
        masked_indices |= retrieval_masked
        labels[~masked_indices] = -100
        _apply_replacements(
            input_ids,
            labels,
            masked_indices,
            self.mask_token_id,
            self.vocab_size,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "_retrieval_count": retrieval_masked.sum().item(),
            "_total_masked": masked_indices.sum().item(),
        }


class StandardMLMCollator:
    """Standard MLM collator without retrieval-aware masking."""

    def __init__(self, tokenizer, mlm_probability: float = 0.30):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        self.special_token_ids = _special_token_ids(tokenizer, include_mask=False)

    def __call__(self, examples):
        input_ids, attention_mask = _stack_inputs(examples, self.tokenizer.pad_token_id)
        labels = input_ids.clone()
        masked_indices = _standard_mask(
            labels,
            attention_mask,
            self.mlm_probability,
            self.special_token_ids,
        )
        labels[~masked_indices] = -100
        _apply_replacements(
            input_ids,
            labels,
            masked_indices,
            self.mask_token_id,
            self.vocab_size,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "_retrieval_count": 0,
            "_total_masked": masked_indices.sum().item(),
        }
