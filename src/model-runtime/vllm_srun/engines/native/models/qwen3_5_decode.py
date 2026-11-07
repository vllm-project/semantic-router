"""Greedy decoding of a Qwen3.5 backbone, as Transformers 5.17 ``generate`` runs it.

The prefill runs the chunked gated delta rule and keeps its final state, the last
convolution frames (left-padded with zeros when the prompt is shorter) and every
attention key and value, padding included. Positions are the attention mask's
cumsum, and a padding position is 0; the logit is the last position's, so a
shorter row is read at its padding. Each new token takes the position one past
that row's last one (``generate`` appends it; it does not recompute the cumsum)
and runs ``causal_conv1d_update`` and the recurrent gated delta rule. It attends
to the cached keys with the prompt's padding masked out, and with no mask when
the batch has none. The LM head runs in the weight's dtype and the arg-max in FP32.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
import torch.nn.functional as F

from .common import causal_mask
from .qwen3_5 import Qwen3_5Backbone, Qwen3_5Layer

LayerState = tuple[torch.Tensor, torch.Tensor]


def rotary_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Text rotary positions ``[3, batch, length]``: the mask's cumsum, padding at 0.

    The same index on every section. This is ``_prepare_position_ids_for_generation``
    with an attention mask (padding would otherwise be position -1).
    """
    positions = attention_mask.long().cumsum(dim=-1) - 1
    positions = positions.masked_fill(attention_mask == 0, 0)
    return positions.unsqueeze(0).expand(3, -1, -1)


def linear_padding(mask: torch.Tensor) -> torch.Tensor | None:
    """The prefill's linear-attention mask: none for one token or an unpadded batch."""
    if mask.shape[1] == 1 or bool(torch.all(mask == 1)):
        return None
    return mask


def decode_mask(attention_mask: torch.Tensor) -> torch.Tensor | None:
    """The one-query mask ``[batch, 1, 1, keys]``: keep the real keys, or none when nothing is padding.

    A generated token is never padding, so this only drops the prompt's right-padding.
    One query is not causal; the mask is the whole key side.
    """
    if bool(torch.all(attention_mask == 1)):
        return None
    return attention_mask.to(dtype=torch.bool)[:, None, None, :]


def _layers(
    backbone: Qwen3_5Backbone,
    hidden: torch.Tensor,
    rotary: tuple[torch.Tensor, torch.Tensor],
    full_mask: torch.Tensor | None,
    linear_mask: torch.Tensor | None,
    past: list[LayerState] | None,
) -> tuple[torch.Tensor, list[LayerState]]:
    assert backbone.kernels is not None, "bind kernels before running the backbone"
    states: list[LayerState] = []
    for index, layer in enumerate(backbone.layers):
        hidden, state = cast(Qwen3_5Layer, layer).forward_cached(
            hidden,
            rotary,
            full_mask,
            linear_mask,
            backbone.kernels,
            None if past is None else past[index],
        )
        states.append(state)
    return backbone.norm(hidden), states


def prefill(
    backbone: Qwen3_5Backbone, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, list[LayerState]]:
    """Final-norm hidden states of a right-padded causal prompt, and each layer's cache."""
    hidden = backbone.embed_tokens(input_ids)
    length = hidden.shape[1]
    rotary = backbone.rotary_emb(hidden, rotary_positions(attention_mask))
    full = causal_mask(attention_mask, length)
    return _layers(backbone, hidden, rotary, full, linear_padding(attention_mask), None)


def step(
    backbone: Qwen3_5Backbone,
    token_ids: torch.Tensor,
    past: list[LayerState],
    positions: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, list[LayerState]]:
    """One new token per row at ``positions`` ``[batch]``, from ``past``.

    ``attention_mask`` covers the cached keys and this token. Padding keys are masked.
    """
    hidden = backbone.embed_tokens(token_ids[:, None])
    rotary = backbone.rotary_emb(hidden, positions.view(1, -1, 1).expand(3, -1, 1))
    return _layers(backbone, hidden, rotary, decode_mask(attention_mask), None, past)


def pad_batch(
    rows: Sequence[Sequence[int]], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad ``rows`` with ``pad_token_id``. The mask is 1 on real tokens."""
    if not rows or any(len(row) == 0 for row in rows):
        raise ValueError("generation needs one or more non-empty token rows")
    width = max(len(row) for row in rows)
    ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.as_tensor(row, dtype=torch.long, device=device)
        mask[index, : len(row)] = 1
    return ids, mask


def logits_of(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The LM head on the last position, in the weight's dtype, as FP32 logits ``[batch, vocab]``."""
    last = hidden[:, -1].to(weight.dtype)
    return F.linear(last, weight).float()


def greedy(
    backbone: Qwen3_5Backbone,
    weight: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    pad_token_id: int,
) -> list[list[int]]:
    """Greedy new tokens for each row, not including ``eos_token_id``.

    A row that has already finished is fed ``pad_token_id``; the row that just
    emitted the end token is fed that token for one step, as ``generate`` does.
    """
    if max_new_tokens < 1:
        return [[] for _ in range(input_ids.shape[0])]
    positions = rotary_positions(attention_mask)[0]
    hidden, past = prefill(backbone, input_ids, attention_mask)
    # The next position is one past the last one, per row. A right-padded row's last
    # position is 0, so its first new token is position 1.
    cursor = positions[:, -1] + 1
    mask = attention_mask
    done = [False] * input_ids.shape[0]
    chosen: list[list[int]] = [[] for _ in done]
    for step_index in range(max_new_tokens):
        tokens = torch.argmax(logits_of(hidden, weight), dim=-1)
        already = list(done)
        for index, token in enumerate(tokens.tolist()):
            if done[index]:
                continue
            if eos_token_id is not None and token == eos_token_id:
                done[index] = True
            else:
                chosen[index].append(token)
        if all(done) or step_index == max_new_tokens - 1:
            break
        feed = tokens.clone()
        for index, flag in enumerate(already):
            if flag:
                feed[index] = pad_token_id
        mask = torch.cat(
            [
                mask,
                torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=mask.device),
            ],
            dim=1,
        )
        hidden, past = step(backbone, feed, past, cursor, mask)
        cursor = cursor + 1
    return chosen
