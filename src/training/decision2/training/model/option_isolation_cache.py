"""Technical prefix-cache proof for independent System One candidates.

Each branch receives the same state/question token prefix and only its own
candidate tail.  This is an inference feasibility probe, not a trained
Decision 2.0 architecture or a release adapter.  In particular, its hidden
vectors have no learned decision head or calibration.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import torch

from .option_isolation import candidate_prompts


def branch_tokens(
    row: dict[str, Any], tokenizer: Any
) -> tuple[list[int], list[list[int]]]:
    """Return one exact common prefix and nonempty independent tails."""
    prompts = candidate_prompts(row)
    prefix = tokenizer.encode(prompts[0].prefix, add_special_tokens=False)
    tails = [
        tokenizer.encode(prompt.tail, add_special_tokens=False) for prompt in prompts
    ]
    if not prefix or any(not tail for tail in tails):
        raise ValueError("Independent candidate tokens cannot be empty")
    return prefix, tails


def _padded_batch(
    sequences: Sequence[Sequence[int]],
    *,
    pad_id: int,
    device: torch.device,
    prefix_length: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Every candidate sequence must be nonempty")
    lengths = torch.tensor([len(sequence) for sequence in sequences], device=device)
    width = int(lengths.max().item())
    tokens = torch.full(
        (len(sequences), width), pad_id, dtype=torch.long, device=device
    )
    mask = torch.zeros(
        (len(sequences), prefix_length + width), dtype=torch.long, device=device
    )
    mask[:, :prefix_length] = 1
    for index, sequence in enumerate(sequences):
        tokens[index, : len(sequence)] = torch.tensor(
            sequence, dtype=torch.long, device=device
        )
        mask[index, prefix_length : prefix_length + len(sequence)] = 1
    return tokens, mask, lengths


def independent_endpoints(
    backbone: Any,
    prefix: Sequence[int],
    tails: Sequence[Sequence[int]],
    *,
    pad_id: int,
    device: torch.device,
    chunk_size: int,
    max_length: int,
    reuse_prefix: bool,
) -> torch.Tensor:
    """Collect one final hidden vector per branch with or without KV reuse.

    The caller must set ``backbone.eval()`` and inference/autocast contexts.
    Right padding is masked and each final vector is gathered at its own
    final unpadded token. A fresh copy of the prefix cache is used per chunk,
    so branches in different chunks cannot share mutable tail state.
    """
    if chunk_size < 1 or max_length < 1 or not prefix or not tails:
        raise ValueError("Invalid prefix, candidate set or resource limits")
    if any(not tail or len(prefix) + len(tail) > max_length for tail in tails):
        raise ValueError("Independent candidate exceeds the native length cap")
    if backbone.training:
        raise ValueError("The cache proof requires an eval-mode backbone")
    vectors: list[torch.Tensor] = []
    prefix_length = len(prefix)
    prefix_cache = None
    if reuse_prefix:
        prefix_ids = torch.tensor([list(prefix)], dtype=torch.long, device=device)
        prefix_output = backbone(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            use_cache=True,
            return_dict=True,
        )
        prefix_cache = prefix_output.past_key_values
        if prefix_cache is None or not hasattr(prefix_cache, "batch_repeat_interleave"):
            raise RuntimeError("Backbone has no branchable dynamic KV cache")
    for start in range(0, len(tails), chunk_size):
        chunk = tails[start : start + chunk_size]
        if reuse_prefix:
            tokens, attention_mask, lengths = _padded_batch(
                chunk, pad_id=pad_id, device=device, prefix_length=prefix_length
            )
            branch_cache = copy.deepcopy(prefix_cache)
            branch_cache.batch_repeat_interleave(len(chunk))
            outputs = backbone(
                input_ids=tokens,
                attention_mask=attention_mask,
                past_key_values=branch_cache,
                cache_position=torch.arange(
                    prefix_length,
                    prefix_length + tokens.shape[1],
                    dtype=torch.long,
                    device=device,
                ),
                use_cache=True,
                return_dict=True,
            )
        else:
            complete = [list(prefix) + list(tail) for tail in chunk]
            tokens, attention_mask, lengths = _padded_batch(
                complete, pad_id=pad_id, device=device
            )
            outputs = backbone(
                input_ids=tokens,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        selected = outputs.last_hidden_state[
            torch.arange(len(chunk), device=device), lengths - 1
        ]
        vectors.append(selected.float().cpu())
        del outputs
    return torch.cat(vectors, dim=0)
