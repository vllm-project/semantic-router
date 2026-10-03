"""Move token-ordered K/V to and from vLLM's standardized paged cache view."""

from __future__ import annotations

import torch

_KV_TENSOR_RANK = 3


def _slots(
    block_ids: list[int], num_tokens: int, block_size: int, num_blocks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_size <= 0 or num_tokens < 0:
        raise ValueError("invalid block size or token count")
    required = (num_tokens + block_size - 1) // block_size
    if len(block_ids) < required:
        raise ValueError("not enough blocks for token prefix")
    selected = block_ids[:required]
    if len(set(selected)) != len(selected) or any(
        block < 0 or block >= num_blocks for block in selected
    ):
        raise ValueError("invalid or duplicate cache block id")
    positions = torch.arange(num_tokens)
    ids = torch.tensor(selected, dtype=torch.long)
    return ids[positions // block_size], positions % block_size


def _check_cache(
    cache: torch.Tensor, block_size: int, heads: int, head_dim: int
) -> None:
    if cache.ndim not in (4, 5) or cache.shape[1:3] != (2, block_size):
        raise ValueError("expected a standardized [blocks, 2, block_size, KV] cache")
    if cache.numel() // (cache.shape[0] * 2 * block_size) != heads * head_dim:
        raise ValueError("paged-cache KV width differs from mapper")


def extract_prefix(
    cache: torch.Tensor,
    block_ids: list[int],
    num_tokens: int,
    *,
    heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return K and V as [tokens, heads, head_dim]."""
    block_size = cache.shape[2]
    _check_cache(cache, block_size, heads, head_dim)
    blocks, offsets = _slots(block_ids, num_tokens, block_size, cache.shape[0])
    values = cache[blocks.to(cache.device), :, offsets.to(cache.device)]
    values = values.reshape(num_tokens, 2, heads, head_dim)
    return values[:, 0], values[:, 1]


def inject_prefix(
    cache: torch.Tensor,
    block_ids: list[int],
    keys: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Write mapped K/V into allocated cache blocks after all checks pass."""
    if keys.shape != values.shape or keys.ndim != _KV_TENSOR_RANK:
        raise ValueError("K and V must have equal [tokens, heads, head_dim] shapes")
    if keys.dtype != cache.dtype or values.dtype != cache.dtype:
        raise ValueError("mapped KV dtype differs from paged cache")
    block_size = cache.shape[2]
    _check_cache(cache, block_size, keys.shape[1], keys.shape[2])
    blocks, offsets = _slots(block_ids, keys.shape[0], block_size, cache.shape[0])
    stacked = torch.stack((keys, values), dim=1).reshape(
        keys.shape[0], 2, *cache.shape[3:]
    )
    cache[blocks.to(cache.device), :, offsets.to(cache.device)] = stacked.to(
        cache.device
    )
