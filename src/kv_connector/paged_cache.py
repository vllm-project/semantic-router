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


def _cache_layout(cache: torch.Tensor, heads: int, head_dim: int) -> str:
    """Recognize the connector's legacy view or vLLM's LBNHC layer view."""
    legacy = (
        cache.ndim in (4, 5)
        and cache.shape[1] == 2
        and cache.numel() // (cache.shape[0] * 2 * cache.shape[2]) == heads * head_dim
    )
    lbnhc = (
        cache.ndim == 4 and cache.shape[1] == heads and cache.shape[3] == 2 * head_dim
    )
    if legacy:
        return "legacy"
    if lbnhc:
        return "lbnhc"
    raise ValueError("paged-cache shape differs from mapper")


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
    layout = _cache_layout(cache, heads, head_dim)
    blocks, offsets = _slots(block_ids, num_tokens, block_size, cache.shape[0])
    block_indices, token_offsets = blocks.to(cache.device), offsets.to(cache.device)
    if layout == "lbnhc":
        values = cache[block_indices, :, token_offsets]
        return values[..., :head_dim], values[..., head_dim:]
    values = cache[block_indices, :, token_offsets].reshape(
        num_tokens, 2, heads, head_dim
    )
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
    layout = _cache_layout(cache, keys.shape[1], keys.shape[2])
    blocks, offsets = _slots(block_ids, keys.shape[0], block_size, cache.shape[0])
    if layout == "lbnhc":
        stacked = torch.cat((keys, values), dim=-1)
    else:
        stacked = torch.stack((keys, values), dim=1).reshape(
            keys.shape[0], 2, *cache.shape[3:]
        )
    cache[blocks.to(cache.device), :, offsets.to(cache.device)] = stacked.to(
        cache.device
    )


def validate_prefix_destination(
    cache: torch.Tensor,
    block_ids: list[int],
    num_tokens: int,
    *,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> None:
    """Check a destination before any mapped layer is written."""
    if cache.dtype != dtype:
        raise ValueError("mapped KV dtype differs from paged cache")
    block_size = cache.shape[2]
    _cache_layout(cache, heads, head_dim)
    _slots(block_ids, num_tokens, block_size, cache.shape[0])
