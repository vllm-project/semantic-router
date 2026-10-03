"""Validated mapping and paged-cache injection for a complete source prefix."""

from __future__ import annotations

import torch

from src.kv_connector.paged_cache import inject_prefix, validate_prefix_destination
from src.kv_connector.runtime import MapperArtifact
from src.kv_connector.snapshot import SourceSnapshot
from src.kv_connector.transform import map_source_cache

_MIN_CACHE_RANK = 3


def apply_handoff(
    artifact: MapperArtifact,
    snapshot: SourceSnapshot,
    *,
    namespace: str,
    cache_id: str,
    mapper_id: str,
    target_prompt_ids: list[int],
    target_caches: dict[int, torch.Tensor],
    block_ids: list[int],
    source_rope_theta: float,
    target_rope_theta: float,
    cache_layout: str | None = None,
) -> int:
    """Map and inject KV, returning the token count eligible as an external hit.

    Callers must let vLLM recompute when this raises. Every destination is
    checked before writing so ordinary validation failures leave caches alone.
    """
    compat = artifact.manifest.compatibility
    if snapshot.namespace != namespace or snapshot.cache_id != cache_id:
        raise ValueError("source snapshot scope or cache ID mismatch")
    if mapper_id != artifact.manifest.mapper_id:
        raise ValueError("mapper ID mismatch")
    if (
        snapshot.source_model != compat.source_model
        or snapshot.source_revision != compat.source_revision
    ):
        raise ValueError("source model or revision mismatch")
    count = len(snapshot.token_ids)
    if (
        count >= len(target_prompt_ids)
        or tuple(target_prompt_ids[:count]) != snapshot.token_ids
    ):
        raise ValueError("target prompt does not extend the source token prefix")
    if set(target_caches) != set(
        range(len(artifact.manifest.source_layers_per_target["k"]))
    ):
        raise ValueError("target cache layers differ from mapper")
    source_shape = next(iter(snapshot.layers.values()))[0].shape
    if source_shape[1:] != (compat.num_kv_heads, compat.head_dim):
        raise ValueError("source KV shape differs from mapper")
    first_cache = next(iter(target_caches.values()))
    dtype = first_cache.dtype
    if first_cache.ndim < _MIN_CACHE_RANK or first_cache.shape[2] <= 0:
        raise ValueError("target cache has no valid block dimension")
    if count % first_cache.shape[2]:
        raise ValueError("source prefix must end on a target cache block boundary")
    if any(cache.device != first_cache.device for cache in target_caches.values()):
        raise ValueError("target cache layers must be on one device")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("target cache dtype is unsupported")
    if (dtype == torch.bfloat16 and compat.precision != "bf16") or (
        dtype == torch.float16 and compat.precision != "fp16"
    ):
        raise ValueError("target cache precision differs from mapper")
    if snapshot.layers[0][0].dtype != dtype:
        raise ValueError("source cache precision differs from mapper")
    mapped = map_source_cache(
        artifact,
        snapshot.layers,
        torch.arange(count),
        source_rope_theta=source_rope_theta,
        target_rope_theta=target_rope_theta,
        device=first_cache.device,
        dtype=dtype,
    )
    for cache in target_caches.values():
        validate_prefix_destination(
            cache,
            block_ids,
            count,
            heads=compat.num_kv_heads,
            head_dim=compat.head_dim,
            dtype=dtype,
            layout=cache_layout,
        )
    for layer, (keys, values) in mapped.items():
        inject_prefix(
            target_caches[layer], block_ids, keys, values, layout=cache_layout
        )
    return count
