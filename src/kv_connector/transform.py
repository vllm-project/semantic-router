"""Qwen3 source-cache conversion using the fitted pre-RoPE mapper."""

from __future__ import annotations

import torch

from src.kv_connector.runtime import MapperArtifact

_KV_TENSOR_RANK = 3


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    half = values.shape[-1] // 2
    return torch.cat((-values[..., half:], values[..., :half]), dim=-1)


def qwen3_rope(
    keys: torch.Tensor,
    positions: torch.Tensor,
    *,
    theta: float,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply or undo unscaled Qwen3 RoPE on [tokens, KV heads, head dim]."""
    if keys.ndim != _KV_TENSOR_RANK or keys.shape[-1] % 2:
        raise ValueError("keys must be [tokens, heads, even head_dim]")
    if positions.ndim != 1 or positions.shape[0] != keys.shape[0]:
        raise ValueError("positions must have one entry per token")
    if theta <= 0:
        raise ValueError("RoPE theta must be positive")
    dim = keys.shape[-1]
    indices = torch.arange(0, dim, 2, device=keys.device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (indices / dim))
    angles = (
        positions.to(device=keys.device, dtype=torch.float32)[:, None] * inv_freq[None]
    )
    embeddings = torch.cat((angles, angles), dim=-1)[:, None, :]
    cosine, sine = embeddings.cos(), embeddings.sin()
    if inverse:
        sine = -sine
    values = keys.float()
    return (values * cosine + _rotate_half(values) * sine).to(keys.dtype)


def map_source_cache(
    artifact: MapperArtifact,
    source: dict[int, tuple[torch.Tensor, torch.Tensor]],
    positions: torch.Tensor,
    *,
    source_rope_theta: float,
    target_rope_theta: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Map a complete source prefix to target post-RoPE K/V tensors.

    Source tensors are the post-RoPE K/V stored in a vLLM cache. The ridge
    weights were fitted on pre-RoPE keys, so source K is unrotated before the
    affine mapping and target K is rotated back at the same absolute positions.
    """
    if not source:
        raise ValueError("source cache is empty")
    if positions.numel() == 0 or positions[0].item() != 0:
        raise ValueError(
            "connector currently requires a complete prefix from position zero"
        )
    if not torch.equal(
        positions, torch.arange(len(positions), device=positions.device)
    ):
        raise ValueError("source positions must be contiguous")
    source_keys = {
        layer: qwen3_rope(key, positions, theta=source_rope_theta, inverse=True)
        for layer, (key, _) in source.items()
    }
    source_values = {layer: value for layer, (_, value) in source.items()}
    target: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in range(len(artifact.manifest.source_layers_per_target["k"])):
        pre_rope_key = artifact.apply_layer(
            layer, "k", source_keys, device=device, dtype=dtype
        )
        key = qwen3_rope(
            pre_rope_key,
            positions.to(device=pre_rope_key.device),
            theta=target_rope_theta,
        )
        value = artifact.apply_layer(
            layer, "v", source_values, device=device, dtype=dtype
        )
        target[layer] = (key, value)
    return target
