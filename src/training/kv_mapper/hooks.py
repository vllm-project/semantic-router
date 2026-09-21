"""Capture K/V after k_norm and before RoPE. Requires torch at runtime."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class CapturedKV:
    """Per-layer K and V, shape (seq, n_kv, head_dim), CPU, no RoPE on K."""

    keys: list[Tensor]
    values: list[Tensor]


def as_bshd(out: Tensor, n_kv: int, head_dim: int) -> Tensor:
    """Normalize hook output to (batch, seq, n_kv, head_dim)."""
    if out.dim() == 4:
        _batch, a, c, d = out.shape
        if c == n_kv and d == head_dim:
            return out
        if a == n_kv and d == head_dim:
            return out.transpose(1, 2)
        raise ValueError(
            f"Unexpected 4D KV {tuple(out.shape)}; want n_kv={n_kv} head_dim={head_dim}"
        )
    if out.dim() != 3:
        raise ValueError(f"Unexpected KV rank {out.dim()} shape {tuple(out.shape)}")
    return out.view(out.shape[0], out.shape[1], n_kv, head_dim)


def attach_pre_rope_hooks(
    model: nn.Module,
    n_kv: int,
    head_dim: int,
    layer_indices: list[int] | None = None,
):
    storage: dict[int, dict[str, Tensor]] = {}
    handles = []
    layers = model.model.layers
    indices = layer_indices if layer_indices is not None else list(range(len(layers)))

    for idx in indices:
        attn = layers[idx].self_attn
        k_mod = getattr(attn, "k_norm", None)
        if k_mod is None:
            k_mod = attn.k_proj

        def make_k_hook(layer_idx: int):
            def hook(_mod, _inp, out: Tensor) -> None:
                storage.setdefault(layer_idx, {})["k"] = as_bshd(
                    out.detach(), n_kv, head_dim
                )

            return hook

        def make_v_hook(layer_idx: int):
            def hook(_mod, _inp, out: Tensor) -> None:
                storage.setdefault(layer_idx, {})["v"] = as_bshd(
                    out.detach(), n_kv, head_dim
                )

            return hook

        handles.append(k_mod.register_forward_hook(make_k_hook(idx)))
        handles.append(attn.v_proj.register_forward_hook(make_v_hook(idx)))

    return storage, handles


def remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()


@torch.inference_mode()
def capture_kv(
    model: nn.Module,
    input_ids: Tensor,
    n_kv: int,
    head_dim: int,
    layer_indices: list[int] | None = None,
) -> CapturedKV:
    indices = (
        layer_indices
        if layer_indices is not None
        else list(range(len(model.model.layers)))
    )
    storage, handles = attach_pre_rope_hooks(model, n_kv, head_dim, indices)
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        remove_hooks(handles)

    keys, values = [], []
    for i in indices:
        slot = storage.get(i)
        if not slot or "k" not in slot or "v" not in slot:
            raise RuntimeError(
                f"KV hook missed layer {i}; k_proj/k_norm or v_proj did not run"
            )
        keys.append(slot["k"][0].float().cpu())
        values.append(slot["v"][0].float().cpu())
    seq = keys[0].shape[0]
    for i, (key, value) in enumerate(zip(keys, values)):
        if key.shape != (seq, n_kv, head_dim) or value.shape != (seq, n_kv, head_dim):
            raise ValueError(
                f"layer {indices[i]} KV {tuple(key.shape)} / {tuple(value.shape)} "
                f"!= ({seq}, {n_kv}, {head_dim})"
            )
    return CapturedKV(keys=keys, values=values)
