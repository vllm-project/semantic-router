"""Stage 2 of mapper fitting: self-distillation of the fitted linear maps.

The ridge fit (stage 1) makes the mapped cache close to the target's own cache.
Stage 2 keeps the same maps and trains them so the target predicts the same
next tokens from the mapped cache as from its own (KV-Lingo, arXiv 2609.32610).
The source prefills x[1:n-1], the maps translate that cache, and the target
reads x[n] and the continuation y[1:m-1] on top of it; the loss is the KL from
the target's own distribution over y[1:m]. Both models stay frozen, and the KL
flows back through the target into the maps. Requires torch at runtime.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from src.training.kv_mapper.artifact import Manifest
from src.training.kv_mapper.hooks import attach_pre_rope_hooks, remove_hooks

CHANNELS = ("k", "v")


class LinearMapper(nn.Module):
    """The artifact's per-target-layer maps as trainable float32 parameters.

    With `rank`, each W stays frozen and only a low-rank correction is trained,
    parameterised as NoRA (arXiv 2608.31036): the down projection's columns are
    kept at unit norm, and the update is not rescaled (the paper's alpha = rank
    under the usual alpha / rank scaling), so each input coordinate's first
    update matches full training. Biases stay full.
    """

    def __init__(
        self, manifest: Manifest, tensors: dict[str, np.ndarray], rank: int = 0
    ):
        super().__init__()
        selections = manifest.source_layers_per_target["k"]
        self.source_layers = [selections[str(i)] for i in range(len(selections))]
        self.num_kv_heads = manifest.compatibility.num_kv_heads
        self.head_dim = manifest.compatibility.head_dim
        self.rank = rank
        params = {}
        for name, value in tensors.items():
            key = name.replace(".", "_")
            tensor = torch.from_numpy(np.array(value, dtype=np.float32))
            if rank and name.endswith(".W"):
                self.register_buffer(f"base_{key}", tensor)
                down = torch.empty(rank, tensor.shape[0])
                nn.init.kaiming_uniform_(down, a=math.sqrt(5))
                params[f"{key}_A"] = nn.Parameter(down)
                params[f"{key}_B"] = nn.Parameter(torch.zeros(tensor.shape[1], rank))
            else:
                params[key] = nn.Parameter(tensor)
        self.params = nn.ParameterDict(params)

    def weight(self, layer: int, channel: str, part: str) -> Tensor:
        key = f"target_{layer}_{channel}_{part}"
        if not self.rank or part == "b":
            return self.params[key]
        down = self.params[f"{key}_A"]
        down = down / down.norm(dim=0, keepdim=True).clamp_min(1e-6)
        update = self.params[f"{key}_B"] @ down
        return getattr(self, f"base_{key}") + update.T

    def forward(self, source: Sequence[tuple[Tensor, Tensor]]) -> list[list[Tensor]]:
        """Map source (key, value) pairs, each (seq, heads, dim) before RoPE."""
        mapped = []
        for layer, indices in enumerate(self.source_layers):
            channels = []
            for position, channel in enumerate(CHANNELS):
                features = torch.cat(
                    [
                        source[i][position].reshape(source[i][position].shape[0], -1)
                        for i in indices
                    ],
                    dim=-1,
                ).float()
                out = features @ self.weight(layer, channel, "W")
                out = out + self.weight(layer, channel, "b")
                channels.append(
                    out.reshape(features.shape[0], self.num_kv_heads, self.head_dim)
                )
            mapped.append(channels)
        return mapped

    def export(self) -> dict[str, np.ndarray]:
        """Merged maps in the artifact layout, whatever the training parameterisation."""
        return {
            f"target.{layer}.{channel}.{part}": self.weight(layer, channel, part)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            for layer in range(len(self.source_layers))
            for channel in CHANNELS
            for part in ("W", "b")
        }


def rotate_keys(model: nn.Module, keys: Tensor) -> Tensor:
    """(seq, heads, dim) keys before RoPE to the cache layout (1, heads, seq, dim)."""
    key = keys.transpose(0, 1).unsqueeze(0)
    positions = torch.arange(key.shape[-2], device=key.device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(key, positions)
    return apply_rotary_pos_emb(key, key, cos, sin)[1]


def mapped_cache(target: nn.Module, mapped: list[list[Tensor]]) -> DynamicCache:
    dtype = next(target.parameters()).dtype
    cache = DynamicCache()
    for layer, (key, value) in enumerate(mapped):
        cache.update(
            rotate_keys(target, key.to(dtype)),
            value.to(dtype).transpose(0, 1).unsqueeze(0),
            layer,
        )
    return cache


@torch.no_grad()
def source_pairs(
    source: nn.Module, prefix: Tensor, num_kv_heads: int, head_dim: int
) -> list[tuple[Tensor, Tensor]]:
    """Source K (after k_norm, before RoPE) and V for every layer on `prefix`."""
    slots, handles = attach_pre_rope_hooks(source, num_kv_heads, head_dim)
    try:
        source(input_ids=prefix, use_cache=False)
    finally:
        remove_hooks(handles)
    return [
        (slots[i]["k"][0], slots[i]["v"][0]) for i in range(len(source.model.layers))
    ]


def distillation_loss(
    source: nn.Module,
    target: nn.Module,
    mapper: LinearMapper,
    prefix: Sequence[int],
    continuation: Sequence[int],
) -> Tensor:
    """Mean KL(target's own || target on the mapped cache) over the continuation."""
    if len(prefix) < 2 or not continuation:  # noqa: PLR2004
        raise ValueError("need a prefix of two or more tokens and a continuation")
    device = next(target.parameters()).device
    ids = torch.tensor([list(prefix) + list(continuation)], device=device)
    n, m = len(prefix), len(continuation)
    with torch.no_grad():
        teacher = target(input_ids=ids[:, :-1], use_cache=False).logits[0, n - 1 :]
        teacher = teacher.float().log_softmax(dim=-1)
    pairs = source_pairs(
        source,
        ids[:, : n - 1].to(next(source.parameters()).device),
        mapper.num_kv_heads,
        mapper.head_dim,
    )
    pairs = [(k.to(device), v.to(device)) for k, v in pairs]
    cache = mapped_cache(target, mapper(pairs))
    student = target(
        input_ids=ids[:, n - 1 : n + m - 1], past_key_values=cache, use_cache=True
    ).logits[0]
    student = student.float().log_softmax(dim=-1)
    return (teacher.exp() * (teacher - student)).sum(dim=-1).mean()


def cosine_lr(step: int, total: int, peak: float, warmup_fraction: float) -> float:
    warmup = max(1, int(total * warmup_fraction))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def mean_kl(
    source: nn.Module,
    target: nn.Module,
    mapper: LinearMapper,
    samples: Iterable[tuple[Sequence[int], Sequence[int]]],
) -> float:
    with torch.no_grad():
        losses = [
            float(distillation_loss(source, target, mapper, prefix, continuation))
            for prefix, continuation in samples
        ]
    if not losses:
        raise ValueError("no validation samples")
    return float(np.mean(losses))


def relative_rates(mapper: LinearMapper) -> list[dict]:
    """One parameter group per tensor, its rate scaled by the RMS of the map it changes.

    Adam moves every entry by about the learning rate whatever the entry's size,
    and fitted maps differ in scale by up to fifty times (value maps against key
    maps for Qwen3-14B to 32B), so a shared rate moves the small maps far more.
    A low-rank down projection only sets directions and keeps the base rate.
    """
    groups = []
    for name, param in mapper.params.items():
        if name.endswith("_A"):
            scale = 1.0
        elif name.endswith("_B"):
            scale = getattr(mapper, f"base_{name[:-2]}").pow(2).mean().sqrt()
        else:
            scale = param.detach().pow(2).mean().sqrt()
        groups.append({"params": [param], "scale": float(scale)})
    return groups


def train(
    source: nn.Module,
    target: nn.Module,
    mapper: LinearMapper,
    samples: Iterable[tuple[Sequence[int], Sequence[int]]],
    *,
    steps: int,
    batch: int = 8,
    lr: float = 3e-5,
    warmup_fraction: float = 0.05,
    clip: float = 1.0,
    relative_lr: bool = False,
    log_every: int = 50,
    log=print,
) -> list[float]:
    """AdamW with no weight decay, cosine schedule, one sample at a time.

    Each step accumulates `batch` samples, so prefixes of any length are never
    padded together. With `relative_lr`, each tensor's rate is `lr` times the
    RMS of its map (see `relative_rates`). Returns the mean loss of every step.
    """
    for module in (source, target):
        module.requires_grad_(False)
        module.eval()
    groups = (
        relative_rates(mapper)
        if relative_lr
        else [{"params": list(mapper.parameters()), "scale": 1.0}]
    )
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=0.0)
    stream = iter(samples)
    history = []
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = group["scale"] * cosine_lr(step, steps, lr, warmup_fraction)
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(batch):
            prefix, continuation = next(stream)
            loss = distillation_loss(source, target, mapper, prefix, continuation)
            (loss / batch).backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), clip)
        optimizer.step()
        history.append(total / batch)
        if log_every and (step + 1) % log_every == 0:
            log(f"step {step + 1}/{steps} kl {history[-1]:.5f}")
    return history
