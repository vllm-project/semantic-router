"""SigLIP vision tower (Vela 1.0 Omni's image encoders), returning the attention-pooled image vector.

Every block reproduces the operation order of Transformers'
``SiglipVisionTransformer`` with SDPA attention: a patch convolution plus
learned positions, pre-norm layers (attention, residual; tanh-GELU MLP,
residual), the post LayerNorm and the multihead attention pooling head, whose
attention follows ``torch.nn.functional.multi_head_attention_forward`` with
attention weights requested (the path ``nn.MultiheadAttention`` takes there).
Parameter names match the checkpoints' ``SiglipVisionTransformer`` namespace.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ....accel.kernels import KernelSet
from .modernbert import ACTIVATIONS

MODEL_TYPE = "siglip_vision_model"
# SiglipVisionConfig's defaults, for configs that omit a field.
DEFAULTS: dict[str, Any] = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_channels": 3,
    "image_size": 224,
    "patch_size": 16,
    "hidden_act": "gelu_pytorch_tanh",
    "layer_norm_eps": 1e-6,
}


def vision_config(config: dict[str, Any]) -> dict[str, Any]:
    """A vision config with SiglipVisionConfig's defaults filled in."""
    return {**DEFAULTS, **config}


class SiglipEmbeddings(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden, patch = config["hidden_size"], config["patch_size"]
        self.patch_embedding = nn.Conv2d(
            config["num_channels"], hidden, kernel_size=patch, stride=patch
        )
        self.num_positions = (config["image_size"] // patch) ** 2
        self.position_embedding = nn.Embedding(self.num_positions, hidden)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(
            pixel_values.to(self.patch_embedding.weight.dtype)
        )
        embeddings = patches.flatten(2).transpose(1, 2)
        if embeddings.shape[1] != self.num_positions:
            raise ValueError(
                f"the image has {embeddings.shape[1]} patches; the model takes {self.num_positions}"
            )
        positioned: torch.Tensor = embeddings + self.position_embedding.weight[None]
        return positioned


class SiglipAttention(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.head_dim = hidden // self.heads
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, hidden_states: torch.Tensor, kernels: KernelSet) -> torch.Tensor:
        batch, length, hidden = hidden_states.shape
        shape = (batch, length, self.heads, self.head_dim)
        query = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        output = kernels("sdpa")(
            query,
            key,
            value,
            None,
            scale=self.head_dim**-0.5,
            is_causal=False,
            enable_gqa=False,
        )
        output = output.transpose(1, 2).contiguous().reshape(batch, length, hidden)
        out: torch.Tensor = self.out_proj(output)
        return out


class SiglipMLP(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        activation = config["hidden_act"]
        if activation not in ACTIVATIONS:
            raise ValueError(f"unsupported SigLIP activation {activation!r}")
        self.act = ACTIVATIONS[activation]
        self.fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.fc2(self.act(self.fc1(hidden_states)))
        return out


class SiglipLayer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        eps = config["layer_norm_eps"]
        self.layer_norm1 = nn.LayerNorm(config["hidden_size"], eps=eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(config["hidden_size"], eps=eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states: torch.Tensor, kernels: KernelSet) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states), kernels
        )
        out: torch.Tensor = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return out


class SiglipEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.layers = nn.ModuleList(
            [SiglipLayer(config) for _ in range(config["num_hidden_layers"])]
        )


class PoolingAttention(nn.Module):
    """``nn.MultiheadAttention``'s parameters, run as its functional form with weights requested.

    The query (the probe) differs from the key and value (the patches), so
    the packed input projection splits into a query part and a key-value part.
    """

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * hidden, hidden))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * hidden))
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """``query`` ``[B, 1, E]`` attends to ``memory`` ``[B, S, E]``; returns ``[B, 1, E]``."""
        query, memory = query.transpose(0, 1), memory.transpose(0, 1)
        targets, batch, hidden = query.shape
        head_dim = hidden // self.heads
        w_q, w_kv = torch.split(self.in_proj_weight, [hidden, 2 * hidden])
        b_q, b_kv = torch.split(self.in_proj_bias, [hidden, 2 * hidden])
        q = F.linear(query, w_q, b_q)
        kv = F.linear(memory, w_kv, b_kv)
        kv = (
            torch.unflatten(kv, -1, (2, hidden))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
        )
        k, v = kv.contiguous()
        q = q.view(targets, batch * self.heads, head_dim).transpose(0, 1)
        k = k.view(k.shape[0], batch * self.heads, head_dim).transpose(0, 1)
        v = v.view(v.shape[0], batch * self.heads, head_dim).transpose(0, 1)
        scores = torch.bmm(q * math.sqrt(1.0 / float(head_dim)), k.transpose(-2, -1))
        output = torch.bmm(F.softmax(scores, dim=-1), v)
        output = output.transpose(0, 1).contiguous().view(targets * batch, hidden)
        projected: torch.Tensor = self.out_proj(output).view(targets, batch, hidden)
        return projected.transpose(0, 1)


class SiglipPoolingHead(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden = config["hidden_size"]
        self.probe = nn.Parameter(torch.empty(1, 1, hidden))
        self.attention = PoolingAttention(hidden, config["num_attention_heads"])
        self.layernorm = nn.LayerNorm(hidden, eps=config["layer_norm_eps"])
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        probe = self.probe.repeat(hidden_states.shape[0], 1, 1)
        pooled: torch.Tensor = self.attention(probe, hidden_states)
        pooled = pooled + self.mlp(self.layernorm(pooled))
        return pooled[:, 0]


class SiglipVisionBackbone(nn.Module):
    """``pixel_values`` ``[B, C, S, S]`` to ``{"pooled": [B, hidden]}`` (the head's output)."""

    model_type = MODEL_TYPE

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        config = vision_config(config)
        if not config.get("vision_use_head", True):
            raise ValueError("the SigLIP vision tower needs its attention pooling head")
        self.config = config
        self.embeddings = SiglipEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config["hidden_size"], eps=config["layer_norm_eps"]
        )
        self.head = SiglipPoolingHead(config)
        self.kernels: KernelSet | None = None

    @property
    def image_size(self) -> int:
        return int(self.config["image_size"])

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self.kernels is not None, "bind kernels before running the backbone"
        hidden_states = self.embeddings(pixel_values)
        for layer in self.encoder.layers:
            hidden_states = layer(hidden_states, self.kernels)
        return {"pooled": self.head(self.post_layernorm(hidden_states))}
