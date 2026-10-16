"""Qwen3 dense decoder backbone (Decision-2.0-Kai-0.6B), returning final hidden states."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ....accel.kernels import KernelSet
from .common import GatedMLP, RMSNorm, apply_rotary, attention, causal_mask
from .tree import Tree

MODEL_TYPE = "qwen3"


class Qwen3Attention(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden = config["hidden_size"]
        self.head_dim = (
            config.get("head_dim") or hidden // config["num_attention_heads"]
        )
        self.groups = config["num_attention_heads"] // config["num_key_value_heads"]
        self.scaling = self.head_dim**-0.5
        bias = bool(config.get("attention_bias", False))
        self.q_proj = nn.Linear(
            hidden, config["num_attention_heads"] * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            hidden, config["num_key_value_heads"] * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            hidden, config["num_key_value_heads"] * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            config["num_attention_heads"] * self.head_dim, hidden, bias=bias
        )
        self.q_norm = RMSNorm(self.head_dim, config["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, config["rms_norm_eps"])

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor | Tree | None,
        kernels: KernelSet,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(
            1, 2
        )
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = rotary
        query, key = apply_rotary(query, key, cos, sin)
        output = attention(
            kernels, query, key, value, mask, groups=self.groups, scaling=self.scaling
        )
        output = output.reshape(*input_shape, -1).contiguous()
        out: torch.Tensor = self.o_proj(output)
        return out


class Qwen3Layer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = GatedMLP(config["hidden_size"], config["intermediate_size"])
        self.input_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(
            config["hidden_size"], config["rms_norm_eps"]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor | Tree | None,
        kernels: KernelSet,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, rotary, mask, kernels)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3Rotary(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        rope = config.get("rope_parameters") or {
            "rope_theta": config.get("rope_theta", 10000.0),
            "rope_type": "default",
        }
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"unsupported Qwen3 rope type {rope.get('rope_type')!r}")
        base = rope["rope_theta"]
        dim = (
            config.get("head_dim")
            or config["hidden_size"] // config["num_attention_heads"]
        )
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * 1.0
            sin = emb.sin() * 1.0
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3Backbone(nn.Module):
    model_type = MODEL_TYPE

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        if any(
            kind != "full_attention"
            for kind in config.get("layer_types") or ["full_attention"]
        ):
            raise ValueError("sliding-window Qwen3 layers are not supported")
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            [Qwen3Layer(config) for _ in range(config["num_hidden_layers"])]
        )
        self.norm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.rotary_emb = Qwen3Rotary(config)
        self.kernels: KernelSet | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        masks: dict[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """``masks`` (``{"full", ...}``, built from host-known padding) replaces the mask derived from
        ``attention_mask``, which reads the mask back to the host."""
        assert self.kernels is not None, "bind kernels before running the backbone"
        hidden_states = self.embed_tokens(input_ids)
        length = hidden_states.shape[1]
        position_ids = torch.arange(length, device=hidden_states.device).unsqueeze(0)
        mask = causal_mask(attention_mask, length) if masks is None else masks["full"]
        rotary = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary, mask, self.kernels)
        normed: torch.Tensor = self.norm(hidden_states)
        return normed

    def forward_tree(self, input_ids: torch.Tensor, tree: Tree) -> torch.Tensor:
        """The packed shared-context row ([1, L] ids) through every layer; [1, L, hidden]."""
        assert self.kernels is not None, "bind kernels before running the backbone"
        hidden_states = self.embed_tokens(input_ids)
        rotary = self.rotary_emb(hidden_states, tree.positions)
        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary, tree, self.kernels)
        normed: torch.Tensor = self.norm(hidden_states)
        return normed
