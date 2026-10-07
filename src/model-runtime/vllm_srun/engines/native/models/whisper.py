"""Whisper encoder (Vela 1.0 Omni's speech branch), returning the last hidden states.

Every block reproduces the operation order of Transformers' ``WhisperEncoder``
with SDPA attention: two GELU convolutions (the second strides by two), the
learned positions of every source frame, pre-norm layers whose query is
scaled before attention (SDPA then runs with scale 1, as Whisper's original
order asks), and the final LayerNorm. Input is always a full 30-second
window (``max_source_positions`` frames after the stride). Parameter names
match the checkpoints' ``WhisperEncoder`` namespace.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ....accel.kernels import KernelSet
from .modernbert import ACTIVATIONS

MODEL_TYPE = "whisper_encoder"


class WhisperAttention(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden = config["d_model"]
        self.heads = config["encoder_attention_heads"]
        self.head_dim = hidden // self.heads
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, hidden_states: torch.Tensor, kernels: KernelSet) -> torch.Tensor:
        batch, length, hidden = hidden_states.shape
        shape = (batch, length, self.heads, self.head_dim)
        query = (self.q_proj(hidden_states) * self.scaling).view(shape)
        key = self.k_proj(hidden_states).view(shape)
        value = self.v_proj(hidden_states).view(shape)
        query, key, value = (
            x.transpose(1, 2).contiguous() for x in (query, key, value)
        )
        output = kernels("sdpa")(
            query, key, value, None, scale=1.0, is_causal=False, enable_gqa=False
        )
        output = output.transpose(1, 2).contiguous().reshape(batch, length, hidden)
        out: torch.Tensor = self.out_proj(output)
        return out


class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden = config["d_model"]
        activation = config.get("activation_function", "gelu")
        if activation not in ACTIVATIONS:
            raise ValueError(f"unsupported Whisper activation {activation!r}")
        self.act = ACTIVATIONS[activation]
        self.self_attn = WhisperAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, config["encoder_ffn_dim"])
        self.fc2 = nn.Linear(config["encoder_ffn_dim"], hidden)
        self.final_layer_norm = nn.LayerNorm(hidden)

    def forward(self, hidden_states: torch.Tensor, kernels: KernelSet) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.self_attn_layer_norm(hidden_states), kernels
        )
        out: torch.Tensor = hidden_states + self.fc2(
            self.act(self.fc1(self.final_layer_norm(hidden_states)))
        )
        return out


class WhisperEncoderBackbone(nn.Module):
    """Log-mel ``input_features`` ``[B, mels, frames]`` to ``{"hidden": [B, positions, d_model]}``."""

    model_type = MODEL_TYPE

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        if config.get("scale_embedding"):
            raise ValueError("scaled Whisper embeddings are not supported")
        hidden, mels = config["d_model"], config["num_mel_bins"]
        self.config = config
        self.positions: int = config["max_source_positions"]
        self.conv1 = nn.Conv1d(mels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(self.positions, hidden)
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(config["encoder_layers"])]
        )
        self.layer_norm = nn.LayerNorm(hidden)
        self.kernels: KernelSet | None = None

    @property
    def frames(self) -> int:
        """The input frames of one window: the positions times both convolutions' strides."""
        return self.positions * 2

    def forward(self, input_features: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self.kernels is not None, "bind kernels before running the backbone"
        if input_features.shape[-1] != self.frames:
            raise ValueError(
                f"Whisper takes {self.frames} feature frames, not {input_features.shape[-1]}"
            )
        hidden_states = F.gelu(self.conv1(input_features))
        hidden_states = F.gelu(self.conv2(hidden_states)).permute(0, 2, 1)
        hidden_states = hidden_states + self.embed_positions.weight
        for layer in self.layers:
            hidden_states = layer(hidden_states, self.kernels)
        return {"hidden": self.layer_norm(hidden_states)}
