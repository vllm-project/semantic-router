"""BERT encoder backbone (Vela 1.0 Omni Nano's text tower), returning hidden states at layer exits.

Every block reproduces the operation order of Transformers' ``BertModel``
with SDPA attention: word, token-type (all zero) and absolute position
embeddings, then post-norm layers (attention, residual, LayerNorm; GELU MLP,
residual, LayerNorm). Parameter names match the checkpoints' ``BertModel``
namespace. Like the ModernBERT backbone it runs padded ``[B, T]`` rows or
sequences packed back to back (``modernbert.Layout``), attending per grid
with a key-padding mask only where a grid pads, so a row packed with others
attends exactly as it would alone.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from ....accel.kernels import KernelSet
from .modernbert import ACTIVATIONS, FULL, Group, Layout, length_groups

MODEL_TYPE = "bert"


def _group(lengths: Sequence[int], width: int, device: torch.device | str) -> Group:
    """One grid over packed rows of ``lengths``; a mask only where a row is padded."""
    width = max(*lengths, width)
    padded = any(length != width for length in lengths)
    index = None
    if len(lengths) > 1 or padded:
        index = torch.cat(
            [
                torch.arange(length, dtype=torch.long) + row * width
                for row, length in enumerate(lengths)
            ]
        ).to(device)
    mask = None
    if padded:
        valid = torch.arange(width)[None, :] < torch.tensor(list(lengths))[:, None]
        mask = valid[:, None, None, :].to(device).expand(len(lengths), 1, width, width)
    return Group(len(lengths), width, {FULL: mask}, packed=True, index=index)


class BertEmbeddings(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden = config["hidden_size"]
        self.word_embeddings = nn.Embedding(config["vocab_size"], hidden)
        self.position_embeddings = nn.Embedding(
            config["max_position_embeddings"], hidden
        )
        self.token_type_embeddings = nn.Embedding(config["type_vocab_size"], hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=config["layer_norm_eps"])

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Token type 0 everywhere, as Transformers fills an omitted ``token_type_ids``."""
        embeddings = self.word_embeddings(input_ids) + self.token_type_embeddings(
            torch.zeros_like(input_ids)
        )
        embeddings += self.position_embeddings(positions)
        normed: torch.Tensor = self.LayerNorm(embeddings)
        return normed


class BertSelfAttention(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        if hidden % self.heads:
            raise ValueError("hidden_size must be a multiple of num_attention_heads")
        self.head_dim = hidden // self.heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)

    def forward(
        self, hidden_states: torch.Tensor, layout: Layout, kernels: KernelSet
    ) -> torch.Tensor:
        query, key, value = (
            self.query(hidden_states),
            self.key(hidden_states),
            self.value(hidden_states),
        )
        if len(layout.groups) == 1:
            return self.attend(query, key, value, layout.groups[0], kernels)
        sizes = list(layout.sizes)
        parts = [
            self.attend(q, k, v, group, kernels)
            for group, q, k, v in zip(
                layout.groups,
                query.split(sizes),
                key.split(sizes),
                value.split(sizes),
                strict=True,
            )
        ]
        return torch.cat(parts)

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        group: Group,
        kernels: KernelSet,
    ) -> torch.Tensor:
        """One grid's attention; token-major in and out."""
        rows = [group.to_rows(x) for x in (query, key, value)]
        batch, width = rows[0].shape[:2]
        query, key, value = (
            x.view(batch, width, self.heads, self.head_dim).transpose(1, 2)
            for x in rows
        )
        output = kernels("sdpa")(
            query,
            key,
            value,
            group.masks[FULL],
            scale=self.head_dim**-0.5,
            is_causal=False,
            enable_gqa=False,
        )
        return group.to_tokens(output.transpose(1, 2).reshape(batch, width, -1))


class BertSelfOutput(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden = config["hidden_size"]
        self.dense = nn.Linear(hidden, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=config["layer_norm_eps"])

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        normed: torch.Tensor = self.LayerNorm(self.dense(hidden_states) + residual)
        return normed


class BertAttention(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)


class BertIntermediate(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        activation = config.get("hidden_act", "gelu")
        if activation not in ACTIVATIONS:
            raise ValueError(f"unsupported BERT activation {activation!r}")
        self.act = ACTIVATIONS[activation]
        self.dense = nn.Linear(config["hidden_size"], config["intermediate_size"])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.act(self.dense(hidden_states))


class BertOutput(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.dense = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(
            config["hidden_size"], eps=config["layer_norm_eps"]
        )

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        normed: torch.Tensor = self.LayerNorm(self.dense(hidden_states) + residual)
        return normed


class BertLayer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self, hidden_states: torch.Tensor, layout: Layout, kernels: KernelSet
    ) -> torch.Tensor:
        attended = self.attention.self(hidden_states, layout, kernels)
        hidden_states = self.attention.output(attended, hidden_states)
        out: torch.Tensor = self.output(self.intermediate(hidden_states), hidden_states)
        return out


class BertEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config["num_hidden_layers"])]
        )


class BertBackbone(nn.Module):
    """The BERT encoder without its pooler; ``encode`` returns hidden states at layer exits."""

    model_type = MODEL_TYPE

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        if config.get("position_embedding_type", "absolute") != "absolute":
            raise ValueError("only absolute BERT position embeddings are supported")
        if config.get("is_decoder") or config.get("add_cross_attention"):
            raise ValueError("only bidirectional BERT encoders are supported")
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.kernels: KernelSet | None = None

    @property
    def num_layers(self) -> int:
        return len(self.encoder.layer)

    def packed(
        self,
        lengths: Sequence[int],
        device: torch.device | str,
        width: int | None = None,
        uniform: bool = False,
    ) -> Layout:
        """Rows of ``lengths`` packed back to back, in grids of similar (``uniform``: equal) length."""
        groups = length_groups(lengths, width or 0, uniform)
        if len(groups) == 1:
            return Layout((_group(lengths, width or 0, device),))
        starts = [0, *itertools.accumulate(lengths)]
        rows = [row for group in groups for row in group]
        order = torch.cat([torch.arange(starts[row], starts[row + 1]) for row in rows])
        restore = torch.empty_like(order)
        restore[order] = torch.arange(order.numel())
        return Layout(
            tuple(
                _group([lengths[row] for row in group], width or 0, device)
                for group in groups
            ),
            order=order.to(device),
            restore=restore.to(device),
            sizes=tuple(sum(lengths[row] for row in group) for group in groups),
        )

    def padded(
        self,
        attention_mask: torch.Tensor | None,
        rows: int,
        width: int,
        device: torch.device | str,
    ) -> Layout:
        """Padded ``[rows, width]`` rows; a key mask only when some position is padding."""
        mask = None
        if attention_mask is not None and not bool(attention_mask.all()):
            valid = attention_mask.to(device=device, dtype=torch.bool)
            mask = valid[:, None, None, :].expand(rows, 1, width, width)
        return Layout((Group(rows, width, {FULL: mask}),))

    def masked(
        self, valid: torch.Tensor, rows: int, width: int, device: torch.device | str
    ) -> Layout:
        """Padded rows masked from a device-side key mask, never read back (graphs)."""
        mask = valid[:, None, None, :].expand(rows, 1, width, width)
        return Layout((Group(rows, width, {FULL: mask}),))

    def _positions(self, layout: Layout, input_ids: torch.Tensor) -> torch.Tensor:
        """Each token's position in its own sequence, in the order the layers run."""
        if not layout.groups[0].packed:
            width = input_ids.shape[-1]
            return torch.arange(width, device=input_ids.device).expand_as(input_ids)
        parts = []
        for group in layout.groups:
            if group.index is None:
                parts.append(torch.arange(group.width, device=input_ids.device))
            else:
                parts.append(group.index % group.width)
        return torch.cat(parts) if len(parts) > 1 else parts[0]

    def encode(
        self,
        input_ids: torch.Tensor,
        layout: Layout,
        exits: Sequence[int] = (),
        normalize_exits: bool = False,
    ) -> dict[int, torch.Tensor]:
        """Hidden states by exit in one pass; the last layer only when ``exits`` is empty.

        Exit 0 is the embedding output and exit ``k`` the output of layer
        ``k`` (post-norm layers end in a LayerNorm, so ``normalize_exits``
        changes nothing). ``input_ids`` is ``[N]`` packed or ``[rows, width]``.
        """
        assert self.kernels is not None, "bind kernels before running the backbone"
        exits = tuple(exits) or (self.num_layers,)
        if not all(0 <= layer <= self.num_layers for layer in exits):
            raise ValueError(f"layer exits must lie in 0..{self.num_layers}: {exits}")
        if layout.order is not None:
            input_ids = input_ids.index_select(0, layout.order)
        hidden_states = self.embeddings(input_ids, self._positions(layout, input_ids))
        out: dict[int, torch.Tensor] = {}
        if 0 in exits:
            out[0] = hidden_states
        for count, layer in enumerate(self.encoder.layer[: max(exits)], start=1):
            hidden_states = layer(hidden_states, layout, self.kernels)
            if count in exits:
                out[count] = hidden_states
        if layout.restore is not None:
            out = {
                layer: value.index_select(0, layout.restore)
                for layer, value in out.items()
            }
        return out

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Padded ``[B, T]`` rows to ``[B, T, hidden]`` last hidden states (Transformers' contract)."""
        rows, width = input_ids.shape
        layout = self.padded(attention_mask, rows, width, input_ids.device)
        return self.encode(input_ids, layout)[self.num_layers]
