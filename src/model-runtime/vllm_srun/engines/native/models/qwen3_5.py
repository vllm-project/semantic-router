"""Qwen3.5 hybrid text backbone: Gated DeltaNet layers interleaved with gated full attention.

Decision-2.0 Eos, Sol, Nox and Lux are full checkpoints of this backbone;
Vega-27B is a LoRA over the Qwen3.8-27B text model with the same layout.
Decision 3.0 configs set ``attention_mode: noncausal_full_attention``: the
full-attention layers then see the whole (left-padded) prompt while the Gated
DeltaNet layers stay causal, and image prompts take the multimodal rotary
positions of ``rope_index``.
"""

from __future__ import annotations

import itertools
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from ....accel.kernels import KernelSet
from .common import (
    GatedMLP,
    GatedRMSNorm,
    ZeroCenteredRMSNorm,
    apply_partial_rotary,
    attention,
    causal_mask,
    recurrent_mask,
)
from .forest import Forest, ForestShape, forest_attention, forest_gated_delta
from .tree import Tree, is_tree, tree_gated_delta

MODEL_TYPE = "qwen3_5_text"


class GatedDeltaNet(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden = config["hidden_size"]
        self.num_v_heads = config["linear_num_value_heads"]
        self.num_k_heads = config["linear_num_key_heads"]
        self.head_k_dim = config["linear_key_head_dim"]
        self.head_v_dim = config["linear_value_head_dim"]
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config["linear_conv_kernel_dim"]
        self.activation = config.get("hidden_act", "silu")
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = GatedRMSNorm(self.head_v_dim, config["rms_norm_eps"])
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False)
        self.in_proj_qkv = nn.Linear(
            hidden, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(hidden, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | Tree | None,
        kernels: KernelSet,
    ) -> torch.Tensor:
        if is_tree(mask):
            return tree_gated_delta(self, hidden_states, mask, kernels)
        if mask is not None:
            dtype = hidden_states.dtype
            hidden_states = (hidden_states * mask[:, :, None]).to(dtype)
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(
            batch_size, seq_len, -1, self.head_v_dim
        )
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        mixed_qkv = kernels("causal_conv1d")(
            mixed_qkv,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            activation=self.activation,
        )
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        core, _ = kernels("chunk_gated_delta_rule")(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        core = core.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core = self.norm(core, z)
        core = core.reshape(batch_size, seq_len, -1)
        out: torch.Tensor = self.out_proj(core)
        return out


class GatedAttention(nn.Module):
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
            hidden, config["num_attention_heads"] * self.head_dim * 2, bias=bias
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
        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, config["rms_norm_eps"])
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, config["rms_norm_eps"])

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor | Tree | None,
        kernels: KernelSet,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query = self.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = rotary
        query, key = apply_partial_rotary(query, key, cos, sin)
        output = attention(
            kernels, query, key, value, mask, groups=self.groups, scaling=self.scaling
        )
        output = output.reshape(*input_shape, -1).contiguous()
        output = output * torch.sigmoid(gate)
        out: torch.Tensor = self.o_proj(output)
        return out


class Qwen3_5Layer(nn.Module):
    def __init__(self, config: dict[str, Any], kind: str):
        super().__init__()
        self.kind = kind
        if kind == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        elif kind == "full_attention":
            self.self_attn = GatedAttention(config)
        else:
            raise ValueError(f"unsupported Qwen3.5 layer type {kind!r}")
        self.mlp = GatedMLP(config["hidden_size"], config["intermediate_size"])
        self.input_layernorm = ZeroCenteredRMSNorm(
            config["hidden_size"], config["rms_norm_eps"]
        )
        self.post_attention_layernorm = ZeroCenteredRMSNorm(
            config["hidden_size"], config["rms_norm_eps"]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        full_mask: torch.Tensor | Tree | None,
        linear_mask: torch.Tensor | Tree | None,
        kernels: KernelSet,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.kind == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, linear_mask, kernels)
        else:
            hidden_states = self.self_attn(hidden_states, rotary, full_mask, kernels)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def forward_forest(
        self,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        rotary: tuple[
            tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ],
        forest: Forest,
        kernels: KernelSet,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The layer over prefix rows and blocks (``forest.py``); ``rotary`` holds both tables."""
        normed_prefix = self.input_layernorm(prefix)
        normed_blocks = self.input_layernorm(blocks)
        if self.kind == "linear_attention":
            mixed = forest_gated_delta(
                self.linear_attn, normed_prefix, normed_blocks, forest, kernels
            )
        else:
            mixed = forest_attention(
                self.self_attn, normed_prefix, normed_blocks, *rotary, forest
            )
        prefix, blocks = prefix + mixed[0], blocks + mixed[1]
        prefix = prefix + self.mlp(self.post_attention_layernorm(prefix))
        blocks = blocks + self.mlp(self.post_attention_layernorm(blocks))
        return prefix, blocks


class Qwen3_5Rotary(nn.Module):
    """Interleaved multimodal RoPE; text positions are equal across its three sections."""

    inv_freq: torch.Tensor

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        rope = config["rope_parameters"]
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"unsupported Qwen3.5 rope type {rope.get('rope_type')!r}")
        base = rope["rope_theta"]
        partial = rope.get("partial_rotary_factor", 1.0)
        head_dim = (
            config.get("head_dim")
            or config["hidden_size"] // config["num_attention_heads"]
        )
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = rope.get("mrope_section", [11, 11, 10])

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)
            cos = freqs.cos() * 1.0
            sin = freqs.sin() * 1.0
        sin = self._recompose(sin)
        cos = self._recompose(cos)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _recompose(self, freq: torch.Tensor) -> torch.Tensor:
        freqs_thw = freq[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            index = slice(offset, length, 3)
            freqs_thw[..., index] = freq[dim, ..., index]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)


NONCAUSAL = "noncausal_full_attention"


def rope_index(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    grids: list[tuple[int, int, int]],
    image_token_id: int,
    merge: int,
) -> torch.Tensor:
    """``Qwen3_5Model.get_rope_index`` for image prompts: rotary positions ``[3, B, L]``, 0 on padding.

    Over each row's real tokens, text runs count on from the position reached and
    each image's tokens take (t, row, column) positions offset by it; the image
    then advances the position by its longer merged side. Images are consumed in
    row order.
    """
    positions = torch.zeros(
        3, *input_ids.shape, dtype=input_ids.dtype, device=input_ids.device
    )
    types = (input_ids == image_token_id).to(torch.int32)
    images = iter(grids)
    device = input_ids.device
    for row in range(input_ids.shape[0]):
        keep = attention_mask[row].bool()
        kinds = types[row][keep].tolist()
        parts = []
        current = 0
        for kind, group in itertools.groupby(enumerate(kinds), lambda item: item[1]):
            members = list(group)
            span = members[-1][0] + 1 - members[0][0]
            if kind == 0:
                parts.append(
                    torch.arange(span, device=device).view(1, -1).expand(3, -1)
                    + current
                )
                current += span
                continue
            t, h, w = next(images)
            grid_t, grid_h, grid_w = t, h // merge, w // merge
            temporal = torch.arange(grid_t, device=device) * 1
            rows = torch.arange(grid_h, device=device) + current
            cols = torch.arange(grid_w, device=device) + current
            t_grid, h_grid, w_grid = torch.meshgrid(temporal, rows, cols, indexing="ij")
            vision = torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(3, -1)
            vision[0] += current
            parts.append(vision)
            current += max(h, w) // merge
        positions[:, row, keep] = (
            torch.cat(parts, dim=1).reshape(3, -1).to(positions.device)
        )
    return positions


def noncausal_masks(
    attention_mask: torch.Tensor, padded: bool | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The full-attention padding mask ``[B, 1, 1, L]`` (always given) and the gated-delta padding mask.

    ``padded`` (host-known) skips reading the mask back; None derives it from the mask.
    """
    full = attention_mask.bool()[:, None, None, :]
    if padded is None:
        return full, recurrent_mask(attention_mask, attention_mask.shape[1])
    return full, attention_mask if padded and attention_mask.shape[1] > 1 else None


class Qwen3_5Backbone(nn.Module):
    model_type = MODEL_TYPE

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        kinds = config["layer_types"]
        if len(kinds) != config["num_hidden_layers"]:
            raise ValueError("layer_types must name every layer")
        self.noncausal = config.get("attention_mode", "causal") == NONCAUSAL
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([Qwen3_5Layer(config, kind) for kind in kinds])
        self.norm = ZeroCenteredRMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.rotary_emb = Qwen3_5Rotary(config)
        self.kernels: KernelSet | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        masks: dict[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """``masks`` (``{"full", "linear"}``, built from host-known padding) replaces the masks derived
        from ``attention_mask``, which read the mask back to the host."""
        hidden_states = self.embed_tokens(input_ids)
        batch, length = hidden_states.shape[:2]
        position_ids = (
            torch.arange(length, device=hidden_states.device)
            .view(1, 1, -1)
            .expand(4, batch, -1)
        )
        if masks is not None:
            full_mask, linear_mask = masks["full"], masks["linear"]
        elif self.noncausal:
            assert (
                attention_mask is not None
            ), "noncausal attention needs the padding mask"
            full_mask, linear_mask = noncausal_masks(attention_mask)
        else:
            full_mask = causal_mask(attention_mask, length)
            linear_mask = recurrent_mask(attention_mask, length)
        return self.forward_embeds(
            hidden_states, position_ids[1:], full_mask, linear_mask
        )

    def forward_embeds(
        self,
        hidden_states: torch.Tensor,
        rope_positions: torch.Tensor,
        full_mask: torch.Tensor | Tree | None,
        linear_mask: torch.Tensor | Tree | None,
    ) -> torch.Tensor:
        """Every layer and the final norm over input embeddings, with rotary positions ``[3, B, L]``."""
        assert self.kernels is not None, "bind kernels before running the backbone"
        rotary = self.rotary_emb(hidden_states, rope_positions)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, rotary, full_mask, linear_mask, self.kernels
            )
        normed: torch.Tensor = self.norm(hidden_states)
        return normed

    def forward_images(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        features: torch.Tensor,
        image_token_id: int,
        positions: torch.Tensor,
        padded: bool,
    ) -> torch.Tensor:
        """A noncausal image prompt: image features in place of the placeholder tokens, rotary ``positions`` (``rope_index``)."""
        embeds = self.embed_tokens(input_ids)
        embeds = embeds.masked_scatter(
            (input_ids == image_token_id).unsqueeze(-1),
            features.to(embeds.device, embeds.dtype),
        )
        full_mask, linear_mask = noncausal_masks(attention_mask, padded)
        return self.forward_embeds(embeds, positions, full_mask, linear_mask)

    def forward_forest(
        self,
        prefix_ids: torch.Tensor,
        prefix_mask: torch.Tensor,
        block_ids: torch.Tensor,
        block_mask: torch.Tensor,
        owner: torch.Tensor,
        shape: ForestShape,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-padded prefix rows ``[B, Lp]`` and right-padded blocks ``[N, Lb]`` through every layer.

        Block ``i`` continues from prefix row ``owner[i]``; ``shape`` holds the
        same layout as host values, so the forward reads nothing back. Returns
        the final-normed states of both, ``[B, Lp, hidden]`` and ``[N, Lb, hidden]``.
        """
        assert self.kernels is not None, "bind kernels before running the backbone"
        rows, width = prefix_ids.shape
        count, block_width = block_ids.shape
        device = prefix_ids.device
        prefix_positions = (prefix_mask.cumsum(1) - 1).clamp(min=0)
        block_positions = (
            prefix_mask.sum(1)[owner][:, None]
            + torch.arange(block_width, device=device)[None]
        )
        prefix = self.embed_tokens(prefix_ids)
        blocks = self.embed_tokens(block_ids)
        rotary = (
            self.rotary_emb(prefix, prefix_positions[None].expand(3, rows, width)),
            self.rotary_emb(
                blocks, block_positions[None].expand(3, count, block_width)
            ),
        )
        forest = Forest(prefix_mask, block_mask, owner, shape)
        for layer in self.layers:
            prefix, blocks = cast(Qwen3_5Layer, layer).forward_forest(
                prefix, blocks, rotary, forest, self.kernels
            )
        return self.norm(prefix), self.norm(blocks)

    def forward_tree(self, input_ids: torch.Tensor, tree: Tree) -> torch.Tensor:
        """The packed shared-context row ([1, L] ids) through every layer; [1, L, hidden]."""
        assert self.kernels is not None, "bind kernels before running the backbone"
        hidden_states = self.embed_tokens(input_ids)
        rope_positions = tree.positions[None].expand(4, 1, -1)[1:]
        rotary = self.rotary_emb(hidden_states, rope_positions)
        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary, tree, tree, self.kernels)
        normed: torch.Tensor = self.norm(hidden_states)
        return normed
