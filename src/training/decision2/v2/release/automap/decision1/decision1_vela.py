# Copyright 2026 The vLLM Semantic Router Authors.
# Copyright 2024 Answer.AI, LightOn, and contributors, and the HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
#
# The ModernBERT encoder below is adapted from Hugging Face Transformers 4.57.6
# (models/modernbert/modeling_modernbert.py and modeling_rope_utils.py, Apache-2.0):
# the SDPA path with YaRN rotary embeddings, reduced to inference.
"""Kai / Lex / Route: one ModernBERT encoder with Choice, Noul and Score paths.

Inference follows the native Decision 1.0 runtime: FP32 weights and math (no TF32,
no fused attention fast path), one marker per candidate, complete inputs only (no
truncation), and rows sorted by question type into physical batches of eight.
"""

from __future__ import annotations

import copy
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from .decision1_system_one import DecisionInputTooLongError, Row, content_text

KINDS = ("choice", "noul", "score")
PHYSICAL_BATCH = 8
NOUL_DEFAULT_FALSE = "No. The statement or question is not satisfied."
NOUL_DEFAULT_TRUE = "Yes. The statement or question is satisfied."
ENCODER_GEOMETRY = {
    "model_type": "modernbert",
    "hidden_size": 768,
    "num_hidden_layers": 22,
    "num_attention_heads": 12,
    "max_position_embeddings": 32768,
}


def _yarn_inverse_frequencies(
    config: dict[str, Any], base: float
) -> tuple[torch.Tensor, float]:
    scaling = config["rope_scaling"]
    dim = config["hidden_size"] // config["num_attention_heads"]
    factor = scaling["factor"]
    original = (
        scaling.get("original_max_position_embeddings")
        or config["max_position_embeddings"]
    )

    def get_mscale(scale, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    attention_factor = scaling.get("attention_factor")
    mscale, mscale_all_dim = scaling.get("mscale"), scaling.get("mscale_all_dim")
    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(
                get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
            )
        else:
            attention_factor = get_mscale(factor)
    beta_fast = scaling.get("beta_fast") or 32
    beta_slow = scaling.get("beta_slow") or 1

    def correction_dim(rotations):
        return (dim * math.log(original / (rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low, high = correction_dim(beta_fast), correction_dim(beta_slow)
    if scaling.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp(
        (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1
    )
    pos_freqs = base ** (torch.arange(0, dim, 2).to(dtype=torch.float) / dim)
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (factor * pos_freqs)
    extrapolation_factor = 1 - ramp.to(dtype=torch.float)
    inverse = (
        interpolation * (1 - extrapolation_factor)
        + extrapolation * extrapolation_factor
    )
    return inverse, attention_factor


class RotaryEmbedding(nn.Module):
    def __init__(self, config: dict[str, Any], base: float):
        super().__init__()
        scaling = config.get("rope_scaling")
        if (
            not isinstance(scaling, dict)
            or scaling.get("rope_type", scaling.get("type")) != "yarn"
        ):
            raise ValueError("Decision 1.0 encoders use YaRN rotary embeddings")
        inverse, self.attention_scaling = _yarn_inverse_frequencies(config, base)
        self.register_buffer("inv_freq", inverse, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inverse = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        positions = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inverse.float() @ positions.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class Attention(nn.Module):
    def __init__(self, config: dict[str, Any], layer_id: int):
        super().__init__()
        hidden = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = hidden // self.num_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.Wqkv = nn.Linear(
            hidden, 3 * self.all_head_size, bias=config["attention_bias"]
        )
        if layer_id % config["global_attn_every_n_layers"] != 0:
            self.local = True
            base = config["local_rope_theta"]
            if base is None:
                base = config["global_rope_theta"]
        else:
            self.local = False
            base = config["global_rope_theta"]
        self.rotary_emb = RotaryEmbedding(config, base)
        self.Wo = nn.Linear(hidden, hidden, bias=config["attention_bias"])

    def forward(self, hidden_states, attention_mask, sliding_window_mask, position_ids):
        batch = hidden_states.shape[0]
        qkv = self.Wqkv(hidden_states).view(batch, -1, 3, self.num_heads, self.head_dim)
        cos, sin = self.rotary_emb(qkv, position_ids=position_ids)
        query, key, value = qkv.transpose(3, 1).unbind(dim=2)
        query, key = _apply_rotary(query, key, cos, sin)
        mask = sliding_window_mask if self.local else attention_mask
        if (
            torch.version.hip is not None
            and hidden_states.device.type == "cuda"
            and torch.backends.cuda.mem_efficient_sdp_enabled()
        ):
            # ROCm's efficient SDPA kernel needs contiguous post-RoPE inputs.
            query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        output = functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask
        )
        output = output.transpose(1, 2).contiguous().view(batch, -1, self.all_head_size)
        return self.Wo(output)


class MLP(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden, intermediate = config["hidden_size"], int(config["intermediate_size"])
        if config["hidden_activation"] != "gelu":
            raise ValueError("Decision 1.0 encoders use exact GELU")
        self.Wi = nn.Linear(hidden, intermediate * 2, bias=config["mlp_bias"])
        self.Wo = nn.Linear(intermediate, hidden, bias=config["mlp_bias"])

    def forward(self, hidden_states):
        values, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(functional.gelu(values) * gate)


class EncoderLayer(nn.Module):
    def __init__(self, config: dict[str, Any], layer_id: int):
        super().__init__()
        hidden, eps, bias = (
            config["hidden_size"],
            config["norm_eps"],
            config["norm_bias"],
        )
        self.attn_norm = (
            nn.Identity() if layer_id == 0 else nn.LayerNorm(hidden, eps=eps, bias=bias)
        )
        self.attn = Attention(config, layer_id)
        self.mlp_norm = nn.LayerNorm(hidden, eps=eps, bias=bias)
        self.mlp = MLP(config)

    def forward(self, hidden_states, attention_mask, sliding_window_mask, position_ids):
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states),
            attention_mask,
            sliding_window_mask,
            position_ids,
        )
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class Embeddings(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden = config["hidden_size"]
        self.tok_embeddings = nn.Embedding(
            config["vocab_size"], hidden, padding_idx=config["pad_token_id"]
        )
        self.norm = nn.LayerNorm(
            hidden, eps=config["norm_eps"], bias=config["norm_bias"]
        )

    def forward(self, input_ids):
        return self.norm(self.tok_embeddings(input_ids))


class Encoder(nn.Module):
    """Parameter names match transformers' ModernBertModel."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(config, index)
                for index in range(config["num_hidden_layers"])
            ]
        )
        self.final_norm = nn.LayerNorm(
            config["hidden_size"], eps=config["norm_eps"], bias=config["norm_bias"]
        )

    def masks(self, attention_mask: torch.Tensor, dtype: torch.dtype):
        batch, length = attention_mask.shape
        expanded = (
            attention_mask[:, None, None, :].expand(batch, 1, length, length).to(dtype)
        )
        inverted = torch.tensor(1.0, dtype=dtype) - expanded
        global_mask = inverted.masked_fill(
            inverted.to(torch.bool), torch.finfo(dtype).min
        )
        rows = torch.arange(length).unsqueeze(0)
        window = (
            (torch.abs(rows - rows.T) <= self.config["local_attention"] // 2)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(attention_mask.device)
        )
        sliding = global_mask.masked_fill(window.logical_not(), torch.finfo(dtype).min)
        return global_mask, sliding


class VelaDecision(nn.Module):
    """Shared encoder (Noul path), private Choice and Score encoder copies, typed heads."""

    def __init__(self, encoder_config: dict[str, Any], head: dict[str, Any]):
        super().__init__()
        for key, expected in ENCODER_GEOMETRY.items():
            if encoder_config.get(key) != expected:
                raise ValueError(
                    f"Unsupported Decision 1.0 encoder configuration: {key}"
                )
        if head.get("head_layers") != 2 or head.get("head_heads") != 12:
            raise ValueError("Unsupported Decision 1.0 head geometry")
        hidden = encoder_config["hidden_size"]
        self.encoder = Encoder(encoder_config)
        self.type_embedding = nn.Embedding(3, hidden)
        self.heads = nn.ModuleDict(
            {
                kind: nn.ModuleList(
                    [
                        nn.TransformerEncoderLayer(
                            hidden,
                            12,
                            4 * hidden,
                            0.1,
                            activation="relu",
                            batch_first=True,
                            norm_first=True,
                        )
                        for _ in range(2)
                    ]
                )
                for kind in KINDS
            }
        )
        self.scorers = nn.ModuleDict(
            {
                kind: nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, 1),
                )
                for kind in KINDS
            }
        )
        self.choice_blocks = nn.ModuleList(
            copy.deepcopy(layer) for layer in self.encoder.layers
        )
        self.choice_final_norm = copy.deepcopy(self.encoder.final_norm)
        self.score_blocks = nn.ModuleList(
            copy.deepcopy(layer) for layer in self.encoder.layers
        )
        self.score_final_norm = copy.deepcopy(self.encoder.final_norm)

    @staticmethod
    def _path(hidden, layers, final_norm, masks, positions):
        for layer in layers:
            hidden = layer(hidden, masks[0], masks[1], positions)
        return final_norm(hidden)

    def forward(
        self, input_ids, attention_mask, kind_ids, marker_positions, valid_candidates
    ):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(
            0
        )
        masks = self.encoder.masks(attention_mask, torch.float32)
        embedded = self.encoder.embeddings(input_ids)
        present = set(kind_ids.tolist())
        type_offset = self.type_embedding(kind_ids)[:, None, :]
        hidden_by_kind = {}
        if 0 in present:
            hidden = self._path(
                embedded, self.choice_blocks, self.choice_final_norm, masks, positions
            )
            hidden_by_kind["choice"] = hidden + type_offset.to(hidden.dtype)
        if 1 in present:
            hidden = self._path(
                embedded, self.encoder.layers, self.encoder.final_norm, masks, positions
            )
            hidden_by_kind["noul"] = hidden + type_offset.to(hidden.dtype)
        if 2 in present:
            hidden = self._path(
                embedded, self.score_blocks, self.score_final_norm, masks, positions
            )
            hidden_by_kind["score"] = hidden + type_offset.to(hidden.dtype)
        pad = ~attention_mask.bool()
        output = torch.empty(
            marker_positions.shape, device=input_ids.device, dtype=torch.float32
        )
        for index, kind in enumerate(KINDS):
            rows = torch.nonzero(kind_ids == index, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            hidden = hidden_by_kind[kind].index_select(0, rows)
            branch_pad = pad.index_select(0, rows)
            for layer in self.heads[kind]:
                hidden = layer(hidden, src_key_padding_mask=branch_pad)
            where = marker_positions.index_select(0, rows)
            markers = torch.gather(
                hidden, 1, where[:, :, None].expand(-1, -1, hidden.shape[-1])
            )
            output = output.index_copy(
                0, rows, self.scorers[kind](markers).squeeze(-1).float()
            )
        return output.masked_fill(~valid_candidates, torch.finfo(torch.float32).min)


@contextmanager
def native_flags():
    """The published runtime's settings: no fused attention fast path and no TF32, restored afterwards."""
    fastpath = torch.backends.mha.get_fastpath_enabled()
    matmul, cudnn = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class VelaRuntime:
    """Loaded weights, tokenizer and the complete-input limit of one Decision 1.0 encoder."""

    noul_default_false = NOUL_DEFAULT_FALSE
    noul_default_true = NOUL_DEFAULT_TRUE
    noul_explicit_null = "use_default"

    def __init__(self, model: VelaDecision, tokenizer: Any, max_input_tokens: int):
        self.model = model
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        special = (
            (
                tokenizer.cls_token_id
                if tokenizer.cls_token_id is not None
                else tokenizer.bos_token_id
            ),
            (
                tokenizer.sep_token_id
                if tokenizer.sep_token_id is not None
                else tokenizer.eos_token_id
            ),
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
        )
        if any(value is None for value in special):
            raise ValueError(
                "The tokenizer must define CLS/BOS, SEP/EOS, PAD and MASK tokens"
            )
        self.bos, self.sep, self.pad, self.marker = special

    @classmethod
    def load(
        cls, root: Path, descriptor: dict[str, Any], *, max_input_tokens: int, device
    ):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        model_config = json.loads(
            (root / descriptor["model_config"]).read_text(encoding="utf-8")
        )
        if (
            model_config.get("arm") != "all22"
            or model_config.get("training_arm") != "S22"
            or model_config.get("type_order") != list(KINDS)
            or model_config.get("packing", {}).get("state_truncation") != "error"
        ):
            raise ValueError(
                "Only the three-path all22 / S22 Decision 1.0 encoder is supported"
            )
        encoder_config = json.loads(
            (root / descriptor["backbone"]["config"]).read_text(encoding="utf-8")
        )
        model = VelaDecision(encoder_config, model_config["head"])
        state = {
            f"encoder.{name}": tensor
            for name, tensor in load_file(
                str(root / descriptor["backbone"]["weights"][0])
            ).items()
        }
        for role in ("decision_heads", "choice_encoder", "score_encoder"):
            part = load_file(str(root / descriptor["decision_weights"][role]))
            if set(part) & set(state):
                raise ValueError("Decision 1.0 weight files overlap")
            state.update(part)
        if any(tensor.dtype != torch.float32 for tensor in state.values()):
            raise ValueError("Decision 1.0 encoder weights must be FP32")
        model.load_state_dict(state, strict=True)
        expected = model_config.get("parameters")
        loaded = sum(parameter.numel() for parameter in model.parameters())
        if expected is not None and loaded != expected:
            raise ValueError(
                f"Loaded {loaded:,} parameters; the model declares {expected:,}"
            )
        model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            str((root / descriptor["tokenizer"]["json"]).parent),
            trust_remote_code=False,
        )
        return cls(model, tokenizer, max_input_tokens)

    def _tokens(self, text: str) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=False, truncation=False)[
            "input_ids"
        ]
        if not ids:
            raise ValueError("A candidate, question or state renders to no tokens")
        return list(ids)

    def encode(self, row: Row, cache: dict[str, list[int]]) -> dict[str, Any]:
        def tokens(text):
            if text not in cache:
                cache[text] = self._tokens(text)
            return cache[text]

        ids = [
            self.bos,
            *tokens(f"{row.type} question: {content_text(row.instructions)}"),
            self.sep,
        ]
        positions = []
        for index, candidate in enumerate(row.candidates):
            if candidate.description is None:
                text = candidate.key
            else:
                text = content_text(candidate.description)
                if row.type == "choice":
                    text = f"{candidate.key}: {text}"
            if row.type == "score":
                text = f"level {index}: {text}"
            positions.append(len(ids))
            ids.extend((self.marker, *tokens(text), self.sep))
        state = tokens(row.state)
        room = self.max_input_tokens - len(ids) - 1
        if room < 1 or len(state) > room:
            raise DecisionInputTooLongError(
                f"{row.question_id}: the complete input exceeds {self.max_input_tokens} "
                "tokens; no truncation allowed"
            )
        ids.extend((*state, self.sep))
        return {"ids": ids, "positions": positions, "kind": KINDS.index(row.type)}

    def predict(self, rows: list[Row]) -> tuple[list[list[float]], list[int]]:
        """Probabilities per row in request order, and input tokens per row."""
        cache: dict[str, list[int]] = {}
        encoded = [self.encode(row, cache) for row in rows]
        order = sorted(
            range(len(rows)), key=lambda index: rows[index].type.capitalize()
        )
        device = next(self.model.parameters()).device
        results: list[list[float] | None] = [None] * len(rows)
        with torch.inference_mode(), native_flags():
            for start in range(0, len(order), PHYSICAL_BATCH):
                chunk = order[start : start + PHYSICAL_BATCH]
                items = [encoded[index] for index in chunk]
                length = max(len(item["ids"]) for item in items)
                width = max(len(item["positions"]) for item in items)
                input_ids = torch.full((len(items), length), self.pad, dtype=torch.long)
                mask = torch.zeros((len(items), length), dtype=torch.bool)
                markers = torch.zeros((len(items), width), dtype=torch.long)
                valid = torch.zeros((len(items), width), dtype=torch.bool)
                for slot, item in enumerate(items):
                    input_ids[slot, : len(item["ids"])] = torch.tensor(item["ids"])
                    mask[slot, : len(item["ids"])] = True
                    markers[slot, : len(item["positions"])] = torch.tensor(
                        item["positions"]
                    )
                    valid[slot, : len(item["positions"])] = True
                kinds = torch.tensor([item["kind"] for item in items])
                logits = self.model(
                    input_ids.to(device),
                    mask.to(device),
                    kinds.to(device),
                    markers.to(device),
                    valid.to(device),
                )
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite Decision logits")
                for slot, index in enumerate(chunk):
                    count = len(encoded[index]["positions"])
                    results[index] = logits[slot, :count].softmax(-1).cpu().tolist()
        return results, [len(item["ids"]) for item in encoded]  # type: ignore[return-value]
