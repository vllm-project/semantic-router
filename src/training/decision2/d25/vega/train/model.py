"""Code-readout decision model for Decision 2.5 Vega training, plus checkpoint I/O.

The trainable model is the Qwen3.5 text backbone of ``Qwen3_5ForConditionalGeneration`` and a
255-way readout initialised from the ``lm_head`` rows of the answer codes (the recipe of
perplexity-ai/pplx-decider-v1.1-27b, Apache-2.0). The vision tower is frozen: it never goes to the
GPU and is copied unchanged into every export.

Training runs the backbone *packed*: the rows of a micro-batch are concatenated into one sequence
with restarted positions, the 48 Gated DeltaNet layers run FLA's varlen kernels (``cu_seqlens``) and
the 16 full-attention layers run varlen flash attention, causal or bidirectional within each row.
This is the same function as Perplexity's left-padded SDPA forward, without padding.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from d25.vega.common import decision_format as fmt

BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ATTENTION_MODES = ("causal", "noncausal_full_attention")
ATTN_NAME = "d25_varlen"
NEG_INF = -1e9
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)

_KERNELS: dict[str, Any] = {}


def kernels() -> dict[str, Any]:
    """FLA and flash-attention entry points; fails loudly instead of falling back to reference code."""
    if not _KERNELS:
        try:
            from fla.modules.convolution import causal_conv1d
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        except Exception as exc:  # pragma: no cover - environment error
            raise RuntimeError(
                "flash-linear-attention 0.5.2 must be importable (put the fla overlay on PYTHONPATH); "
                "the reference gated-delta path is slow and faulted in backward on ROCm"
            ) from exc
        _KERNELS["chunk_gated_delta_rule"] = chunk_gated_delta_rule
        _KERNELS["causal_conv1d"] = causal_conv1d
        try:
            from flash_attn import flash_attn_varlen_func

            _KERNELS["flash_attn_varlen_func"] = flash_attn_varlen_func
        except Exception:
            _KERNELS["flash_attn_varlen_func"] = None
    return _KERNELS


# ---------------------------------------------------------------------------------------------
# Packed (varlen) forward pieces
# ---------------------------------------------------------------------------------------------

ATTENTION_BACKEND = {
    "name": "flash"
}  # "flash" (flash_attn varlen) or "sdpa" (per-row SDPA loop)
CONV_BACKEND = {
    "name": "fla"
}  # "fla" (Triton varlen kernel) or "torch" (conv1d + boundary fix)


def varlen_attention(
    module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs
):
    """Attention over packed rows: [1, H, T, D] in, [1, T, H, D] out; causality from the module."""
    cu = kwargs.get("cu_seq_lens_q")
    if cu is None or attention_mask is not None or query.shape[0] != 1:
        raise RuntimeError(
            "the d25 training model only runs packed batches (cu_seq_lens_q, no padding mask)"
        )
    causal = bool(getattr(module, "is_causal", True))
    q = query.transpose(1, 2)[0]
    k = key.transpose(1, 2)[0]
    v = value.transpose(1, 2)[0]
    flash = kernels()["flash_attn_varlen_func"]
    if ATTENTION_BACKEND["name"] == "flash" and flash is not None:
        max_len = int(kwargs["max_length_q"])
        out = flash(
            q,
            k,
            v,
            cu,
            cu,
            max_len,
            max_len,
            dropout_p=0.0,
            softmax_scale=scaling,
            causal=causal,
        )
        return out[None], None
    bounds = kwargs["cu_seqlens_list"]
    outs = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        qi = q[start:end].transpose(0, 1)[None]
        ki = k[start:end].transpose(0, 1)[None]
        vi = v[start:end].transpose(0, 1)[None]
        oi = F.scaled_dot_product_attention(
            qi, ki, vi, is_causal=causal, scale=scaling, enable_gqa=True
        )
        outs.append(oi[0].transpose(0, 1))
    return torch.cat(outs, dim=0)[None], None


def register_attention() -> None:
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    AttentionInterface.register(ATTN_NAME, varlen_attention)
    AttentionMaskInterface.register(ATTN_NAME, sdpa_mask)


def _conv_torch(
    x: torch.Tensor, weight: torch.Tensor, cu: torch.Tensor | None, bounds
) -> torch.Tensor:
    """Depthwise causal conv + SiLU over packed rows ([1, T, C]); taps never cross a row start."""
    _, length, channels = x.shape
    taps = weight.shape[-1]
    y = F.conv1d(
        x.transpose(1, 2), weight.unsqueeze(1), None, padding=taps - 1, groups=channels
    )[..., :length]
    y = y.transpose(1, 2)
    if cu is not None and len(bounds) > 2:
        fixes = []
        for start, end in zip(bounds[1:-1], bounds[2:]):
            for d in range(min(taps - 1, end - start)):
                t = start + d
                leak = sum(
                    weight[:, taps - 1 - j] * x[0, t - j]
                    for j in range(d + 1, taps)
                    if t - j >= 0
                )
                fixes.append((t, leak))
        if fixes:
            index = torch.tensor([t for t, _ in fixes], device=x.device)
            delta = torch.stack([leak for _, leak in fixes]).to(y.dtype)
            y = y.index_put((torch.zeros_like(index), index), -delta, accumulate=True)
    return F.silu(y)


def gated_deltanet_forward(
    self, hidden_states, cache_params=None, attention_mask=None, **kwargs
):
    """Qwen3_5GatedDeltaNet.forward for packed rows: FLA varlen conv + chunked gated delta rule.

    Same algebra as transformers 5.17 (projections, conv + SiLU, q/k repeat, in-kernel q/k L2 norm,
    gated RMSNorm, out projection); only the kernels see the row boundaries.
    """
    if cache_params is not None or attention_mask is not None:
        raise RuntimeError(
            "the d25 training model only runs packed batches without a cache or padding"
        )
    k = kernels()
    cu = kwargs["fla_cu_seqlens"]
    cu_cpu = kwargs.get("fla_cu_seqlens_cpu")
    batch, length, _ = hidden_states.shape
    mixed_qkv = self.in_proj_qkv(hidden_states)
    z = self.in_proj_z(hidden_states).reshape(batch, length, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    weight = self.conv1d.weight.squeeze(1)
    if CONV_BACKEND["name"] == "fla":
        mixed_qkv, _ = k["causal_conv1d"](
            mixed_qkv,
            weight,
            None,
            activation="silu",
            backend="triton",
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
        )
    else:
        mixed_qkv = _conv_torch(mixed_qkv, weight, cu, kwargs["cu_seqlens_list"])
    query, key, value = torch.split(
        mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
    )
    query = query.reshape(batch, length, -1, self.head_k_dim)
    key = key.reshape(batch, length, -1, self.head_k_dim)
    value = value.reshape(batch, length, -1, self.head_v_dim)
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    core, _ = k["chunk_gated_delta_rule"](
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
    )
    core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return self.out_proj(core.reshape(batch, length, -1))


def patch_text_model(text_model: nn.Module, attention_mode: str) -> None:
    """Route a Qwen3_5TextModel to the packed kernels and set the full-attention causality."""
    import types

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Attention,
        Qwen3_5GatedDeltaNet,
    )

    if attention_mode not in ATTENTION_MODES:
        raise ValueError(f"attention_mode must be one of {ATTENTION_MODES}")
    kernels()
    register_attention()
    text_model.config._attn_implementation = ATTN_NAME
    seen = {"gdn": 0, "attn": 0}
    for module in text_model.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            module.forward = types.MethodType(gated_deltanet_forward, module)
            seen["gdn"] += 1
        elif isinstance(module, Qwen3_5Attention):
            module.is_causal = attention_mode == "causal"
            seen["attn"] += 1
    if not seen["gdn"] or not seen["attn"]:
        raise RuntimeError(f"unexpected Qwen3.5 layout: {seen}")


class PackedBatch:
    """One micro-batch: rows concatenated into one sequence; CPU tensors until ``to``."""

    def __init__(
        self,
        token_lists: list[list[int]],
        counts: list[int],
        targets: list[list[float]],
        weights: list[float],
    ):
        lengths = [len(ids) for ids in token_lists]
        bounds = [0]
        for n in lengths:
            bounds.append(bounds[-1] + n)
        self.bounds = bounds
        self.rows = len(token_lists)
        self.tokens = bounds[-1]
        self.max_len = max(lengths)
        self.input_ids = torch.tensor(
            [t for ids in token_lists for t in ids], dtype=torch.long
        )
        self.position_ids = torch.cat(
            [torch.arange(n, dtype=torch.long) for n in lengths]
        )
        self.cu_cpu = torch.tensor(bounds, dtype=torch.long)
        self.last = self.cu_cpu[1:] - 1
        self.counts = torch.tensor(counts, dtype=torch.long)
        self.target = torch.zeros(
            len(token_lists), fmt.MAX_OPTIONS, dtype=torch.float32
        )
        for i, row in enumerate(targets):
            self.target[i, : len(row)] = torch.tensor(row, dtype=torch.float32)
        self.weight = torch.tensor(weights, dtype=torch.float32)

    def pin(self) -> PackedBatch:
        for name in ("input_ids", "position_ids", "last", "counts", "target", "weight"):
            setattr(self, name, getattr(self, name).pin_memory())
        return self

    def to(
        self, device: torch.device
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
        """(model inputs, target [N, 255] FP32, weight [N]) on ``device``."""
        cu = self.cu_cpu.to(device, non_blocking=True)
        inputs = {
            "input_ids": self.input_ids.to(device, non_blocking=True)[None],
            "position_ids": self.position_ids.to(device, non_blocking=True)[None],
            "fla_cu_seqlens": cu,
            "fla_cu_seqlens_cpu": self.cu_cpu,
            "cu_seq_lens_q": cu.to(torch.int32),
            "cu_seq_lens_k": cu.to(torch.int32),
            "max_length_q": self.max_len,
            "max_length_k": self.max_len,
            "cu_seqlens_list": self.bounds,
            "last": self.last.to(device, non_blocking=True),
            "counts": self.counts.to(device, non_blocking=True),
        }
        return (
            inputs,
            self.target.to(device, non_blocking=True),
            self.weight.to(device, non_blocking=True),
        )


class DecisionReadout(nn.Module):
    """Qwen3.5 text backbone + 255-way code readout (last-token pooling, packed rows)."""

    def __init__(self, text_model: nn.Module, hidden_size: int):
        super().__init__()
        self.text = text_model
        self.readout = nn.Linear(hidden_size, fmt.MAX_OPTIONS, bias=False)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """``batch`` is ``PackedBatch.to(device)[0]``; returns masked FP32 logits [rows, 255]."""
        out = self.text(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            attention_mask={"full_attention": None, "linear_attention": None},
            use_cache=False,
            cu_seq_lens_q=batch["cu_seq_lens_q"],
            cu_seq_lens_k=batch["cu_seq_lens_k"],
            max_length_q=batch["max_length_q"],
            max_length_k=batch["max_length_k"],
            fla_cu_seqlens=batch["fla_cu_seqlens"],
            fla_cu_seqlens_cpu=batch["fla_cu_seqlens_cpu"],
            cu_seqlens_list=batch["cu_seqlens_list"],
        )
        hidden = out.last_hidden_state[0].index_select(0, batch["last"])
        logits = self.readout(hidden.float()).float()
        valid = (
            torch.arange(fmt.MAX_OPTIONS, device=logits.device)[None]
            < batch["counts"][:, None]
        )
        return logits.masked_fill(~valid, NEG_INF)


def build_skeleton(
    config, attention_mode: str, device: str | torch.device = "meta"
) -> DecisionReadout:
    """DecisionReadout with uninitialised storage (meta by default), patched for packed training."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    text_config = config.text_config
    register_attention()
    text_config._attn_implementation = ATTN_NAME
    with torch.device(device):
        text = Qwen3_5TextModel(text_config)
        model = DecisionReadout(text, text_config.hidden_size)
    patch_text_model(model.text, attention_mode)
    return model


def reset_rotary(model: nn.Module, device: torch.device) -> None:
    """Recompute the non-persistent RoPE buffers after ``to_empty``."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    for module in model.modules():
        if isinstance(module, Qwen3_5TextRotaryEmbedding):
            inv_freq, scaling = module.compute_default_rope_parameters(
                module.config, device
            )
            module.inv_freq = inv_freq.to(device)
            module.original_inv_freq = inv_freq.clone().to(device)
            module.attention_scaling = scaling


# ---------------------------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------------------------


def row_losses(logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-row soft-target cross-entropy, Brier and argmax agreement (FP32)."""
    logp = F.log_softmax(logits.float(), dim=-1)
    ce = -(target * logp).sum(-1)
    brier = (logp.exp() - target).square().sum(-1)
    correct = (logits.argmax(-1) == target.argmax(-1)).float()
    return {"ce": ce, "brier": brier, "correct": correct}


# ---------------------------------------------------------------------------------------------
# Checkpoint reading (rank 0)
# ---------------------------------------------------------------------------------------------


def _weight_files(directory: Path) -> list[Path]:
    index = directory / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [directory / name for name in names]
    single = directory / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"no safetensors weights in {directory}")


def _prefixes(keys: Iterable[str]) -> tuple[str, str]:
    for key in keys:
        if key.endswith("language_model.embed_tokens.weight"):
            text = key[: -len("embed_tokens.weight")]
            return text, text[: -len("language_model.")] + "visual."
    raise ValueError("checkpoint has no language_model.embed_tokens.weight")


def read_checkpoint(
    directory: str | Path, token_ids: list[int], parts=("text", "visual", "readout")
) -> dict[str, Any]:
    """Text, vision and readout weights of a base / merged / exported checkpoint (CPU tensors).

    Accepts ``Qwen3_5ForConditionalGeneration`` layouts (readout from ``lm_head`` rows) and our
    code-readout exports (``Qwen3_5Model`` + ``readout.safetensors``).
    """
    from safetensors import safe_open

    directory = Path(directory)
    parts = set(parts)
    files = _weight_files(directory)
    keys: dict[str, Path] = {}
    for path in files:
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                keys[key] = path
    text_prefix, visual_prefix = _prefixes(keys)
    text: dict[str, torch.Tensor] = {}
    visual: dict[str, torch.Tensor] = {}
    lm_rows = None
    by_file: dict[Path, list[str]] = {}
    for key, path in keys.items():
        by_file.setdefault(path, []).append(key)
    for path, names in by_file.items():
        with safe_open(str(path), framework="pt") as handle:
            for key in names:
                if key.startswith(text_prefix) and "text" in parts:
                    text[key[len(text_prefix) :]] = handle.get_tensor(key)
                elif key.startswith(visual_prefix) and "visual" in parts:
                    visual[key[len(visual_prefix) :]] = handle.get_tensor(key)
                elif key == "lm_head.weight" and "readout" in parts:
                    lm_rows = handle.get_tensor(key)[token_ids].float().clone()
    readout_file = directory / "readout.safetensors"
    if "readout" not in parts:
        return {"text": text, "visual": visual, "readout": None, "readout_source": None}
    if readout_file.exists():
        from safetensors.torch import load_file

        readout = load_file(str(readout_file))["weight"].float()
        source = "readout.safetensors"
    elif lm_rows is not None:
        readout = lm_rows
        source = "lm_head rows of the answer codes"
    else:
        raise ValueError(
            f"{directory} has neither readout.safetensors nor lm_head.weight"
        )
    if readout.shape[0] != fmt.MAX_OPTIONS:
        raise ValueError(
            f"readout has {readout.shape[0]} rows, expected {fmt.MAX_OPTIONS}"
        )
    return {
        "text": text,
        "visual": visual,
        "readout": readout,
        "readout_source": source,
    }


def load_config(directory: str | Path):
    """Qwen3_5Config of a checkpoint directory (base, merged or export)."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(directory))


def load_tokenizer(directory: str | Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(directory))


def check_codes(directory: str | Path, codes: list[str], token_ids: list[int]) -> None:
    saved = Path(directory) / "decision_config.json"
    if saved.exists():
        config = json.loads(saved.read_text())
        if config.get("codes") != codes or config.get("token_ids") != token_ids:
            raise ValueError(
                f"{directory}: answer codes differ from decision_format.answer_codes()"
            )


# ---------------------------------------------------------------------------------------------
# Export ("code-readout v1")
# ---------------------------------------------------------------------------------------------


def export_checkpoint(
    destination: str | Path,
    *,
    config,
    text_state: dict[str, torch.Tensor],
    visual_state: dict[str, torch.Tensor],
    readout: torch.Tensor,
    tokenizer_source: str | Path,
    decision_config: dict[str, Any],
) -> None:
    """Write a SPEC "code-readout v1" directory atomically (``<dest>.partial`` then rename)."""
    from safetensors.torch import save_file
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    config = config.__class__.from_dict(config.to_dict())
    config.architectures = ["Qwen3_5Model"]
    config.text_config._attn_implementation = "sdpa"
    config._attn_implementation = "sdpa"
    state = {
        f"language_model.{k}": v.to(torch.bfloat16).contiguous()
        for k, v in text_state.items()
    }
    state.update(
        {
            f"visual.{k}": v.to(torch.bfloat16).contiguous()
            for k, v in visual_state.items()
        }
    )
    with torch.device("meta"):
        backbone = Qwen3_5Model(config)
    expected = set(backbone.state_dict().keys())
    missing, unexpected = expected - set(state), set(state) - expected
    if missing or unexpected:
        raise ValueError(
            f"export state mismatch: missing {sorted(missing)[:5]} unexpected {sorted(unexpected)[:5]}"
        )
    backbone.load_state_dict(state, strict=True, assign=True)
    backbone.save_pretrained(
        str(partial), max_shard_size="5GB", safe_serialization=True
    )
    save_file(
        {"weight": readout.detach().float().cpu().contiguous()},
        str(partial / "readout.safetensors"),
    )
    copy_tokenizer(tokenizer_source, partial)
    (partial / "decision_config.json").write_text(
        json.dumps(decision_config, indent=2) + "\n"
    )
    for path in partial.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    os.replace(partial, destination)


def copy_tokenizer(source: str | Path, destination: str | Path) -> list[str]:
    source, destination = Path(source), Path(destination)
    copied = []
    for name in TOKENIZER_FILES:
        path = source / name
        if path.exists():
            shutil.copyfile(path, destination / name)
            copied.append(name)
    if "tokenizer.json" not in copied:
        raise FileNotFoundError(f"{source} has no tokenizer.json")
    return copied


def decision_config(
    *,
    codes: list[str],
    token_ids: list[int],
    attention_mode: str,
    max_length: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "format_id": fmt.FORMAT_ID,
        "prompt": "d25-vega",
        "base_model": BASE_MODEL,
        "revision": BASE_REVISION,
        "codes": list(codes),
        "token_ids": list(token_ids),
        "temperature": 1.0,
        "attention_mode": attention_mode,
        "pooling": "last",
        "max_length": int(max_length),
        "readout_dtype": "float32",
        "provenance": provenance,
    }


def text_key_order(names: Iterable[str]) -> list[str]:
    """Stable, human-ordered parameter names (layers numerically)."""

    def key(name: str):
        return [
            int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)
        ]

    return sorted(names, key=key)


def flops_per_token(config, tokens_mean_len: float) -> float:
    """Approximate training FLOPs per token (6N matmul + attention), used only for MFU logging."""
    t = config.text_config
    n_layers = t.num_hidden_layers
    hidden = t.hidden_size
    dense = 0.0
    for kind in t.layer_types:
        if kind == "full_attention":
            q = t.num_attention_heads * t.head_dim
            kv = t.num_key_value_heads * t.head_dim
            dense += hidden * (2 * q + 2 * kv) + q * hidden
        else:
            kdim = t.linear_num_key_heads * t.linear_key_head_dim
            vdim = t.linear_num_value_heads * t.linear_value_head_dim
            dense += (
                hidden * (2 * kdim + vdim)
                + hidden * vdim
                + 2 * hidden * t.linear_num_value_heads
                + vdim * hidden
            )
        dense += 3 * hidden * t.intermediate_size
    full_layers = sum(kind == "full_attention" for kind in t.layer_types)
    attention = full_layers * 4 * tokens_mean_len * t.num_attention_heads * t.head_dim
    del n_layers
    return 6.0 * dense + 3.0 * attention


def gib(value: float) -> float:
    return value / 2**30


def finite_or_raise(value: float, what: str) -> float:
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite {what}: {value}")
    return value
