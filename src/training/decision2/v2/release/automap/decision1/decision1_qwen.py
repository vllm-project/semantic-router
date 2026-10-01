# Copyright 2026 The vLLM Semantic Router Authors.
# SPDX-License-Identifier: Apache-2.0
"""Eos / Sol / Nox / Lux: a Qwen3.5 text backbone with a candidate-endpoint head.

Inference follows the native Decision 1.0 runtime: every candidate is tokenized
as its own segment, its last token is the candidate endpoint and the final token
the global query; a shared bilinear + MLP head scores the endpoints in FP32.
On a GPU the backbone runs in BF16 (its stored precision) under BF16 autocast;
on CPU it runs in FP32. Questions run in physical batches of eight, padded to a
multiple of 32 tokens. Complete inputs only: nothing is truncated.
"""

from __future__ import annotations

import functools
import inspect
import json
import math
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from .decision1_system_one import (
    DecisionInputTooLongError,
    Row,
    canonical_json,
    content_text,
)

PHYSICAL_BATCH = 8
PAD_MULTIPLE = 32
NOUL_DEFAULT_FALSE = "The answer to the question is no."
NOUL_DEFAULT_TRUE = "The answer to the question is yes."
PROMPT_VERSION = "structured-segmented-candidate-endpoints-global-query-v2"
SUFFIX = "\n\nSelect the single option best supported by the context and instructions.\nDecision:"
QWEN3_5_MODELING = "transformers.models.qwen3_5.modeling_qwen3_5"
# Gated-delta functions that Transformers binds at import to these packages' GPU-only kernels.
GATED_DELTA = (
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
)
KERNEL_PACKAGES = ("fla", "causal_conv1d")


def _accepting(function: Any) -> Any:
    """``function`` called with only the keywords it takes, as Transformers' fallback wrapper calls it."""
    parameters = inspect.signature(function).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return function

    @functools.wraps(function)
    def call(*args: Any, **kwargs: Any) -> Any:
        return function(*args, **{k: v for k, v in kwargs.items() if k in parameters})

    return call


def cpu_reference_layers(root: nn.Module) -> int:
    """Bind the Qwen3.5 gated-delta layers under ``root`` to the PyTorch reference functions.

    Transformers binds those functions at import to the flash-linear-attention /
    causal-conv1d kernels when they are installed, and the kernels are GPU-only.
    Each layer gets its own forward whose globals name the reference functions;
    nothing global changes. Returns the number of layers rebound.
    """
    modeling = sys.modules.get(QWEN3_5_MODELING)
    layer_class = getattr(modeling, "Qwen3_5GatedDeltaNet", None)
    if layer_class is None or not any(name in sys.modules for name in KERNEL_PACKAGES):
        return 0
    references = {
        name: _accepting(inspect.unwrap(getattr(modeling, name)))
        for name in GATED_DELTA
        if callable(getattr(modeling, name, None))
    }
    forward = inspect.unwrap(layer_class.forward)
    reference_forward = types.FunctionType(
        forward.__code__,
        {**forward.__globals__, **references},
        forward.__name__,
        forward.__defaults__,
        forward.__closure__,
    )
    reference_forward.__kwdefaults__ = forward.__kwdefaults__
    layers = [m for m in root.modules() if isinstance(m, layer_class)]
    for layer in layers:
        layer.forward = types.MethodType(reference_forward, layer)
    return len(layers)


class CandidateHead(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.candidate_norm = nn.LayerNorm(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key = nn.Linear(hidden_size, head_dim, bias=False)
        self.query = nn.Linear(hidden_size, head_dim, bias=False)
        self.candidate_mlp = nn.Linear(hidden_size, head_dim, bias=True)
        self.query_mlp = nn.Linear(hidden_size, head_dim, bias=False)
        self.scalar = nn.Linear(head_dim, 1, bias=False)

    def forward(self, candidates, query):
        with torch.autocast(device_type=candidates.device.type, enabled=False):
            c = self.candidate_norm(candidates.float())
            q = self.query_norm(query.float())
            bilinear = (self.key(c) * self.query(q)[:, None, :]).sum(-1) / math.sqrt(
                self.head_dim
            )
            interaction = self.scalar(
                functional.gelu(self.candidate_mlp(c) + self.query_mlp(q)[:, None, :])
            ).squeeze(-1)
            return bilinear + interaction


class QwenDecision(nn.Module):
    def __init__(self, backbone: nn.Module, head: CandidateHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        input_ids,
        attention_mask,
        candidate_positions,
        candidate_mask,
        query_positions,
    ):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        batches = torch.arange(hidden.shape[0], device=hidden.device)
        candidates = hidden[batches[:, None], candidate_positions]
        query = hidden[batches, query_positions]
        scores = self.head(candidates, query).float()
        return scores.masked_fill(~candidate_mask, -float("inf"))


def head_parameters(hidden: int, head_dim: int) -> int:
    return 4 * hidden + 4 * hidden * head_dim + 2 * head_dim


class QwenRuntime:
    """Loaded backbone, head, tokenizer, temperatures and prompt policy of one decoder."""

    noul_default_false = NOUL_DEFAULT_FALSE
    noul_default_true = NOUL_DEFAULT_TRUE
    noul_explicit_null = "preserve_json_null"

    def __init__(
        self, model, tokenizer, temperatures, max_input_tokens, choice_null_description
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperatures = temperatures
        self.max_input_tokens = max_input_tokens
        self.choice_null_description = choice_null_description
        pad = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        if pad is None:
            raise ValueError("The tokenizer must define a PAD or EOS token")
        self.pad = pad

    @classmethod
    def load(
        cls,
        root: Path,
        descriptor: dict[str, Any],
        *,
        max_input_tokens: int,
        choice_null_description: str,
        device,
    ):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        metadata = json.loads(
            (root / descriptor["model_config"]).read_text(encoding="utf-8")
        )
        if metadata.get("prompt_version") != PROMPT_VERSION:
            raise ValueError("Not a pointer-v2 Decision 1.0 checkpoint")
        dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
        loaded = Qwen3_5TextModel.from_pretrained(
            str((root / descriptor["backbone"]["config"]).parent),
            dtype=dtype,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        backbone, info = loaded
        if any(
            info.get(key)
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError("Backbone tensors do not match the Qwen3.5 architecture")
        backbone.config.use_cache = False
        hidden = backbone.config.hidden_size
        head = CandidateHead(hidden, metadata["head_dim"])
        state = load_file(str(root / descriptor["decision_weights"]["decision_head"]))
        if any(tensor.dtype != torch.float32 for tensor in state.values()):
            raise ValueError("The candidate head must be FP32")
        head.load_state_dict(state, strict=True)
        model = QwenDecision(backbone, head)
        expected_text = metadata.get("text_parameter_count")
        loaded_text = sum(parameter.numel() for parameter in backbone.parameters())
        if expected_text is not None and loaded_text != expected_text:
            raise ValueError(
                f"Loaded {loaded_text:,} backbone parameters; expected {expected_text:,}"
            )
        if sum(p.numel() for p in head.parameters()) != head_parameters(
            hidden, metadata["head_dim"]
        ):
            raise ValueError("Unexpected candidate-head geometry")
        model.to(device).eval()
        if device.type == "cpu":
            cpu_reference_layers(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str((root / descriptor["tokenizer"]["json"]).parent),
            trust_remote_code=False,
        )
        temperatures = _temperatures(root, descriptor)
        return cls(
            model, tokenizer, temperatures, max_input_tokens, choice_null_description
        )

    def segments(self, row: Row) -> tuple[str, list[str]]:
        prefix = (
            f"Context:\n{content_text(row.state)}\n\n"
            f"Task type: {row.type}\n"
            f"Question:\n{content_text(row.instructions)}\n"
            "Options:"
        )
        options = []
        for candidate in row.candidates:
            description = candidate.description
            if (
                description is None
                and row.type == "choice"
                and self.choice_null_description == "render_key"
            ):
                description = candidate.key
            options.append(
                "\n<option>\n"
                + canonical_json({"key": candidate.key, "description": description})
                + "\n</option>"
            )
        return prefix, options

    def encode(self, row: Row, cache: dict[str, list[int]]) -> dict[str, Any]:
        def tokens(text):
            if text not in cache:
                cache[text] = list(
                    self.tokenizer.encode(text, add_special_tokens=False)
                )
            return cache[text]

        prefix, options = self.segments(row)
        ids = list(tokens(prefix))
        positions = []
        for option in options:
            part = tokens(option)
            if not part:
                raise ValueError("A candidate renders to no tokens")
            ids.extend(part)
            positions.append(len(ids) - 1)
        ids.extend(tokens(SUFFIX))
        if len(ids) > self.max_input_tokens:
            raise DecisionInputTooLongError(
                f"{row.question_id}: {len(ids)} tokens exceeds max_length="
                f"{self.max_input_tokens}; no truncation allowed"
            )
        return {"ids": ids, "positions": positions, "query": len(ids) - 1}

    def _check_precision(self, device) -> None:
        backbone = {parameter.dtype for parameter in self.model.backbone.parameters()}
        wanted = torch.float32 if device.type == "cpu" else torch.bfloat16
        head = {parameter.dtype for parameter in self.model.head.parameters()}
        if backbone != {wanted} or head != {torch.float32}:
            raise RuntimeError(
                "The model was cast or moved outside Decision1Model.to(); reload it"
            )

    def predict(self, rows: list[Row]) -> tuple[list[list[float]], list[int]]:
        """Probabilities per row in request order, and input tokens per row."""
        cache: dict[str, list[int]] = {}
        encoded = [self.encode(row, cache) for row in rows]
        device = next(self.model.parameters()).device
        self._check_precision(device)
        results = []
        with torch.inference_mode():
            for start in range(0, len(rows), PHYSICAL_BATCH):
                items = encoded[start : start + PHYSICAL_BATCH]
                length = (
                    (max(len(item["ids"]) for item in items) + PAD_MULTIPLE - 1)
                    // PAD_MULTIPLE
                ) * PAD_MULTIPLE
                width = max(len(item["positions"]) for item in items)
                input_ids = torch.full((len(items), length), self.pad, dtype=torch.long)
                mask = torch.zeros_like(input_ids)
                positions = torch.zeros((len(items), width), dtype=torch.long)
                candidate_mask = torch.zeros((len(items), width), dtype=torch.bool)
                for slot, item in enumerate(items):
                    input_ids[slot, : len(item["ids"])] = torch.tensor(item["ids"])
                    mask[slot, : len(item["ids"])] = 1
                    positions[slot, : len(item["positions"])] = torch.tensor(
                        item["positions"]
                    )
                    candidate_mask[slot, : len(item["positions"])] = True
                queries = torch.tensor([item["query"] for item in items])
                autocast = (
                    nullcontext()
                    if device.type == "cpu"
                    else torch.autocast(device.type, dtype=torch.bfloat16)
                )
                with autocast:
                    logits = self.model(
                        input_ids.to(device),
                        mask.to(device),
                        positions.to(device),
                        candidate_mask.to(device),
                        queries.to(device),
                    )
                staged = []
                for row, item, values in zip(
                    rows[start : start + PHYSICAL_BATCH], items, logits
                ):
                    values = values[: len(item["positions"])].float()
                    staged.append((values / self.temperatures[row.type]).softmax(-1))
                host = torch.cat(staged).tolist()
                offset = 0
                for item in items:
                    count = len(item["positions"])
                    values = host[offset : offset + count]
                    offset += count
                    if any(not math.isfinite(value) for value in values):
                        raise FloatingPointError("Non-finite Decision probabilities")
                    total = sum(values)
                    results.append([value / total for value in values])
        return results, [len(item["ids"]) for item in encoded]


def _temperatures(root: Path, descriptor: dict[str, Any]) -> dict[str, float]:
    calibration = descriptor.get("calibration") or {}
    if "temperature" in calibration:
        value = calibration["temperature"]
        temperatures = {kind: value for kind in ("choice", "noul", "score")}
    elif "temperature_file" in calibration:
        document = json.loads(
            (root / calibration["temperature_file"]).read_text(encoding="utf-8")
        )
        per_type = document.get("temperatures")
        if isinstance(per_type, dict) and set(per_type) == {"choice", "noul", "score"}:
            temperatures = dict(per_type)
        else:
            temperatures = {
                kind: document["temperature"] for kind in ("choice", "noul", "score")
            }
    else:
        raise ValueError("Decision 1.0 decoders need a calibration temperature")
    for value in temperatures.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Calibration temperatures must be finite and positive")
    return {kind: float(value) for kind, value in temperatures.items()}
