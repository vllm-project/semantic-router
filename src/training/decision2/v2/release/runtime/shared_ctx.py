"""Shared-context prefill for multi-question System One requests (opt-in).

The exact path encodes every question of a request as its own row, ``[state +
question + options]``, and runs the rows as one padded batch, so the shared
input is recomputed for every question. With ``share_context`` on, the longest
token prefix all rows share (cut before the first option endpoint and rounded
down to a multiple of ``align``) runs once, as a batch of one; every row's
suffix then continues from it in one padded batch. Full-attention layers attend
to the prefix keys and values; gated-delta layers start from the prefix-end
recurrent and convolution states. With ``align`` a multiple of the gated-delta
chunk (64 tokens), the suffix chunks are the exact path's chunks. Positions and
masks are those of a cache continuation, and the decision head reads the same
question and option tokens, so every head variant runs unchanged.

The arithmetic is not the exact path's (other GEMM shapes, attention kernels and
reduction orders), so answers can move slightly. ``tau`` bounds that: every
answer whose top-two probability margin (Noul: ``|2p - 1|``) is below ``tau`` is
re-scored on the exact path, in its exact-path batch's padded length and mask
regime (``fallback="rows"``), or the whole request is (``fallback="request"``).
``min_questions`` and ``min_shared_tokens`` keep requests where one extra
forward costs more than the sharing saves on the exact path.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from typing import Any

TESTED_TRANSFORMERS = ("5.17.",)
SUPPORTED_MODEL_TYPES = ("qwen3", "qwen3_5_text")
LAYER_TYPES = ("full_attention", "linear_attention")
FALLBACKS = ("rows", "request")


@dataclass(frozen=True)
class SharePolicy:
    """When a request shares its prefix, and which answers go back to the exact path."""

    tau: float = 0.0
    min_questions: int = 2
    min_shared_tokens: int = 0
    align: int = 64
    fallback: str = "rows"

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must be within [0, 1]")
        if self.min_questions < 2 or self.min_shared_tokens < 0:
            raise ValueError("min_questions must be >= 2 and min_shared_tokens >= 0")
        if self.align < 1 or self.align % 64:
            raise ValueError("align must be a positive multiple of 64")
        if self.fallback not in FALLBACKS:
            raise ValueError(f"fallback must be one of {FALLBACKS}")


DEFAULT_POLICY = SharePolicy()


def resolve(value: Any) -> SharePolicy | None:
    """The policy a ``share_context`` value asks for; None runs the exact path."""
    if value is None or value is False:
        return None
    if value is True:
        return DEFAULT_POLICY
    if isinstance(value, SharePolicy):
        return value
    if isinstance(value, dict):
        known = {field.name for field in fields(SharePolicy)}
        if set(value) - known:
            raise ValueError(
                f"unknown share_context keys: {sorted(set(value) - known)}"
            )
        return SharePolicy(**value)
    raise ValueError("share_context must be a bool, a dict or a SharePolicy")


def shared_prefix(encoded: list[dict[str, Any]], align: int) -> int:
    """Tokens all rows start with, before any option endpoint, rounded to ``align``."""
    ids = [item["ids"] for item in encoded]
    low, high = min(ids), max(ids)
    common = next(
        (i for i, (a, b) in enumerate(zip(low, high)) if a != b),
        min(len(low), len(high)),
    )
    limit = min(min(item["candidate_positions"]) for item in encoded)
    return min(common, limit) // align * align


def _padded(length: int) -> int:
    return -(-length // 8) * 8


def _core(backbone: Any) -> Any:
    get = getattr(backbone, "get_base_model", None)
    return get() if callable(get) else backbone


def _supported(backend: Any) -> str | None:
    """Why this backend cannot share a prefix; None when it can."""
    import transformers

    if not transformers.__version__.startswith(TESTED_TRANSFORMERS):
        return f"transformers {transformers.__version__} is not verified"
    config = _core(backend.model.backbone).config
    if getattr(config, "model_type", None) not in SUPPORTED_MODEL_TYPES:
        return f"backbone {getattr(config, 'model_type', None)} is not supported"
    if getattr(config, "_attn_implementation", None) != "sdpa":
        return "needs SDPA attention"
    if not set(getattr(config, "layer_types", None) or ["full_attention"]) <= set(
        LAYER_TYPES
    ):
        return "needs full / linear attention layers only"
    return None


def _suffix_cache(prefix: Any, rows: int, torch: Any) -> Any:
    """The prefix cache as ``rows`` suffix rows see it; the suffix stores nothing."""
    from transformers.cache_utils import Cache

    class Expanding(dict):
        """Recurrent states copied per row on read (one layer's copies at a time)."""

        def __getitem__(self, key: Any) -> Any:
            value = dict.__getitem__(self, key)
            if value is None:
                return None
            return value.expand(rows, *value.shape[1:]).contiguous()

    class SuffixCache(Cache):
        def update(self, key_states, value_states, layer_idx, *args, **kwargs):
            layer = self.layers[layer_idx]
            return (
                torch.cat([layer.keys.expand(rows, -1, -1, -1), key_states], dim=-2),
                torch.cat(
                    [layer.values.expand(rows, -1, -1, -1), value_states], dim=-2
                ),
            )

        def update_conv_state(self, conv_states, layer_idx, state_idx=0, **kwargs):
            state = self.layers[layer_idx].conv_states[state_idx]
            return torch.cat([state.expand(rows, -1, -1), conv_states], dim=-1)

        def update_recurrent_state(
            self, recurrent_states, layer_idx, state_idx=0, **kwargs
        ):
            return recurrent_states

    layers = []
    for layer in prefix.layers:
        view = copy.copy(layer)
        states = getattr(layer, "recurrent_states", None)
        if isinstance(states, dict):
            view.recurrent_states = Expanding(states)
        layers.append(view)
    return SuffixCache(layers=layers)


@contextmanager
def _with_cache(backbone: Any, cache: Any):
    """Every call of ``backbone`` inside continues from ``cache``."""

    def inject(module: Any, args: Any, kwargs: dict[str, Any]) -> Any:
        return args, {**kwargs, "past_key_values": cache, "use_cache": True}

    handle = backbone.register_forward_pre_hook(inject, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def _context(backend: Any) -> Any:
    torch = backend.torch
    if backend.device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _masks(layer_types: set[str], attention_mask: Any, torch: Any) -> dict[str, Any]:
    """The masks Transformers builds for a right-padded batch."""
    length = attention_mask.shape[1]
    causal = torch.ones(
        length, length, dtype=torch.bool, device=attention_mask.device
    ).tril()
    masks = {}
    if "full_attention" in layer_types:
        masks["full_attention"] = (
            causal[None, None] & attention_mask.bool()[:, None, None, :]
        )
    if "linear_attention" in layer_types:
        masks["linear_attention"] = attention_mask
    return masks


def exact_rows(
    backend: Any, jobs: list, rows: list[int], pad_id: int
) -> dict[int, Any]:
    """Logits of ``rows`` as the exact path computes them.

    Each row runs in its exact-path micro-batch's padded length and mask regime,
    one forward per micro-batch that holds any of ``rows``.
    """
    from ._vendor.dev2model.decision_model import collate
    from .qwen import micro_batches

    torch = backend.torch
    config = _core(backend.model.backbone).config
    layer_types = set(getattr(config, "layer_types", None) or ["full_attention"])
    lengths = [len(encoded["ids"]) for _, _, encoded in jobs]
    wanted = set(rows)
    found: dict[int, Any] = {}
    for group in micro_batches(lengths, backend.batch_tokens):
        subset = [index for index in group if index in wanted]
        if not subset:
            continue
        width = _padded(max(lengths[index] for index in group))
        batch = collate([jobs[index][2] for index in subset], pad_id)
        extra = width - batch["input_ids"].shape[1]
        if extra:
            batch["input_ids"] = torch.nn.functional.pad(
                batch["input_ids"], (0, extra), value=pad_id
            )
            batch["attention_mask"] = torch.nn.functional.pad(
                batch["attention_mask"], (0, extra), value=0
            )
        batch = {
            key: value.to(backend.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        if any(lengths[index] != width for index in group):
            batch["attention_mask"] = _masks(
                layer_types, batch["attention_mask"], torch
            )
        with torch.inference_mode(), _context(backend):
            output = backend.model(**batch)
        for index, values in zip(subset, output.float().cpu()):
            found[index] = values
    return found


def margins(backend: Any, jobs: list, logits: list) -> list[float]:
    """Top-two probability margin of each answer as reported; -1 when invalid."""
    from ._vendor.dev2model.infer import normalized_answer

    if backend.score_bias is not None:
        from ._vendor.dev2model.score_bias import apply as apply_score_bias
    out = []
    for (_, row, encoded), values in zip(jobs, logits):
        try:
            values = values[: len(encoded["keys"])].float().cpu().tolist()
            if backend.score_bias is not None and row["task_type"] == "score":
                values = apply_score_bias(
                    backend.score_bias, values, len(row["options"])
                )
            answer = normalized_answer(
                row["task_type"],
                encoded["keys"],
                values,
                backend.temperatures[row["task_type"]],
            )
        except ValueError:
            out.append(-1.0)
            continue
        if row["task_type"] == "noul":
            out.append(abs(2.0 * answer["noul"] - 1.0))
        else:
            top = sorted(answer["probabilities"].values(), reverse=True)
            out.append(top[0] - top[1])
    return out


def shared_logits(
    backend: Any, jobs: list, pad_id: int, policy: SharePolicy
) -> list | None:
    """Per-question logits with the shared prefix run once; None: run it exactly.

    ``backend.share_stats`` records what the last request did.
    """
    from ._vendor.dev2model.decision_model import collate
    from .qwen import micro_batches

    torch = backend.torch
    encoded = [item for _, _, item in jobs]
    stats: dict[str, Any] = {"questions": len(jobs), "shared": False}
    backend.share_stats = stats
    if len(jobs) < policy.min_questions:
        stats["reason"] = "too few questions"
        return None
    prefix = shared_prefix(encoded, policy.align)
    stats["prefix_tokens"] = prefix
    if prefix == 0 or (len(jobs) - 1) * prefix < policy.min_shared_tokens:
        stats["reason"] = "too little shared input"
        return None
    reason = _supported(backend)
    if reason:
        stats["reason"] = reason
        return None
    model = backend.model
    backbone = model.backbone
    config = _core(backbone).config
    layer_types = set(getattr(config, "layer_types", None) or ["full_attention"])
    lengths = [len(item["ids"]) for item in encoded]
    from transformers.cache_utils import DynamicCache

    with torch.inference_mode(), _context(backend):
        cache = DynamicCache(config=config)
        ids = torch.tensor(
            [encoded[0]["ids"][:prefix]], dtype=torch.long, device=backend.device
        )
        mask = None
        if any(length != _padded(max(lengths)) for length in lengths):
            # A padded exact batch runs SDPA with an explicit mask; so does the prefix.
            causal = torch.ones(
                prefix, prefix, dtype=torch.bool, device=backend.device
            ).tril()
            mask = {
                name: causal[None, None] if name == "full_attention" else None
                for name in layer_types
            }
        backbone(
            input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True
        )
    logits: list = [None] * len(jobs)
    groups = micro_batches(
        [length - prefix for length in lengths], backend.batch_tokens
    )
    for group in groups:
        items = [
            {
                **encoded[index],
                "ids": encoded[index]["ids"][prefix:],
                "candidate_positions": [
                    p - prefix for p in encoded[index]["candidate_positions"]
                ],
                "query_position": encoded[index]["query_position"] - prefix,
            }
            for index in group
        ]
        batch = collate(items, pad_id)
        batch["attention_mask"] = torch.cat(
            [
                torch.ones(len(group), prefix, dtype=batch["attention_mask"].dtype),
                batch["attention_mask"],
            ],
            dim=1,
        )
        batch = {
            key: value.to(backend.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with torch.inference_mode(), _context(backend):
            with _with_cache(backbone, _suffix_cache(cache, len(group), torch)):
                output = model(**batch)
        if len(output) != len(group):
            raise RuntimeError("Model returned the wrong number of question answers")
        for index, values in zip(group, output.float().cpu()):
            logits[index] = values
    stats.update(shared=True, suffix_batches=len(groups), rescored=0)
    if policy.tau > 0:
        low = [
            i for i, m in enumerate(margins(backend, jobs, logits)) if m < policy.tau
        ]
        stats["rescored"] = len(low)
        if low and policy.fallback == "request":
            stats["shared"] = False
            stats["reason"] = "low-margin answer"
            return None
        if low:
            for index, values in exact_rows(backend, jobs, low, pad_id).items():
                logits[index] = values
    return logits
