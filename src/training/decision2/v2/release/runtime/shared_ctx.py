"""Shared-context prefill for multi-question System One requests (opt-in).

The exact path encodes every question of a request as its own row, ``[state +
question + options]``, and runs the rows as one padded batch, so the shared
input is recomputed for every question. With ``share_context`` on, the longest
token prefix all rows share (cut before the first option endpoint, rounded down
to a multiple of ``align``) is computed once and every row's suffix continues
from it: full-attention layers attend to the prefix keys and values, gated-delta
layers start from the prefix-end recurrent state and convolution window. The
positions are those of the exact path, and the decision head reads the same
question and option tokens, so every head variant runs unchanged.

``mode="tree"`` packs the prefix and every suffix into one row and runs one
forward: suffix queries attend to the prefix and to their own suffix in two
attention kernels whose results are merged by their log-sum-exp, and the
gated-delta layers run the prefix once and the suffixes as variable-length
sequences from its final state. ``mode="cache"`` runs the prefix with a cache,
then the suffixes as a padded batch continuing from it (also the fallback when
the packed row exceeds the forward token budget).

The arithmetic is not the exact path's (other GEMM shapes, attention kernels and
reduction orders), so answers can move slightly. ``tau`` bounds that: every
answer whose top-two probability margin (Noul: ``|2p - 1|``) is below ``tau`` is
re-scored on the exact path, in its exact-path batch's padded length and mask
regime (``fallback="rows"``), or the whole request is (``fallback="request"``).
``min_questions`` and ``min_shared_tokens`` keep requests where sharing costs
more than it saves on the exact path.
"""

from __future__ import annotations

import copy
import importlib
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from functools import partial
from typing import Any

TESTED_TRANSFORMERS = ("5.17.",)
SUPPORTED_MODEL_TYPES = ("qwen3", "qwen3_5_text")
LAYER_TYPES = ("full_attention", "linear_attention")
FALLBACKS = ("rows", "request")
MODES = ("tree", "cache")
TREE_ATTENTION = "decision2_shared_tree"


@dataclass(frozen=True)
class SharePolicy:
    """When a request shares its prefix, and which answers go back to the exact path.

    ``align``: the prefix length is a multiple of it (default 1). In ``cache``
    mode, ``max_buckets`` > 1 splits the suffix rows by length into up to that
    many batches when that saves more than ``bucket_tokens`` padded tokens per
    extra batch.
    """

    tau: float = 0.0
    min_questions: int = 2
    min_shared_tokens: int = 0
    align: int = 1
    fallback: str = "rows"
    mode: str = "tree"
    max_buckets: int = 1
    bucket_tokens: int = 2048

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must be within [0, 1]")
        if self.min_questions < 2 or self.min_shared_tokens < 0:
            raise ValueError("min_questions must be >= 2 and min_shared_tokens >= 0")
        if self.align < 1:
            raise ValueError("align must be positive")
        if self.max_buckets < 1 or self.bucket_tokens < 0:
            raise ValueError("max_buckets must be >= 1 and bucket_tokens >= 0")
        if self.fallback not in FALLBACKS:
            raise ValueError(f"fallback must be one of {FALLBACKS}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")


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


class PrefixTokenizer:
    """``tokenizer.encode`` that reuses the tokens of a leading text already seen.

    The first long text is tokenized in full. A later text that agrees with it
    on its first D characters reuses its tokens up to the last boundary b <= D - 2
    that is both a token and a pre-token boundary of the first text, with ASCII
    on either side and no added-token text near it; only the rest is tokenized.
    The pre-tokenizer splits both texts alike up to b and encodes pieces
    independently, so the tokens are those of a full tokenization. The first
    reuse is also checked against one; a mismatch turns reuse off.
    """

    min_chars = 256

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.first: tuple[str, list[int], list[int], set[int]] | None = None
        self.cuts: dict[int, tuple[int, int] | None] = {}
        self.added = list(getattr(tokenizer, "get_added_vocab", dict)())
        self.width = max((len(token) for token in self.added), default=0)
        self.checked = False
        self.off = False
        self.reused = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)

    def _near_added(self, text: str, b: int) -> bool:
        window = text[max(0, b - self.width) : b + self.width]
        return any(token in window for token in self.added)

    def _start(self, text: str) -> list[int]:
        encoding = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        ids = list(encoding["input_ids"])
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        normalizer = getattr(backend, "normalizer", None)
        pre = getattr(backend, "pre_tokenizer", None)
        if pre is None or (normalizer and normalizer.normalize_str(text) != text):
            self.off = True
            return ids
        pieces = {end for _, (_, end) in pre.pre_tokenize_str(text)}
        ends = [end for _, end in encoding["offset_mapping"]]
        self.first = (text, ids, ends, pieces)
        return ids

    def _cut(self, common: int) -> tuple[int, int] | None:
        text, _, ends, pieces = self.first
        for count in range(len(ends), 0, -1):
            b = ends[count - 1]
            if b > common - 2 or b not in pieces:
                continue
            if not (text[b - 1].isascii() and text[b].isascii()):
                continue
            if self._near_added(text, b):
                continue
            return b, count
        return None

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs: Any):
        if add_special_tokens or kwargs or self.off or len(text) < self.min_chars:
            return self.tokenizer.encode(
                text, add_special_tokens=add_special_tokens, **kwargs
            )
        if self.first is None:
            return self._start(text)
        common = len(os.path.commonprefix([text, self.first[0]]))
        if common not in self.cuts:
            self.cuts[common] = self._cut(common)
        cut = self.cuts[common]
        if cut is None or self._near_added(text, cut[0]):
            return self.tokenizer.encode(text, add_special_tokens=False)
        b, count = cut
        ids = self.first[1][:count] + self.tokenizer.encode(
            text[b:], add_special_tokens=False
        )
        if not self.checked:
            self.checked = True
            full = self.tokenizer.encode(text, add_special_tokens=False)
            if ids != full:
                self.off = True
                return full
        self.reused += 1
        return ids


def tokenizer_for(tokenizer: Any, policy: SharePolicy | None, questions: int) -> Any:
    """The tokenizer a request's encodings use: shared-prefix reuse when sharing."""
    if policy is None or questions < policy.min_questions:
        return tokenizer
    return PrefixTokenizer(tokenizer)


def _padded(length: int) -> int:
    return -(-length // 8) * 8


def buckets(lengths: list[int], most: int, penalty: int) -> list[list[int]]:
    """Rows split by length into at most ``most`` batches, fewest padded tokens.

    Each extra batch costs ``penalty`` padded tokens (one more forward). Every
    batch lists its rows in request order.
    """
    widths = sorted({_padded(length) for length in lengths})
    rows = [sum(_padded(length) == width for length in lengths) for width in widths]
    n = len(widths)
    best = {(0, 0): (0, 0)}
    for parts in range(1, max(1, most) + 1):
        for end in range(1, n + 1):
            options = [
                (
                    best[(parts - 1, start)][0]
                    + sum(rows[start:end]) * widths[end - 1]
                    + (penalty if parts > 1 else 0),
                    start,
                )
                for start in range(end)
                if (parts - 1, start) in best and (start > 0 or parts == 1)
            ]
            if options:
                best[(parts, end)] = min(options)
    parts = min(
        (p for p in range(1, max(1, most) + 1) if (p, n) in best),
        key=lambda p: (best[(p, n)][0], p),
    )
    cuts, end = [], n
    while parts:
        start = best[(parts, end)][1]
        cuts.append(widths[start:end])
        end, parts = start, parts - 1
    return [
        [i for i, length in enumerate(lengths) if _padded(length) in set(cut)]
        for cut in cuts[::-1]
    ]


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


def _prefix_kv(cache: Any, torch: Any) -> dict[int, tuple[Any, Any]]:
    """Each attention layer's prefix keys and values, in the dtype SDPA computes in.

    Under BF16 autocast SDPA casts its keys and values to BF16 anyway; casting
    the prefix once here gives the same values without a per-row cast.
    """
    dtype = None
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
    kv = {}
    for index, layer in enumerate(cache.layers):
        keys = getattr(layer, "keys", None)
        if torch.is_tensor(keys) and keys.numel():
            values = layer.values
            if dtype is not None:
                keys, values = keys.to(dtype), values.to(dtype)
            kv[index] = (keys, values)
    return kv


def _suffix_cache(prefix: Any, kv: dict, rows: int, torch: Any) -> Any:
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
            keys, values = kv[layer_idx]
            return (
                torch.cat(
                    [keys.expand(rows, -1, -1, -1), key_states.to(keys.dtype)], dim=-2
                ),
                torch.cat(
                    [values.expand(rows, -1, -1, -1), value_states.to(values.dtype)],
                    dim=-2,
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


def _release(backend: Any) -> None:
    """Drop the fast path's cached Linear-input cast (fast.py) after a forward."""
    memo = getattr(getattr(backend, "fast", None), "memo", None)
    if memo is not None:
        memo.clear()


class Tree:
    """A request packed into one row: the shared prefix, then every suffix in order.

    ``rows``/``packed`` move suffix tokens between the packed row and a padded
    [questions, width] layout. Padding sits after each suffix and repeats one of
    its tokens: causal attention, convolution and recurrence never carry it
    into a real token, and the head never reads it.
    """

    def __init__(self, prefix: int, lengths: list[int], width: int, device, torch):
        self.torch = torch
        self.prefix = prefix
        self.lengths = lengths
        self.suffix = sum(lengths)
        starts = [0]
        for length in lengths[:-1]:
            starts.append(starts[-1] + length)
        index = torch.empty((len(lengths), width), dtype=torch.long)
        valid = torch.zeros((len(lengths), width), dtype=torch.bool)
        for row, (start, length) in enumerate(zip(starts, lengths)):
            index[row, :length] = torch.arange(start, start + length)
            index[row, length:] = start + length - 1
            valid[row, :length] = True
        self.index = index.to(device)
        # Index (not mask) gathers: a boolean mask would sync the host on every use.
        self.flat = valid.reshape(-1).nonzero().squeeze(1).to(device)
        ends = [start + length for start, length in zip(starts, lengths)]
        self.cu_seqlens_cpu = torch.tensor([0, *ends], dtype=torch.long)
        self.cu_seqlens = self.cu_seqlens_cpu.to(device)
        self.positions = torch.cat(
            [torch.arange(prefix)]
            + [torch.arange(prefix, prefix + length) for length in lengths]
        )[None].to(device)
        self.prefix_mask = None

    def rows(self, packed: Any) -> Any:
        """[suffix tokens, ...] -> [questions, width, ...]."""
        return packed[self.index]

    def packed(self, rows: Any) -> Any:
        """[questions, width, ...] -> [suffix tokens, ...]."""
        return rows.reshape(-1, *rows.shape[2:])[self.flat]


def _attend(query: Any, key: Any, value: Any, causal: bool, scale: float, torch: Any):
    """Attention output and its log-sum-exp per query ([B, H, L, D], [B, H, L])."""
    if query.is_cuda:
        out = torch.ops.aten._scaled_dot_product_efficient_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            None,
            True,
            0.0,
            causal,
            scale=scale,
        )
        return out[0], out[1][..., : query.shape[2]]
    scores = (query.float() @ key.float().transpose(-1, -2)) * scale
    if causal:
        q, k = scores.shape[-2:]
        keep = torch.ones(q, k, dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~keep, -float("inf"))
    lse = scores.logsumexp(-1)
    out = (scores - lse[..., None]).exp() @ value.float()
    return out.to(query.dtype), lse


def _tree_attention(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    """SDPA over a packed tree: prefix causal; each suffix over the prefix and itself.

    A suffix query's attention over [prefix; own suffix] is merged from its
    attention over each part by their log-sum-exp.
    """
    tree = attention_mask
    torch = tree.torch
    groups = getattr(module, "num_key_value_groups", 1)
    if groups > 1:
        b, h, n, d = key.shape
        key = key[:, :, None].expand(b, h, groups, n, d).reshape(b, h * groups, n, d)
        value = value[:, :, None].expand(b, h, groups, n, d)
        value = value.reshape(b, h * groups, n, d)
    dtype = query.dtype
    if query.is_cuda and torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    q, k, v = query.to(dtype), key.to(dtype), value.to(dtype)
    p = tree.prefix
    head = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, :p],
        k[:, :, :p],
        v[:, :, :p],
        attn_mask=tree.prefix_mask,
        is_causal=tree.prefix_mask is None,
        scale=scale,
    )
    out_prefix, lse_prefix = _attend(
        q[:, :, p:], k[:, :, :p], v[:, :, :p], False, scale, torch
    )

    def rows(x: Any) -> Any:
        return tree.rows(x[0].transpose(0, 1)).permute(0, 2, 1, 3)

    out_own, lse_own = _attend(
        rows(q[:, :, p:]), rows(k[:, :, p:]), rows(v[:, :, p:]), True, scale, torch
    )
    out_own = tree.packed(out_own.transpose(1, 2))
    lse_own = tree.packed(lse_own.transpose(1, 2))
    # softmax over [prefix; own] = the two parts weighted by sigmoid of their lse gap
    weight = torch.sigmoid(lse_prefix[0].transpose(0, 1) - lse_own)[..., None]
    tail = torch.lerp(out_own.float(), out_prefix[0].transpose(0, 1).float(), weight)
    out = torch.cat([head[0].transpose(0, 1), tail.to(dtype)], dim=0)
    return out[None], None


def _tree_gated_delta(
    m: Any,
    hidden_states: Any,
    cache_params: Any = None,
    attention_mask: Any = None,
    **_,
) -> Any:
    """``Qwen3_5GatedDeltaNet.forward`` over a packed tree (no padding in the row).

    The prefix runs from a zero state; every suffix runs from the prefix-end
    recurrent state, its convolution window starting with the prefix's last
    inputs.
    """
    tree = attention_mask
    torch = tree.torch
    modeling = importlib.import_module(type(m).__module__)
    _, length, _ = hidden_states.shape
    p = tree.prefix
    mixed = m.in_proj_qkv(hidden_states)
    z = m.in_proj_z(hidden_states).reshape(1, length, -1, m.head_v_dim)
    b = m.in_proj_b(hidden_states)
    a = m.in_proj_a(hidden_states)
    weight, bias, window = m.conv1d.weight.squeeze(1), m.conv1d.bias, m.conv_kernel_size
    head = modeling.causal_conv1d_fn(
        mixed[:, :p].transpose(1, 2), weight, bias, activation=m.activation
    )
    rows = torch.cat(
        [
            mixed[:, p - (window - 1) : p].expand(len(tree.lengths), -1, -1),
            tree.rows(mixed[0, p:]),
        ],
        dim=1,
    )
    tail = modeling.causal_conv1d_fn(
        rows.transpose(1, 2), weight, bias, activation=m.activation
    )[:, :, window - 1 :]
    mixed = torch.cat(
        [head[0].transpose(0, 1), tree.packed(tail.transpose(1, 2))], dim=0
    )[None]
    query, key, value = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
    query = query.reshape(1, length, -1, m.head_k_dim)
    key = key.reshape(1, length, -1, m.head_k_dim)
    value = value.reshape(1, length, -1, m.head_v_dim)
    beta = b.sigmoid()
    g = -m.A_log.float().exp() * torch.nn.functional.softplus(a.float() + m.dt_bias)
    if m.num_v_heads // m.num_k_heads > 1:
        query = query.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
        key = key.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
    rule = modeling.torch_chunk_gated_delta_rule
    out_prefix, state = rule(
        query[:, :p],
        key[:, :p],
        value[:, :p],
        g=g[:, :p],
        beta=beta[:, :p],
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    start = state.expand(len(tree.lengths), *state.shape[1:]).contiguous()
    if _varlen(rule):
        out_suffix, _ = rule(
            query[:, p:],
            key[:, p:],
            value[:, p:],
            g=g[:, p:],
            beta=beta[:, p:],
            initial_state=start,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=tree.cu_seqlens,
            cu_seqlens_cpu=tree.cu_seqlens_cpu,
        )
    else:
        out_rows, _ = rule(
            tree.rows(query[0, p:]),
            tree.rows(key[0, p:]),
            tree.rows(value[0, p:]),
            g=tree.rows(g[0, p:]),
            beta=tree.rows(beta[0, p:]),
            initial_state=start,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        out_suffix = tree.packed(out_rows)[None]
    core = torch.cat([out_prefix, out_suffix], dim=1)
    core = m.norm(core.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim))
    return m.out_proj(core.reshape(1, length, -1))


def _varlen(rule: Any) -> bool:
    """Whether the gated-delta kernel Transformers bound takes ``cu_seqlens`` (FLA)."""
    try:
        import inspect

        bound = inspect.getclosurevars(rule).nonlocals.get("implementation", rule)
    except (TypeError, ValueError):
        bound = rule
    return getattr(bound, "__module__", "").startswith("fla")


@contextmanager
def _as_tree(backbone: Any, tree: Any, ids: Any, layer_types: set[str]):
    """Every ``backbone`` call inside runs the packed tree and returns suffix rows."""
    from transformers import AttentionInterface

    if TREE_ATTENTION not in AttentionInterface._global_mapping:
        AttentionInterface.register(TREE_ATTENTION, _tree_attention)
    config = _core(backbone).config
    gdn = [m for m in backbone.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet"]
    # Fused decoder-layer forwards (fast.py) read tensor masks only: run the originals.
    fused = [m for m in backbone.modules() if hasattr(m, "_decision2_fused")]
    swapped = [(m, m.__dict__.get("forward")) for m in fused]

    def unpack(hidden: Any) -> Any:
        return tree.rows(hidden[0, tree.prefix :])

    def pre(module: Any, args: Any, kwargs: dict[str, Any]) -> Any:
        return args, {
            **kwargs,
            "input_ids": ids,
            "attention_mask": {name: tree for name in layer_types},
            "position_ids": tree.positions,
            "use_cache": False,
        }

    def post(module: Any, args: Any, kwargs: dict[str, Any], output: Any) -> Any:
        output.last_hidden_state = unpack(output.last_hidden_state)
        if getattr(output, "hidden_states", None) is not None:
            output.hidden_states = tuple(unpack(h) for h in output.hidden_states)
        return output

    saved = config._attn_implementation
    forwards = [m.__dict__.get("forward") for m in gdn]
    handles = [
        backbone.register_forward_pre_hook(pre, with_kwargs=True),
        backbone.register_forward_hook(post, with_kwargs=True),
    ]
    for m in gdn:
        m.forward = partial(_tree_gated_delta, m)
    for m in fused:
        m.forward = m._decision2_fused["original"]
    config._attn_implementation = TREE_ATTENTION
    try:
        yield
    finally:
        config._attn_implementation = saved
        for m, forward in [*zip(gdn, forwards), *swapped]:
            if forward is None:
                del m.forward
            else:
                m.forward = forward
        for handle in handles:
            handle.remove()


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
        _release(backend)
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


def _suffix_batch(encoded: list, group: list[int], prefix: int, pad_id: int) -> dict:
    """The suffix rows of ``group`` collated, positions shifted to the suffix."""
    from ._vendor.dev2model.decision_model import collate

    return collate(
        [
            {
                **encoded[index],
                "ids": encoded[index]["ids"][prefix:],
                "candidate_positions": [
                    p - prefix for p in encoded[index]["candidate_positions"]
                ],
                "query_position": encoded[index]["query_position"] - prefix,
            }
            for index in group
        ],
        pad_id,
    )


def _prefix_mask(lengths: list[int], prefix: int, backend: Any) -> Any:
    """The prefix's causal mask; explicit when the exact batch is padded."""
    if all(length == _padded(max(lengths)) for length in lengths):
        return None
    torch = backend.torch
    return torch.ones(prefix, prefix, dtype=torch.bool, device=backend.device).tril()[
        None, None
    ]


def _tree_logits(backend: Any, encoded: list, prefix: int, pad_id: int) -> list:
    torch = backend.torch
    model = backend.model
    config = _core(model.backbone).config
    layer_types = set(getattr(config, "layer_types", None) or ["full_attention"])
    group = list(range(len(encoded)))
    batch = _suffix_batch(encoded, group, prefix, pad_id)
    ids = list(encoded[0]["ids"][:prefix])
    for item in encoded:
        ids.extend(item["ids"][prefix:])
    tree = Tree(
        prefix,
        [len(item["ids"]) - prefix for item in encoded],
        batch["input_ids"].shape[1],
        backend.device,
        torch,
    )
    tree.prefix_mask = _prefix_mask(
        [len(item["ids"]) for item in encoded], prefix, backend
    )
    batch = {
        key: value.to(backend.device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    ids = torch.tensor([ids], dtype=torch.long, device=backend.device)
    with torch.inference_mode(), _context(backend):
        with _as_tree(model.backbone, tree, ids, layer_types):
            output = model(**batch)
    _release(backend)
    if len(output) != len(group):
        raise RuntimeError("Model returned the wrong number of question answers")
    return list(output.float().cpu())


def _cache_logits(
    backend: Any, encoded: list, prefix: int, pad_id: int, policy: SharePolicy
) -> tuple[list, int]:
    from transformers.cache_utils import DynamicCache

    from .qwen import micro_batches

    torch = backend.torch
    model = backend.model
    backbone = model.backbone
    config = _core(backbone).config
    layer_types = set(getattr(config, "layer_types", None) or ["full_attention"])
    lengths = [len(item["ids"]) for item in encoded]
    with torch.inference_mode(), _context(backend):
        cache = DynamicCache(config=config)
        ids = torch.tensor(
            [encoded[0]["ids"][:prefix]], dtype=torch.long, device=backend.device
        )
        causal = _prefix_mask(lengths, prefix, backend)
        mask = None
        if causal is not None:
            mask = {
                name: causal if name == "full_attention" else None
                for name in layer_types
            }
        backbone(
            input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True
        )
        kv = _prefix_kv(cache, torch)
    logits: list = [None] * len(encoded)
    suffix = [length - prefix for length in lengths]
    groups = [
        [bucket[index] for index in group]
        for bucket in buckets(suffix, policy.max_buckets, policy.bucket_tokens)
        for group in micro_batches([suffix[i] for i in bucket], backend.batch_tokens)
    ]
    for group in groups:
        batch = _suffix_batch(encoded, group, prefix, pad_id)
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
            with _with_cache(backbone, _suffix_cache(cache, kv, len(group), torch)):
                output = model(**batch)
        _release(backend)
        if len(output) != len(group):
            raise RuntimeError("Model returned the wrong number of question answers")
        for index, values in zip(group, output.float().cpu()):
            logits[index] = values
    return logits, len(groups)


def shared_logits(
    backend: Any, jobs: list, pad_id: int, policy: SharePolicy
) -> list | None:
    """Per-question logits with the shared prefix run once; None: run it exactly.

    ``backend.share_stats`` records what the last request did.
    """
    encoded = [item for _, _, item in jobs]
    stats: dict[str, Any] = {"questions": len(jobs), "shared": False}
    backend.share_stats = stats
    if len(jobs) < policy.min_questions:
        stats["reason"] = "too few questions"
        return None
    reason = _supported(backend)
    if reason:
        stats["reason"] = reason
        return None
    prefix = shared_prefix(encoded, policy.align)
    stats["prefix_tokens"] = prefix
    if prefix < 4 or (len(jobs) - 1) * prefix < policy.min_shared_tokens:
        stats["reason"] = "too little shared input"
        return None
    packed = prefix + sum(len(item["ids"]) - prefix for item in encoded)
    budget = backend.batch_tokens
    if policy.mode == "tree" and (budget is None or _padded(packed) <= budget):
        logits = _tree_logits(backend, encoded, prefix, pad_id)
        stats.update(shared=True, mode="tree", tokens=packed)
    else:
        logits, batches = _cache_logits(backend, encoded, prefix, pad_id, policy)
        stats.update(shared=True, mode="cache", suffix_batches=batches)
    stats["rescored"] = 0
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
