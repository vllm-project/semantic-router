"""Exact GPU fast path of the Qwen runtime: HIP graphs, kernel trims and fused kernels.

Every piece reproduces the released eager forward (BF16 autocast, BF16-resident
Linear weights, FP32 residual stream, norms and head) bit for bit, so answers
are unchanged; it only removes kernel launches and host work:

- **Graphs.** The backbone forward of each exact padded shape (rows, padded
  length, whether any row is padded) is captured once as a HIP graph, on the
  shape's second use, and replayed afterwards. The per-layer-type attention
  masks are built inside the graph from host-known lengths, as Transformers
  builds them for a right-padded batch, so nothing reads the mask back. Shapes
  above ``MAX_GRAPH_TOKENS`` padded tokens run eagerly; at most ``MAX_GRAPHS``
  graphs (and ``MAX_GRAPH_OUTPUT_BYTES`` of retained outputs) stay cached, least
  recently used first out.
- **Trims.** A Linear input feeding several BF16-resident Linear layers is cast
  to BF16 once instead of once per layer (the same cast autocast performs), and
  the zero-centred RMSNorm's ``1 + w`` is computed once instead of per call.
- **Lean LoRA.** An unmerged vanilla LoRA layer (one adapter, no bias, no
  dropout, power-of-two scaling) holds its factors in BF16, the values autocast
  multiplies with, folds the scaling into B (exact for a power of two) and casts
  its input once: 5 kernels instead of PEFT's 9, the same products and sums.
  Anything else stays on PEFT.
- **Fused kernels** (``fast_kernels``, gfx942 only): the element-wise ops of
  each Qwen3.5 decoder layer in six Triton kernels, and of each Qwen3 decoder
  layer in four, that round exactly like the ATen and causal-conv1d kernels
  they replace. GEMMs, attention and the gated-delta chunk kernel are unchanged.

The fast path is enabled only where it was verified to be exact: a ROCm GPU and
Transformers ``TESTED_TRANSFORMERS``; elsewhere the runtime keeps the eager
forward. ``graphs=False`` / ``kernels=False`` (or ``DECISION2_GRAPHS=0`` /
``DECISION2_KERNELS=0`` / ``DECISION2_FAST=0``) turn pieces off.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import os
import types
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

TESTED_TRANSFORMERS = ("5.17.",)
MAX_GRAPH_TOKENS = 65536
MAX_GRAPHS = 512
MAX_GRAPH_OUTPUT_BYTES = 4 << 30
CAPTURE_AFTER = 2
MASK_TYPES = ("full_attention", "linear_attention")
MASK_ALIGN = 16


def _switched_off(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("0", "false", "no", "off")


def _core(backbone: Any) -> Any:
    """The Transformers model under a PEFT wrapper (or the backbone itself)."""
    get = getattr(backbone, "get_base_model", None)
    return get() if callable(get) else backbone


class CastMemo:
    """The BF16 copy of the last FP32 Linear input, reused while the same tensor feeds the next layer."""

    def __init__(self, torch: Any):
        self.torch = torch
        self.source: Any = None
        self.version: Any = None
        self.value: Any = None

    def bf16(self, x: Any) -> Any:
        torch = self.torch
        if x.dtype != torch.float32:
            return x
        version = None if x.is_inference() else x._version
        if self.source is x and self.version == version:
            return self.value
        value = x.to(torch.bfloat16)
        self.source, self.version, self.value = x, version, value
        return value

    def clear(self) -> None:
        self.source = self.version = self.value = None


def shared_cast_linears(backbone: Any, memo: CastMemo, torch: Any) -> int:
    """BF16-resident ``nn.Linear`` layers cast an FP32 input through ``memo`` under autocast."""
    linear = torch.nn.functional.linear
    count = 0
    for layer in backbone.modules():
        if type(layer) is not torch.nn.Linear or layer.weight.dtype != torch.bfloat16:
            continue
        if layer.bias is not None and layer.bias.dtype != torch.bfloat16:
            continue

        def forward(x: Any, _w: Any = layer.weight, _b: Any = layer.bias) -> Any:
            if x.is_cuda and torch.is_autocast_enabled("cuda"):
                return linear(memo.bf16(x), _w, _b)
            return linear(x, _w, _b)

        layer.forward = forward
        count += 1
    return count


def rmsnorm_plus_one(backbone: Any, torch: Any) -> int:
    """Zero-centred RMSNorm layers (Qwen3.5) multiply by a cached ``1 + w``."""
    count = 0
    for norm in backbone.modules():
        if type(norm).__name__ != "Qwen3_5RMSNorm":
            continue
        with torch.no_grad():
            weight = 1.0 + norm.weight.float()

        def forward(x: Any, _norm: Any = norm, _w: Any = weight) -> Any:
            return (_norm._norm(x.float()) * _w).type_as(x)

        norm.forward = forward
        count += 1
    return count


def lean_lora(backbone: Any, memo: CastMemo, torch: Any) -> dict[str, int]:
    """Lean forward for every vanilla unmerged LoRA Linear; the rest stay on PEFT."""
    try:
        from peft.tuners.lora import Linear as LoraLinear
    except ImportError:
        return {"lean": 0, "peft": 0}
    linear = torch.nn.functional.linear
    lean = peft = 0
    for module in backbone.modules():
        if not isinstance(module, LoraLinear):
            continue
        weights = _lean_weights(module, torch)
        if weights is None:
            peft += 1
            continue
        base, factor_a, factor_b = weights

        def forward(
            x: Any, *args: Any, _w: Any = base, _a: Any = factor_a, _b: Any = factor_b,
            _peft: Any = module.forward, **kwargs: Any,
        ) -> Any:  # fmt: skip
            if args or kwargs or not (x.is_cuda and torch.is_autocast_enabled("cuda")):
                return _peft(x, *args, **kwargs)
            xb = memo.bf16(x)
            return linear(xb, _w) + linear(linear(xb, _a), _b)

        module.forward = forward
        lean += 1
    return {"lean": lean, "peft": peft}


def _lean_weights(module: Any, torch: Any) -> tuple[Any, Any, Any] | None:
    """(base W, A, scaling * B) in BF16 when the lean form is exact, else None."""
    adapters = list(getattr(module, "active_adapters", []) or [])
    if len(adapters) != 1 or getattr(module, "merged", False):
        return None
    if getattr(module, "disable_adapters", False):
        return None
    (name,) = adapters
    if name not in module.lora_A or name not in module.lora_B:
        return None
    if (getattr(module, "use_dora", {}) or {}).get(name):
        return None
    if name in (getattr(module, "lora_variant", {}) or {}):
        return None
    if (getattr(module, "lora_bias", None) or {}).get(name):
        return None
    dropout = module.lora_dropout[name]
    if not isinstance(dropout, torch.nn.Identity) and dropout.training:
        return None
    base = module.get_base_layer()
    if type(base) is not torch.nn.Linear or base.bias is not None:
        return None
    if base.weight.dtype != torch.bfloat16:
        return None
    a, b = module.lora_A[name], module.lora_B[name]
    if type(a) is not torch.nn.Linear or type(b) is not torch.nn.Linear:
        return None
    if a.bias is not None or b.bias is not None:
        return None
    scaling = float(module.scaling[name])
    if scaling <= 0 or not math.log2(scaling).is_integer():
        return None
    with torch.no_grad():
        factor_a = a.weight.detach().to(torch.bfloat16)
        rounded_b = b.weight.detach().to(torch.bfloat16)
        factor_b = (rounded_b.float() * scaling).to(torch.bfloat16)
        if not torch.equal(factor_b.float(), rounded_b.float() * scaling):
            return None
    return base.weight, factor_a, factor_b


def _bound(function: Any) -> Any:
    """The implementation a Transformers hub-kernel wrapper resolved at import (or None)."""
    try:
        return inspect.getclosurevars(function).nonlocals.get("implementation")
    except (TypeError, ValueError):
        return None


def fused_layers(backbone: Any, torch: Any) -> tuple[int, str | None]:
    """Fused forwards for the Qwen3.5 / Qwen3 decoder layers of ``backbone``; (layers, reason if none)."""
    if importlib.util.find_spec("triton") is None:
        return 0, "triton is not installed"
    config = _core(backbone).config
    config = getattr(config, "text_config", None) or config
    if getattr(config, "model_type", None) == "qwen3":
        return _fused_qwen3(backbone, torch, config)
    modeling = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    layers = [
        m for m in backbone.modules() if isinstance(m, modeling.Qwen3_5DecoderLayer)
    ]
    if not layers:
        return 0, "no Qwen3.5 decoder layers"
    conv = _bound(modeling.causal_conv1d_fn)
    chunk = _bound(modeling.torch_chunk_gated_delta_rule)
    checks = {
        "SDPA attention": getattr(config, "_attn_implementation", None) == "sdpa",
        "hidden size a multiple of 256": config.hidden_size % 256 == 0,
        "128-wide gated-delta heads": config.linear_key_head_dim == 128
        and config.linear_value_head_dim == 128,
        "256-wide attention heads": getattr(config, "head_dim", None) == 256,
        "SiLU MLP": getattr(config, "hidden_act", None) == "silu",
        "causal-conv1d kernel": getattr(conv, "__module__", "").startswith(
            "causal_conv1d"
        ),
        "FLA chunk kernel": getattr(chunk, "__module__", "").startswith("fla"),
    }
    missing = [name for name, ok in checks.items() if not ok]
    if missing:
        return 0, "needs " + ", ".join(missing)
    from . import fast_kernels as kernels

    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    forward = _fused_forward(kernels, modeling, sdpa_attention_forward, torch)
    for layer in layers:
        _prepare(layer, torch)
        layer.forward = types.MethodType(forward, layer)
    return len(layers), None


def _prepare(layer: Any, torch: Any) -> None:
    with torch.no_grad():
        p: dict[str, Any] = {
            "original": layer.forward,
            "eps": layer.input_layernorm.eps,
            "w1_in": (1.0 + layer.input_layernorm.weight.float()).contiguous(),
            "w1_post": (
                1.0 + layer.post_attention_layernorm.weight.float()
            ).contiguous(),
        }
        if layer.block_type == "linear_attention":
            m = layer.linear_attn
            p.update(
                conv_w=m.conv1d.weight.squeeze(1).float().contiguous(),
                A_log=m.A_log.float().contiguous(),
                dt_bias=m.dt_bias.float().contiguous(),
            )
        else:
            m = layer.self_attn
            p.update(
                qw1=(1.0 + m.q_norm.weight.float()).contiguous(),
                kw1=(1.0 + m.k_norm.weight.float()).contiguous(),
            )
    layer._decision2_fused = p


def _fused_forward(kernels: Any, modeling: Any, sdpa: Any, torch: Any) -> Any:
    def forward(
        layer: Any,
        hidden_states: Any,
        position_embeddings: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Any:
        p = layer._decision2_fused
        if (
            past_key_values is not None
            or hidden_states.dtype != torch.float32
            or not torch.is_autocast_enabled("cuda")
            or torch.get_autocast_dtype("cuda") != torch.bfloat16
            or (
                layer.block_type == "full_attention"
                and position_embeddings[0].dtype != torch.float32
            )
        ):
            return p["original"](
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
        hidden_states = hidden_states.contiguous()
        _, normed = kernels.add_rmsnorm(hidden_states, None, p["w1_in"], p["eps"])
        if layer.block_type == "linear_attention":
            delta = _gated_delta(
                kernels, modeling, layer.linear_attn, p, normed, attention_mask
            )
        else:
            delta = _attention(
                kernels, sdpa, layer.self_attn, p, normed, position_embeddings,
                attention_mask, {"position_ids": position_ids, **kwargs},
            )  # fmt: skip
        hidden, normed = kernels.add_rmsnorm(
            hidden_states, delta.contiguous(), p["w1_post"], p["eps"]
        )
        mlp = layer.mlp
        act = kernels.silu_mul(mlp.gate_proj(normed), mlp.up_proj(normed))
        return hidden + mlp.down_proj(act)

    return forward


def _gated_delta(
    kernels: Any, modeling: Any, m: Any, p: dict[str, Any], normed: Any, mask: Any
) -> Any:
    normed = modeling.apply_mask_to_padding_states(normed, mask)
    B, T, _ = normed.shape
    mixed = m.in_proj_qkv(normed)
    z = m.in_proj_z(normed)
    b = m.in_proj_b(normed)
    a = m.in_proj_a(normed)
    q, k, v, g, beta = kernels.gdn_prep(
        mixed.contiguous(), b.contiguous(), a.contiguous(), p["conv_w"],
        p["A_log"], p["dt_bias"], m.num_k_heads, m.head_k_dim,
    )  # fmt: skip
    repeat = m.num_v_heads // m.num_k_heads
    if repeat > 1:
        q = q.repeat_interleave(repeat, dim=2)
        k = k.repeat_interleave(repeat, dim=2)
    core, _ = modeling.torch_chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True, cu_seqlens=None,
    )  # fmt: skip
    out = kernels.gated_rmsnorm(
        core.reshape(-1, m.head_v_dim).contiguous(),
        z.reshape(-1, m.head_v_dim).contiguous(),
        m.norm.weight,
        m.norm.variance_epsilon,
    )
    return m.out_proj(out.reshape(B, T, -1))


def _attention(
    kernels: Any,
    sdpa: Any,
    m: Any,
    p: dict[str, Any],
    normed: Any,
    position_embeddings: Any,
    mask: Any,
    kwargs: dict[str, Any],
) -> Any:
    B, T, _ = normed.shape
    hd = m.head_dim
    qp = m.q_proj(normed).contiguous()
    kp = m.k_proj(normed).contiguous()
    vp = m.v_proj(normed)
    cos, sin = (t.contiguous() for t in position_embeddings)
    heads = qp.shape[-1] // (2 * hd)
    kv_heads = kp.shape[-1] // hd
    q, k = kernels.attn_prep(
        qp, kp, p["qw1"], p["kw1"], cos, sin, heads, kv_heads, hd, m.q_norm.eps
    )
    v = vp.view(B, T, kv_heads, hd).transpose(1, 2)
    attn, _ = sdpa(m, q, k, v, mask, dropout=0.0, scaling=m.scaling, **kwargs)
    gate = qp.view(B, T, heads, 2 * hd)[..., hd:]
    return m.o_proj(kernels.sigmoid_gate(attn, gate))


def _fused_qwen3(backbone: Any, torch: Any, config: Any) -> tuple[int, str | None]:
    modeling = importlib.import_module("transformers.models.qwen3.modeling_qwen3")
    layers = [
        m for m in backbone.modules() if isinstance(m, modeling.Qwen3DecoderLayer)
    ]
    if not layers:
        return 0, "no Qwen3 decoder layers"
    layer_types = set(getattr(config, "layer_types", None) or ["full_attention"])
    checks = {
        "SDPA attention": getattr(config, "_attn_implementation", None) == "sdpa",
        "hidden size a multiple of 256": config.hidden_size % 256 == 0,
        "128-wide attention heads": layers[0].self_attn.head_dim == 128,
        "full attention layers only": layer_types == {"full_attention"},
        "no attention bias": not getattr(config, "attention_bias", False),
        "SiLU MLP": getattr(config, "hidden_act", None) == "silu",
    }
    missing = [name for name, ok in checks.items() if not ok]
    if missing:
        return 0, "needs " + ", ".join(missing)
    from . import fast_kernels as kernels

    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    forward = _qwen3_forward(kernels, sdpa_attention_forward, torch)
    for layer in layers:
        with torch.no_grad():
            m = layer.self_attn
            layer._decision2_fused = {
                "original": layer.forward,
                "eps": layer.input_layernorm.variance_epsilon,
                "w_in": layer.input_layernorm.weight.float().contiguous(),
                "w_post": layer.post_attention_layernorm.weight.float().contiguous(),
                "qw": m.q_norm.weight.float().contiguous(),
                "kw": m.k_norm.weight.float().contiguous(),
            }
        layer.forward = types.MethodType(forward, layer)
    return len(layers), None


def _qwen3_forward(kernels: Any, sdpa: Any, torch: Any) -> Any:
    def forward(
        layer: Any,
        hidden_states: Any,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        use_cache: Any = False,
        position_embeddings: Any = None,
        **kwargs: Any,
    ) -> Any:
        p = layer._decision2_fused
        if (
            past_key_values is not None
            or hidden_states.dtype != torch.float32
            or not torch.is_autocast_enabled("cuda")
            or torch.get_autocast_dtype("cuda") != torch.bfloat16
            or position_embeddings is None
            or position_embeddings[0].dtype != torch.float32
        ):
            return p["original"](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = hidden_states.contiguous()
        _, normed = kernels.add_rmsnorm(hidden_states, None, p["w_in"], p["eps"])
        m = layer.self_attn
        B, T, _ = normed.shape
        hd = m.head_dim
        qp = m.q_proj(normed).contiguous()
        kp = m.k_proj(normed).contiguous()
        vp = m.v_proj(normed)
        cos, sin = (t.contiguous() for t in position_embeddings)
        heads = qp.shape[-1] // hd
        kv_heads = kp.shape[-1] // hd
        q, k = kernels.attn_prep(
            qp, kp, p["qw"], p["kw"], cos, sin, heads, kv_heads, hd,
            m.q_norm.variance_epsilon, gated=False, zero_centred=False,
        )  # fmt: skip
        v = vp.view(B, T, kv_heads, hd).transpose(1, 2)
        attn, _ = sdpa(
            m, q, k, v, attention_mask, dropout=0.0, scaling=m.scaling,
            sliding_window=m.sliding_window, position_ids=position_ids,
            use_cache=use_cache, **kwargs,
        )  # fmt: skip
        delta = m.o_proj(attn.reshape(B, T, -1).contiguous())
        hidden, normed = kernels.add_rmsnorm(
            hidden_states, delta.contiguous(), p["w_post"], p["eps"]
        )
        mlp = layer.mlp
        act = kernels.silu_mul(mlp.gate_proj(normed), mlp.up_proj(normed))
        return hidden + mlp.down_proj(act)

    return forward


class Masks:
    """The masks Transformers builds for a right-padded batch, from host-known padding.

    Unpadded: no mask for either layer type (SDPA runs causal; the gated-delta
    layers skip their padding multiply). Padded: the 2D mask itself for the
    gated-delta layers and, for full attention, the [B, 1, T, T] causal-and-key-valid
    mask in the form SDPA turns Transformers' boolean mask into in every layer: BF16
    (the autocast query dtype), 0 where allowed and -inf elsewhere, with rows padded
    to a multiple of ``MASK_ALIGN`` so the memory-efficient kernel needs no re-padding.
    """

    def __init__(self, torch: Any, layer_types: set[str]):
        self.torch = torch
        self.layer_types = layer_types
        self.causal: dict[tuple[int, Any], Any] = {}

    def build(self, attention_mask: Any, padded: bool) -> dict[str, Any]:
        if not padded:
            return {name: None for name in self.layer_types}
        torch = self.torch
        T = attention_mask.shape[1]
        key = (T, attention_mask.device)
        causal = self.causal.get(key)
        if causal is None:
            causal = self.causal[key] = torch.ones(
                T, T, dtype=torch.bool, device=attention_mask.device
            ).tril()
        masks: dict[str, Any] = {}
        if "full_attention" in self.layer_types:
            allowed = causal[None, None] & attention_mask.bool()[:, None, None, :]
            stride = -(-T // MASK_ALIGN) * MASK_ALIGN
            additive = torch.zeros(
                attention_mask.shape[0], 1, T, stride,
                dtype=torch.bfloat16, device=attention_mask.device,
            )[..., :T]  # fmt: skip
            masks["full_attention"] = additive.masked_fill_(
                allowed.logical_not(), float("-inf")
            )
        if "linear_attention" in self.layer_types:
            masks["linear_attention"] = attention_mask
        return masks


def _output_tensors(value: Any, torch: Any) -> list[Any]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for item in value for t in _output_tensors(item, torch)]
    if hasattr(value, "to_tuple"):
        return _output_tensors(value.to_tuple(), torch)
    return []


class Graphs:
    """HIP graphs of the backbone forward per exact padded shape (see the module docstring)."""

    def __init__(
        self,
        backbone: Any,
        torch: Any,
        masks: Masks,
        *,
        capture_after: int = CAPTURE_AFTER,
        max_graphs: int = MAX_GRAPHS,
        max_tokens: int = MAX_GRAPH_TOKENS,
        max_output_bytes: int = MAX_GRAPH_OUTPUT_BYTES,
    ):
        self.torch = torch
        self.masks = masks
        self.forward = backbone.forward
        self.capture_after = capture_after
        self.max_graphs = max_graphs
        self.max_tokens = max_tokens
        self.max_output_bytes = max_output_bytes
        self.graphs: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
        self.seen: dict[tuple, int] = {}
        self.failed: set[tuple] = set()
        self.output_bytes = 0
        self.pool: Any = None
        self.lengths: list[int] | None = None
        self.stats = {
            "captures": 0,
            "replays": 0,
            "eager": 0,
            "evicted": 0,
            "failed": 0,
        }
        backbone.forward = self.__call__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        lengths, self.lengths = self.lengths, None
        torch = self.torch
        input_ids = kwargs.get("input_ids")
        mask = kwargs.get("attention_mask")
        extra = set(kwargs) - {
            "input_ids",
            "attention_mask",
            "use_cache",
            "output_hidden_states",
        }
        if (
            args
            or extra
            or lengths is None
            or kwargs.get("use_cache")
            or input_ids is None
            or not torch.is_tensor(mask)
            or input_ids.dim() != 2
            or mask.shape != input_ids.shape
            or not input_ids.is_cuda
            or not torch.is_inference_mode_enabled()
            or not torch.is_autocast_enabled("cuda")
            or torch.get_autocast_dtype("cuda") != torch.bfloat16
        ):
            return self.forward(*args, **kwargs)
        rows, length = input_ids.shape
        if len(lengths) != rows or max(lengths) > length:
            return self.forward(*args, **kwargs)
        padded = any(n != length for n in lengths)
        key = (rows, length, padded, bool(kwargs.get("output_hidden_states")))
        entry = self.graphs.get(key)
        if entry is None:
            if key in self.failed or rows * length > self.max_tokens:
                self.stats["eager"] += 1
                return self.forward(*args, **kwargs)
            if len(self.seen) > 1 << 16:
                self.seen.clear()
            self.seen[key] = self.seen.get(key, 0) + 1
            if self.seen[key] < self.capture_after:
                self.stats["eager"] += 1
                return self.forward(*args, **kwargs)
            entry = self._capture(key, kwargs, padded)
            if entry is None:
                self.stats["eager"] += 1
                return self.forward(*args, **kwargs)
        else:
            self.graphs.move_to_end(key)
        entry["input_ids"].copy_(input_ids)
        entry["attention_mask"].copy_(mask)
        entry["graph"].replay()
        self.stats["replays"] += 1
        return entry["output"]

    def _capture(self, key: tuple, kwargs: dict[str, Any], padded: bool) -> dict | None:
        torch = self.torch
        static = {
            "input_ids": kwargs["input_ids"].clone(),
            "attention_mask": kwargs["attention_mask"].clone(),
        }
        options = {
            k: v for k, v in kwargs.items() if k not in ("input_ids", "attention_mask")
        }

        def body() -> Any:
            masks = self.masks.build(static["attention_mask"], padded)
            return self.forward(
                input_ids=static["input_ids"], attention_mask=masks, **options
            )

        try:
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        body()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool):
                    output = body()
            torch.cuda.synchronize()
        except Exception:
            torch.cuda.synchronize()
            self.failed.add(key)
            self.stats["failed"] += 1
            return None
        size = sum(t.numel() * t.element_size() for t in _output_tensors(output, torch))
        entry = {**static, "graph": graph, "output": output, "bytes": size}
        self.graphs[key] = entry
        self.output_bytes += size
        self.stats["captures"] += 1
        while len(self.graphs) > 1 and (
            len(self.graphs) > self.max_graphs
            or self.output_bytes > self.max_output_bytes
        ):
            _, old = self.graphs.popitem(last=False)
            self.output_bytes -= old["bytes"]
            self.stats["evicted"] += 1
        return entry


class FastPath:
    """What is installed on one loaded model, and the per-forward hook ``QwenDecision`` calls."""

    def __init__(self, memo: CastMemo, graphs: Graphs | None, summary: dict[str, Any]):
        self.memo = memo
        self.graphs = graphs
        self.summary = summary

    @contextmanager
    def forward(self, lengths: list[int]):
        """Around one backbone forward of a right-padded batch with these unpadded lengths."""
        if self.graphs is not None:
            self.graphs.lengths = list(lengths)
        try:
            yield
        finally:
            if self.graphs is not None:
                self.graphs.lengths = None
            self.memo.clear()

    def receipt(self) -> dict[str, Any]:
        out = dict(self.summary)
        if self.graphs is not None:
            out["graph_stats"] = {
                **self.graphs.stats,
                "cached": len(self.graphs.graphs),
            }
        return out


def install(
    model: Any, torch: Any, *, graphs: bool = True, kernels: bool = True
) -> FastPath | None:
    """Install the exact fast path on a loaded GPU model; None where it is not verified."""
    if _switched_off("DECISION2_FAST"):
        return None
    try:
        import transformers
    except ImportError:
        return None
    if not torch.version.hip or not transformers.__version__.startswith(
        TESTED_TRANSFORMERS
    ):
        return None
    backbone = model.backbone
    config = _core(backbone).config
    config = getattr(config, "text_config", None) or config
    if getattr(config, "model_type", None) not in ("qwen3", "qwen3_5_text", "qwen3_5"):
        return None
    graphs = graphs and not _switched_off("DECISION2_GRAPHS")
    kernels = kernels and not _switched_off("DECISION2_KERNELS")
    memo = CastMemo(torch)
    summary: dict[str, Any] = {
        "transformers": transformers.__version__,
        "shared_cast_linears": shared_cast_linears(backbone, memo, torch),
        "rmsnorm_plus_one": rmsnorm_plus_one(backbone, torch),
        "lora": lean_lora(backbone, memo, torch),
    }
    fused, reason = 0, "off"
    if kernels:
        arch = torch.cuda.get_device_properties(
            next(backbone.parameters()).device
        ).gcnArchName.split(":")[0]
        if arch == "gfx942":
            fused, reason = fused_layers(backbone, torch)
        else:
            reason = f"verified on gfx942 only, not {arch}"
    summary["fused_layers"] = fused
    if not fused:
        summary["fused_skipped"] = reason
    runner = None
    layer_types = set(getattr(config, "layer_types", None) or ["full_attention"])
    if not graphs:
        summary["graphs"] = "off"
    elif (
        not layer_types <= set(MASK_TYPES)
        or getattr(config, "_attn_implementation", None) != "sdpa"
    ):
        summary["graphs"] = "needs SDPA and full / linear attention layers only"
    else:
        runner = Graphs(backbone, torch, Masks(torch, layer_types))
        summary["graphs"] = {
            "capture_after": runner.capture_after,
            "max_graphs": runner.max_graphs,
            "max_tokens": runner.max_tokens,
            "max_output_bytes": runner.max_output_bytes,
        }
    return FastPath(memo, runner, summary)
