"""The exact GPU fast path of the native engine: fused decoder layers, lean LoRA and HIP graphs.

Every piece reproduces the eager backbone under BF16 autocast bit for bit (the
parity records compare the result with the released packages' runtime on the
four scored panels), so it belongs to the exact profile:

- **Fused layers.** The element-wise ops of each Qwen3.5 / Qwen3 decoder layer
  run as the accelerator's fused kernels (``FUSED_SLOTS``; ROCm gfx942 Triton),
  which round exactly like the ops they replace. GEMMs, attention and the
  gated-delta kernel are the eager path's. A layer call whose fused kernels
  fail (for example a compiler error on an unusual shape) runs the layer's
  eager forward instead, and the shape is remembered. A Qwen3.5 layer's
  forest forward (``models/forest.py``) fuses the same steps except the
  gated-delta preparation, whose convolution there is ``F.conv1d``.
- **Lean LoRA.** An unmerged LoRA layer with a power-of-two scaling multiplies
  its factors in BF16 (the values autocast uses) with the scaling folded into B
  (exact for a power of two): 5 kernels instead of 9, the same products.
- **Graphs.** The backbone forward of each exact padded shape (rows, padded
  length, whether any row is padded) is captured once as a HIP / CUDA graph on
  the shape's second use and replayed afterwards, with the masks built from the
  host-known row lengths as the eager path builds them. Shapes above
  ``MAX_GRAPH_TOKENS`` padded tokens run eagerly: they are GPU-bound (on
  Eos-0.8B a graph saves 6% at 2,800 tokens and nothing from 11,000), and
  concurrent traffic forms many distinct large shapes whose captures would cost
  several forwards each. At most ``MAX_GRAPHS`` graphs (and
  ``MAX_GRAPH_OUTPUT_BYTES`` of outputs) are captured; after that, new shapes
  run eagerly. Captured graphs are never destroyed while serving: on ROCm,
  evicting large graphs from the shared memory pool, between large eager
  batches of Index traffic, led to GPU memory access faults. On CUDA the same
  runner also replays Vela 2.0's forest forward (``models/forest.py``) per
  exact forest shape; ROCm keeps forests eager, because an eager forward while
  a forest graph of eight or more Vela-4B layers is alive faults there.
"""

from __future__ import annotations

import math
import types
from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from torch import nn

from ...accel.kernels import KernelSet
from .models import Backbone
from .models.common import attention
from .models.forest import Forest, forest_attention, forest_gated_delta
from .models.lora import LoRALinear
from .models.tree import Tree, is_tree, suffix_rule

if TYPE_CHECKING:
    from .models.qwen3 import Qwen3Backbone, Qwen3Layer
    from .models.qwen3_5 import GatedAttention, GatedDeltaNet, Qwen3_5Layer

FUSED_SLOTS: dict[str | None, tuple[str, ...]] = {
    "qwen3": ("add_rmsnorm", "residual_add", "silu_mul", "attn_prep"),
    "qwen3_5_text": (
        "add_rmsnorm",
        "residual_add",
        "silu_mul",
        "attn_prep",
        "gdn_prep",
        "gated_rmsnorm",
        "sigmoid_gate",
    ),
}
MAX_GRAPH_TOKENS = 4096
MAX_GRAPHS = 512
MAX_GRAPH_OUTPUT_BYTES = 4 << 30
CAPTURE_AFTER = 2
MASK_ALIGN = 16


# ---------------------------------------------------------------------------
# Fused decoder layers
# ---------------------------------------------------------------------------

# Head widths the fused gfx942 kernels are written for.
DENSE_HEAD_DIM = 128
HYBRID_HEAD_DIM = 256
GATED_DELTA_HEAD_DIM = 128


class FusedLayer(Protocol):
    """What ``install_fused`` sets on a decoder layer, besides its ``forward``.

    ``_fused`` holds the prepared weights, ``_fused_forward`` the fused forward
    and ``_fused_failed`` the shapes whose fused call failed; a Qwen3.5 layer's
    ``forward_forest`` is replaced too.
    """

    _fused: dict[str, Any]
    _fused_forward: Callable[..., torch.Tensor]
    _fused_failed: set[tuple[object, ...]]
    forward_forest: Callable[..., tuple[torch.Tensor, torch.Tensor]]


def fused_unavailable(backbone: nn.Module, kernels: KernelSet) -> str | None:
    """Why the fused layers cannot run this backbone with these kernels; None when they can."""
    config = cast(Backbone, backbone).config
    model_type = getattr(backbone, "model_type", None)
    slots = FUSED_SLOTS.get(model_type)
    if slots is None:
        return f"no fused layers for {model_type!r}"
    missing = [name for name in slots if not kernels.has(name)]
    if missing:
        return f"kernels missing: {missing}"
    head = (
        config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    )
    checks = {
        "hidden size a multiple of 256": config["hidden_size"] % 256 == 0,
        "SiLU MLP": config.get("hidden_act", "silu") == "silu",
        "no attention bias": not config.get("attention_bias", False),
    }
    if model_type == "qwen3":
        checks["128-wide attention heads"] = head == DENSE_HEAD_DIM
    else:
        checks["256-wide attention heads"] = head == HYBRID_HEAD_DIM
        checks["128-wide gated-delta heads"] = (
            config["linear_key_head_dim"] == GATED_DELTA_HEAD_DIM
            and config["linear_value_head_dim"] == GATED_DELTA_HEAD_DIM
        )
    if model_type == "qwen3":
        checks["an FP32 stream (BF16 Qwen3 norms round differently)"] = (
            cast("Qwen3Backbone", backbone).norm.weight.dtype == torch.float32
        )
    failed = [name for name, ok in checks.items() if not ok]
    return "needs " + ", ".join(failed) if failed else None


def install_fused(backbone: nn.Module) -> int:
    """Run every Qwen3.5 / Qwen3 decoder layer through its fused forward; returns how many layers run fused."""
    count = 0
    for layer in cast(nn.ModuleList, backbone.layers):
        fused = cast(FusedLayer, layer)
        if type(layer).__name__ == "Qwen3_5Layer":
            fused._fused = _prepare_qwen3_5(cast("Qwen3_5Layer", layer))
            fused._fused_forward = types.MethodType(_qwen3_5_forward, layer)
            _install_fused_forest(cast("Qwen3_5Layer", layer))
        elif type(layer).__name__ == "Qwen3Layer":
            fused._fused = _prepare_qwen3(cast("Qwen3Layer", layer))
            fused._fused_forward = types.MethodType(_qwen3_forward, layer)
        else:
            continue
        fused._fused_failed = set()
        eager = layer.forward

        def forward(
            self: FusedLayer,
            hidden_states: torch.Tensor,
            *args: Any,
            _eager: Callable[..., torch.Tensor] = eager,
        ) -> torch.Tensor:
            return run_fused(self, _eager, hidden_states, *args)

        layer.forward = types.MethodType(forward, layer)
        count += 1
    return count


def _install_fused_forest(layer: Qwen3_5Layer) -> None:
    """Run the layer's forest forward fused where it applies, eagerly otherwise (as ``run_fused``)."""
    eager = layer.forward_forest
    fused: Callable[..., tuple[torch.Tensor, torch.Tensor]] = types.MethodType(
        _qwen3_5_forest, layer
    )

    def forward_forest(
        self: FusedLayer, prefix: torch.Tensor, blocks: torch.Tensor, *args: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = ("forest", tuple(prefix.shape), tuple(blocks.shape))
        if key in self._fused_failed or not fused_applies(prefix):
            return eager(prefix, blocks, *args)
        try:
            return fused(prefix, blocks, *args)
        except RuntimeError:
            self._fused_failed.add(key)
            return eager(prefix, blocks, *args)

    cast(FusedLayer, layer).forward_forest = types.MethodType(forward_forest, layer)


def _prepare_qwen3_5(layer: Qwen3_5Layer) -> dict[str, Any]:
    with torch.no_grad():
        p: dict[str, Any] = {
            "eps": layer.input_layernorm.eps,
            "w1_in": (1.0 + layer.input_layernorm.weight.float()).contiguous(),
            "w1_post": (
                1.0 + layer.post_attention_layernorm.weight.float()
            ).contiguous(),
        }
        if layer.kind == "linear_attention":
            m = layer.linear_attn
            if m.conv1d.bias is not None:
                raise ValueError("the fused gated-delta prep has no convolution bias")
            p.update(
                conv_w=m.conv1d.weight.squeeze(1).float().contiguous(),
                A_log=m.A_log.float().contiguous(),
                dt_bias=m.dt_bias.float().contiguous(),
            )
        else:
            attn = layer.self_attn
            p.update(
                qw1=(1.0 + attn.q_norm.weight.float()).contiguous(),
                kw1=(1.0 + attn.k_norm.weight.float()).contiguous(),
            )
    return p


def _prepare_qwen3(layer: Qwen3Layer) -> dict[str, Any]:
    m = layer.self_attn
    with torch.no_grad():
        return {
            "eps": layer.input_layernorm.variance_epsilon,
            "w_in": layer.input_layernorm.weight.float().contiguous(),
            "w_post": layer.post_attention_layernorm.weight.float().contiguous(),
            "qw": m.q_norm.weight.float().contiguous(),
            "kw": m.k_norm.weight.float().contiguous(),
        }


def fused_applies(hidden_states: torch.Tensor) -> bool:
    """The fused kernels reproduce the eager layer under BF16 autocast on an FP32 or BF16 stream."""
    return (
        hidden_states.dtype in (torch.float32, torch.bfloat16)
        and hidden_states.is_cuda
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    )


def run_fused(
    layer: FusedLayer,
    eager: Callable[..., torch.Tensor],
    hidden_states: torch.Tensor,
    *args: Any,
) -> torch.Tensor:
    """The layer's fused forward, or its eager forward where the fused one does not apply or failed."""
    key = tuple(hidden_states.shape)
    if key in layer._fused_failed or not fused_applies(hidden_states):
        return eager(hidden_states, *args)
    try:
        return layer._fused_forward(hidden_states, *args)
    except RuntimeError:
        if torch.cuda.is_current_stream_capturing():
            raise
        layer._fused_failed.add(key)
        return eager(hidden_states, *args)


def _residual(
    kernels: KernelSet, hidden: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    """The MLP residual ``hidden + delta`` through the fused add where it applies."""
    if (
        delta.shape == hidden.shape
        and delta.dtype in (torch.bfloat16, torch.float32)
        and delta.is_contiguous()
        and hidden.is_contiguous()
    ):
        out: torch.Tensor = kernels("residual_add")(hidden, delta)
        return out
    return hidden + delta


def _qwen3_5_forward(
    layer: Qwen3_5Layer,
    hidden_states: torch.Tensor,
    rotary: tuple[torch.Tensor, torch.Tensor],
    full_mask: torch.Tensor | Tree | None,
    linear_mask: torch.Tensor | Tree | None,
    kernels: KernelSet,
) -> torch.Tensor:
    p = cast(FusedLayer, layer)._fused
    hidden_states = hidden_states.contiguous()
    _, normed = kernels("add_rmsnorm")(hidden_states, None, p["w1_in"], p["eps"])
    if layer.kind == "linear_attention":
        delta = _gated_delta(layer.linear_attn, p, normed, linear_mask, kernels)
    else:
        delta = _gated_attention(layer.self_attn, p, normed, rotary, full_mask, kernels)
    hidden, normed = kernels("add_rmsnorm")(
        hidden_states, delta.contiguous(), p["w1_post"], p["eps"]
    )
    mlp = layer.mlp
    act = kernels("silu_mul")(mlp.gate_proj(normed), mlp.up_proj(normed))
    return _residual(kernels, hidden, mlp.down_proj(act))


def _qwen3_5_forest(
    layer: Qwen3_5Layer,
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    rotary: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    forest: Forest,
    kernels: KernelSet,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``Qwen3_5Layer.forward_forest`` with fused norms, residuals, MLP gates, attention prep and gates."""
    p = cast(FusedLayer, layer)._fused
    prefix, blocks = prefix.contiguous(), blocks.contiguous()
    normed_prefix, normed_blocks = [
        kernels("add_rmsnorm")(x, None, p["w1_in"], p["eps"])[1]
        for x in (prefix, blocks)
    ]
    if layer.kind == "linear_attention":
        m = layer.linear_attn
        mixed = forest_gated_delta(
            m,
            normed_prefix,
            normed_blocks,
            forest,
            kernels,
            norm=lambda core, z: kernels("gated_rmsnorm")(
                core.contiguous(),
                z.contiguous(),
                m.norm.weight,
                m.norm.variance_epsilon,
            ),
        )
    else:
        mixed = forest_attention(
            layer.self_attn,
            normed_prefix,
            normed_blocks,
            *rotary,
            forest,
            project=lambda m, h, table: _forest_projection(m, p, h, table, kernels),
            gate=lambda m, out, gate: m.o_proj(
                kernels("sigmoid_gate")(out.transpose(1, 2), gate)
            ),
        )
    out = []
    for stream, delta in zip((prefix, blocks), mixed, strict=True):
        hidden, normed_post = kernels("add_rmsnorm")(
            stream, delta.contiguous(), p["w1_post"], p["eps"]
        )
        mlp = layer.mlp
        act = kernels("silu_mul")(mlp.gate_proj(normed_post), mlp.up_proj(normed_post))
        out.append(_residual(kernels, hidden, mlp.down_proj(act)))
    return out[0], out[1]


def _forest_projection(
    m: GatedAttention,
    p: dict[str, Any],
    h: torch.Tensor,
    rotary: tuple[torch.Tensor, torch.Tensor],
    kernels: KernelSet,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``forest.project_attention`` through ``attn_prep``; the gate stays a view of the q projection."""
    rows, length, _ = h.shape
    hd = m.head_dim
    qp = m.q_proj(h).contiguous()
    kp = m.k_proj(h).contiguous()
    vp = m.v_proj(h)
    cos, sin = (t.contiguous() for t in rotary)
    heads = qp.shape[-1] // (2 * hd)
    kv_heads = kp.shape[-1] // hd
    q, k = kernels("attn_prep")(
        qp, kp, p["qw1"], p["kw1"], cos, sin, heads, kv_heads, hd, m.q_norm.eps
    )
    v = vp.view(rows, length, kv_heads, hd).transpose(1, 2)
    return q, k, v, qp.view(rows, length, heads, 2 * hd)[..., hd:]


def _tree_gated_delta(
    m: GatedDeltaNet,
    p: dict[str, Any],
    normed: torch.Tensor,
    tree: Tree,
    kernels: KernelSet,
) -> torch.Tensor:
    """``tree.tree_gated_delta`` with the fused kernels: the prefix from a zero state, every suffix from the
    prefix-end state with its convolution window starting in the prefix (rows ``[prefix tail | suffix]``,
    whose first outputs are dropped)."""
    _, length, _ = normed.shape
    pre, n, window = tree.prefix, len(tree.lengths), m.conv_kernel_size
    mixed = m.in_proj_qkv(normed)
    z = m.in_proj_z(normed)
    b = m.in_proj_b(normed)
    a = m.in_proj_a(normed)
    args = (p["conv_w"], p["A_log"], p["dt_bias"], m.num_k_heads, m.head_k_dim)

    def rows(x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [x[:, pre - (window - 1) : pre].expand(n, -1, -1), tree.rows(x[0, pre:])],
            dim=1,
        )

    def packed(x: torch.Tensor) -> torch.Tensor:
        return tree.packed(x[:, window - 1 :])[None]

    prep = kernels("gdn_prep")
    hq, hk, hv, hg, hbeta = prep(
        mixed[:, :pre].contiguous(),
        b[:, :pre].contiguous(),
        a[:, :pre].contiguous(),
        *args,
    )
    sq, sk, sv, sg, sbeta = (
        packed(x) for x in prep(rows(mixed), rows(b), rows(a), *args)
    )
    repeat = m.num_v_heads // m.num_k_heads
    if repeat > 1:
        hq, hk, sq, sk = (x.repeat_interleave(repeat, dim=2) for x in (hq, hk, sq, sk))
    rule = kernels("chunk_gated_delta_rule")
    out_prefix, state = rule(
        hq, hk, hv, g=hg, beta=hbeta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True,
    )  # fmt: skip
    start = state.expand(n, *state.shape[1:]).contiguous()
    out_suffix = suffix_rule(rule, tree, sq, sk, sv, sg, sbeta, start)
    core = torch.cat([out_prefix, out_suffix], dim=1)
    out = kernels("gated_rmsnorm")(
        core.reshape(-1, m.head_v_dim).contiguous(),
        z.reshape(-1, m.head_v_dim).contiguous(),
        m.norm.weight,
        m.norm.variance_epsilon,
    )
    projected: torch.Tensor = m.out_proj(out.reshape(1, length, -1))
    return projected


def _gated_delta(
    m: GatedDeltaNet,
    p: dict[str, Any],
    normed: torch.Tensor,
    mask: torch.Tensor | Tree | None,
    kernels: KernelSet,
) -> torch.Tensor:
    if kernels.select("causal_conv1d").variant is not None:
        # The model's released convolution (a kernel variant) is not what gdn_prep fuses.
        eager: torch.Tensor = m(normed, mask, kernels)
        return eager
    if is_tree(mask):
        return _tree_gated_delta(m, p, normed, mask, kernels)
    if mask is not None:
        normed = (normed * mask[:, :, None]).to(normed.dtype)
    batch, length, _ = normed.shape
    mixed = m.in_proj_qkv(normed)
    z = m.in_proj_z(normed)
    b = m.in_proj_b(normed)
    a = m.in_proj_a(normed)
    q, k, v, g, beta = kernels("gdn_prep")(
        mixed.contiguous(), b.contiguous(), a.contiguous(), p["conv_w"], p["A_log"], p["dt_bias"],
        m.num_k_heads, m.head_k_dim,
    )  # fmt: skip
    repeat = m.num_v_heads // m.num_k_heads
    if repeat > 1:
        q = q.repeat_interleave(repeat, dim=2)
        k = k.repeat_interleave(repeat, dim=2)
    core, _ = kernels("chunk_gated_delta_rule")(
        q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
    )  # fmt: skip
    out = kernels("gated_rmsnorm")(
        core.reshape(-1, m.head_v_dim).contiguous(),
        z.reshape(-1, m.head_v_dim).contiguous(),
        m.norm.weight,
        m.norm.variance_epsilon,
    )
    projected: torch.Tensor = m.out_proj(out.reshape(batch, length, -1))
    return projected


def _gated_attention(
    m: GatedAttention,
    p: dict[str, Any],
    normed: torch.Tensor,
    rotary: tuple[torch.Tensor, torch.Tensor],
    mask: torch.Tensor | Tree | None,
    kernels: KernelSet,
) -> torch.Tensor:
    batch, length, _ = normed.shape
    hd = m.head_dim
    qp = m.q_proj(normed).contiguous()
    kp = m.k_proj(normed).contiguous()
    vp = m.v_proj(normed)
    cos, sin = (t.contiguous() for t in rotary)
    heads = qp.shape[-1] // (2 * hd)
    kv_heads = kp.shape[-1] // hd
    q, k = kernels("attn_prep")(
        qp, kp, p["qw1"], p["kw1"], cos, sin, heads, kv_heads, hd, m.q_norm.eps
    )
    v = vp.view(batch, length, kv_heads, hd).transpose(1, 2)
    out = attention(kernels, q, k, v, mask, groups=m.groups, scaling=m.scaling)
    gate = qp.view(batch, length, heads, 2 * hd)[..., hd:]
    gated: torch.Tensor = m.o_proj(kernels("sigmoid_gate")(out, gate))
    return gated


def _qwen3_forward(
    layer: Qwen3Layer,
    hidden_states: torch.Tensor,
    rotary: tuple[torch.Tensor, torch.Tensor],
    mask: torch.Tensor | Tree | None,
    kernels: KernelSet,
) -> torch.Tensor:
    p = cast(FusedLayer, layer)._fused
    hidden_states = hidden_states.contiguous()
    _, normed = kernels("add_rmsnorm")(hidden_states, None, p["w_in"], p["eps"])
    m = layer.self_attn
    batch, length, _ = normed.shape
    hd = m.head_dim
    qp = m.q_proj(normed).contiguous()
    kp = m.k_proj(normed).contiguous()
    vp = m.v_proj(normed)
    cos, sin = (t.contiguous() for t in rotary)
    heads = qp.shape[-1] // hd
    kv_heads = kp.shape[-1] // hd
    q, k = kernels("attn_prep")(
        qp, kp, p["qw"], p["kw"], cos, sin, heads, kv_heads, hd, m.q_norm.variance_epsilon,
        gated=False, zero_centred=False,
    )  # fmt: skip
    v = vp.view(batch, length, kv_heads, hd).transpose(1, 2)
    out = attention(kernels, q, k, v, mask, groups=m.groups, scaling=m.scaling)
    delta = m.o_proj(out.reshape(batch, length, -1).contiguous())
    hidden, normed = kernels("add_rmsnorm")(
        hidden_states, delta.contiguous(), p["w_post"], p["eps"]
    )
    mlp = layer.mlp
    act = kernels("silu_mul")(mlp.gate_proj(normed), mlp.up_proj(normed))
    return _residual(kernels, hidden, mlp.down_proj(act))


# ---------------------------------------------------------------------------
# Lean LoRA
# ---------------------------------------------------------------------------


def install_lean_lora(backbone: nn.Module) -> dict[str, int]:
    """The lean forward for every LoRA layer whose scaling is a power of two and whose base is BF16."""
    linear = torch.nn.functional.linear
    lean = kept = 0
    for module in backbone.modules():
        if not isinstance(module, LoRALinear):
            continue
        weights = _lean_weights(module)
        if weights is None:
            kept += 1
            continue
        base, factor_a, factor_b, bias = weights
        original = module.forward

        def forward(
            x: torch.Tensor,
            _w: torch.Tensor = base,
            _a: torch.Tensor = factor_a,
            _b: torch.Tensor = factor_b,
            _bias: torch.Tensor | None = bias,
            _original: Callable[[torch.Tensor], torch.Tensor] = original,
        ) -> torch.Tensor:
            if not (x.is_cuda and torch.is_autocast_enabled("cuda")):
                return _original(x)
            xb = x.to(torch.bfloat16)
            return linear(xb, _w, _bias) + linear(linear(xb, _a), _b)

        cast(nn.Module, module).forward = forward
        lean += 1
    return {"lean": lean, "kept": kept}


def _lean_weights(
    module: LoRALinear,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    base = module.base_layer
    scaling = float(module.scaling)
    if (
        base.weight.dtype != torch.bfloat16
        or scaling <= 0
        or not math.log2(scaling).is_integer()
    ):
        return None
    if base.bias is not None and base.bias.dtype != torch.bfloat16:
        return None
    with torch.no_grad():
        factor_a = module.lora_A.weight.detach().to(torch.bfloat16)
        rounded_b = module.lora_B.weight.detach().to(torch.bfloat16)
        factor_b = (rounded_b.float() * scaling).to(torch.bfloat16)
        if not torch.equal(factor_b.float(), rounded_b.float() * scaling):
            return None
    return base.weight, factor_a, factor_b, base.bias


# ---------------------------------------------------------------------------
# Host-built masks and graphs
# ---------------------------------------------------------------------------


class Masks:
    """The masks the eager backbone builds for a right-padded batch, from host-known padding.

    Unpadded: no mask for either layer type (SDPA runs causal; the gated-delta
    layers skip their padding multiply). Padded: the 2D mask for the gated-delta
    layers and, for full attention, the [B, 1, T, T] causal-and-key-valid mask in
    the form SDPA turns the eager boolean mask into: BF16 (the autocast query
    dtype), 0 where allowed and -inf elsewhere, its rows padded to a multiple of
    ``MASK_ALIGN`` so the memory-efficient kernel needs no re-padding.
    """

    def __init__(self) -> None:
        self.causal: dict[tuple[int, Any], torch.Tensor] = {}

    def build(
        self, attention_mask: torch.Tensor, padded: bool
    ) -> dict[str, torch.Tensor | None]:
        if not padded:
            return {"full": None, "linear": None}
        length = attention_mask.shape[1]
        key = (length, attention_mask.device)
        causal = self.causal.get(key)
        if causal is None:
            causal = self.causal[key] = torch.ones(
                length, length, dtype=torch.bool, device=attention_mask.device
            ).tril()
        allowed = causal[None, None] & attention_mask.bool()[:, None, None, :]
        stride = -(-length // MASK_ALIGN) * MASK_ALIGN
        additive = torch.zeros(
            attention_mask.shape[0],
            1,
            length,
            stride,
            dtype=torch.bfloat16,
            device=attention_mask.device,
        )[..., :length]
        return {
            "full": additive.masked_fill_(allowed.logical_not(), float("-inf")),
            "linear": attention_mask,
        }


def _output_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


class Graphs:
    """Graphs of the backbone forward per exact padded shape (see the module docstring)."""

    def __init__(
        self,
        backbone: nn.Module,
        masks: Masks,
        *,
        capture_after: int = CAPTURE_AFTER,
        max_graphs: int = MAX_GRAPHS,
        max_tokens: int = MAX_GRAPH_TOKENS,
        max_output_bytes: int = MAX_GRAPH_OUTPUT_BYTES,
    ):
        self.backbone = backbone
        self.masks = masks
        self.capture_after = capture_after
        self.max_graphs = max_graphs
        self.max_tokens = max_tokens
        self.max_output_bytes = max_output_bytes
        self.graphs: dict[Hashable, dict[str, Any]] = {}
        self.seen: dict[Hashable, int] = {}
        self.failed: set[Hashable] = set()
        self.output_bytes = 0
        self.pool: Any = None
        self.stats = {
            "captures": 0,
            "replays": 0,
            "eager": 0,
            "full": 0,
            "failed": 0,
        }

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, lengths: list[int]
    ) -> torch.Tensor:
        rows, length = input_ids.shape
        padded = any(n != length for n in lengths)
        return self.run(
            (rows, length, padded),
            (input_ids, attention_mask),
            lambda ids, mask: self.eager(ids, mask, padded),
            rows * length,
        )

    def run(
        self,
        key: Hashable,
        inputs: tuple[torch.Tensor, ...],
        body: Callable[..., torch.Tensor],
        tokens: int,
    ) -> torch.Tensor:
        """``body(*inputs)``, replayed from the graph of ``key`` once captured (the module docstring)."""
        entry = self.graphs.get(key)
        if entry is None:
            if (
                key in self.failed
                or tokens > self.max_tokens
                or not torch.is_inference_mode_enabled()
            ):
                self.stats["eager"] += 1
                return body(*inputs)
            if len(self.seen) > 1 << 16:
                self.seen.clear()
            self.seen[key] = self.seen.get(key, 0) + 1
            if self.seen[key] < self.capture_after:
                self.stats["eager"] += 1
                return body(*inputs)
            entry = self._capture(key, inputs, body)
            if entry is None:
                self.stats["eager"] += 1
                return body(*inputs)
        for static, value in zip(entry["inputs"], inputs, strict=True):
            static.copy_(value)
        entry["graph"].replay()
        self.stats["replays"] += 1
        output: torch.Tensor = entry["output"]
        return output

    def eager(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, padded: bool
    ) -> torch.Tensor:
        hidden: torch.Tensor = self.backbone(
            input_ids, attention_mask, masks=self.masks.build(attention_mask, padded)
        )
        return hidden

    def _capture(
        self,
        key: Hashable,
        inputs: tuple[torch.Tensor, ...],
        body: Callable[..., torch.Tensor],
    ) -> dict[str, Any] | None:
        if (
            len(self.graphs) >= self.max_graphs
            or self.output_bytes >= self.max_output_bytes
        ):
            self.stats["full"] += 1
            return None
        static = tuple(value.clone() for value in inputs)
        try:
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]  # torch leaves Stream's constructor unannotated
            stream.wait_stream(torch.cuda.current_stream())
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        body(*static)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                # Device work is serialized, but other threads may still query the device
                # meanwhile (placement); global capture mode would fail them and the capture.
                with torch.cuda.graph(
                    graph, pool=self.pool, capture_error_mode="thread_local"
                ):
                    output = body(*static)
            torch.cuda.synchronize()
        except Exception:
            torch.cuda.synchronize()
            self.failed.add(key)
            self.stats["failed"] += 1
            return None
        size = _output_bytes(output)
        entry = {"inputs": static, "graph": graph, "output": output, "bytes": size}
        self.graphs[key] = entry
        self.output_bytes += size
        self.stats["captures"] += 1
        return entry

    def receipt(self) -> dict[str, Any]:
        return {**self.stats, "cached": len(self.graphs)}
