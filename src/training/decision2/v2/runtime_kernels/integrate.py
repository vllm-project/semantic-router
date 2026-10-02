"""Fused-kernel forward for the Qwen3.5 decoder layers of a loaded package (research only).

``FusedLayers(backbone, mode)`` patches every ``Qwen3_5DecoderLayer`` instance in place
(restored on exit). Modes:

    replay   the ``reference`` op sequences only: must reproduce the shipped forward bit for bit
    shadow   reference trajectory; every fused kernel also runs on the same real activations
             and its output is compared (``stats``), the reference output is carried on
    fused    the fused kernels only (GEMMs, SDPA and FLA's chunk stage unchanged)

The fused GDN path feeds FLA's grouped-value path with q/k at the key-head count, already
L2-normalised by ``gdn_prep`` (bit-identical to the repeated, in-kernel-normalised call).
Linear layers (including PEFT LoRA wrappers) are called as modules, so 27B's unmerged
adapter runs unchanged.
"""

from __future__ import annotations

import types
from typing import Any, Callable

import torch

from . import reference as ref
from . import triton_elementwise as tk
from .fidelity import compare


class Stats:
    """Per-kernel running fidelity: worst match, worst relative L2, worst ULP, element-weighted match."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, float]] = {}

    def add(self, name: str, result: dict[str, float]) -> None:
        row = self.rows.setdefault(
            name,
            {
                "calls": 0,
                "elements": 0,
                "matched": 0.0,
                "worst_match": 1.0,
                "worst_rel_l2": 0.0,
                "max_ulp": 0,
            },
        )
        row["calls"] += 1
        row["elements"] += result["elements"]
        row["matched"] += result["match"] * result["elements"]
        row["worst_match"] = min(row["worst_match"], result["match"])
        row["worst_rel_l2"] = max(row["worst_rel_l2"], result["rel_l2"])
        row["max_ulp"] = max(row["max_ulp"], int(result["max_ulp"]))

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            name: {**row, "match": row["matched"] / max(1, row["elements"])}
            for name, row in sorted(self.rows.items())
        }


class FusedLayers:
    def __init__(self, backbone: Any, mode: str = "fused", stats: Stats | None = None):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

        if mode not in ("replay", "shadow", "fused"):
            raise ValueError(mode)
        self.mode = mode
        self.stats = stats if stats is not None else Stats()
        self.layers = [
            m for m in backbone.modules() if isinstance(m, Qwen3_5DecoderLayer)
        ]
        if not self.layers:
            raise ValueError("no Qwen3_5DecoderLayer in this backbone")
        self.saved: list[tuple[Any, Any]] = []

    # -- op dispatch -------------------------------------------------------------------------
    def op(
        self,
        name: str,
        reference: Callable[[], Any],
        fused: Callable[[], Any],
        view: Callable | None = None,
    ):
        if self.mode == "replay":
            return reference()
        if self.mode == "fused":
            return fused()
        r, f = reference(), fused()
        pairs = (
            view(r, f)
            if view
            else ((r, f) if not isinstance(r, tuple) else tuple(zip(r, f)))
        )
        if isinstance(pairs, tuple) and pairs and isinstance(pairs[0], tuple):
            for i, (a, b) in enumerate(pairs):
                self.stats.add(f"{name}.{i}", compare(torch, a, b))
        else:
            self.stats.add(name, compare(torch, *pairs))
        return r

    # -- patching ------------------------------------------------------------------------------
    def __enter__(self) -> "FusedLayers":
        for layer in self.layers:
            self._prepare(layer)
            self.saved.append((layer, layer.forward))
            layer.forward = types.MethodType(_make_forward(self), layer)
        return self

    def __exit__(self, *exc: Any) -> None:
        for layer, forward in self.saved:
            layer.forward = forward
        self.saved.clear()

    @staticmethod
    def _prepare(layer: Any) -> None:
        p: dict[str, Any] = {
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
                gn_w=m.norm.weight,
                gn_eps=m.norm.variance_epsilon,
            )
        else:
            m = layer.self_attn
            p.update(
                qw1=(1.0 + m.q_norm.weight.float()).contiguous(),
                kw1=(1.0 + m.k_norm.weight.float()).contiguous(),
            )
        layer._rk = p


def _make_forward(F: FusedLayers) -> Callable:
    def forward(
        layer,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        p = layer._rk
        eps = p["eps"]
        normed = F.op(
            "rmsnorm_in",
            lambda: ref.add_rmsnorm(
                torch, hidden_states, None, layer.input_layernorm.weight, eps
            )[1],
            lambda: tk.add_rmsnorm(hidden_states, None, p["w1_in"], eps)[1],
        )
        if layer.block_type == "linear_attention":
            delta = _gdn(F, layer.linear_attn, p, normed, attention_mask)
        else:
            delta = _attn(
                F, layer.self_attn, p, normed, position_embeddings, attention_mask
            )
        hidden, normed2 = F.op(
            "add_rmsnorm_post",
            lambda: ref.add_rmsnorm(
                torch, hidden_states, delta, layer.post_attention_layernorm.weight, eps
            ),
            lambda: tk.add_rmsnorm(hidden_states, delta, p["w1_post"], eps),
        )
        mlp = layer.mlp
        gate, up = mlp.gate_proj(normed2), mlp.up_proj(normed2)
        act = F.op(
            "silu_mul",
            lambda: ref.silu_mul(torch, gate, up),
            lambda: tk.silu_mul(gate, up),
        )
        return hidden + mlp.down_proj(act)

    return forward


def _gdn(
    F: FusedLayers, m: Any, p: dict[str, Any], normed: Any, attention_mask: Any
) -> Any:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    if attention_mask is not None:
        normed = (normed * attention_mask[:, :, None]).to(normed.dtype)
    B, T, _ = normed.shape
    mixed = m.in_proj_qkv(normed)
    z = m.in_proj_z(normed)
    b = m.in_proj_b(normed)
    a = m.in_proj_a(normed)
    nk, dk, dv = m.num_k_heads, m.head_k_dim, m.head_v_dim

    def reference():
        q, k, v, g, beta = ref.gdn_prep(
            torch,
            mixed,
            b,
            a,
            p["conv_w"],
            p["A_log"],
            p["dt_bias"],
            nk,
            dk,
            dv,
            expand_qk=True,
            l2norm=False,
        )
        o, _ = chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True
        )
        return o

    def fused():
        q, k, v, g, beta = tk.gdn_prep(
            mixed, b, a, p["conv_w"], p["A_log"], p["dt_bias"], nk, dk
        )
        o, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta)
        return o

    if F.mode == "shadow":
        # prep against the reference's own pre-FLA tensors (q/k compared at the key-head count)
        F.op(
            "gdn_prep",
            lambda: ref.gdn_prep(
                torch,
                mixed,
                b,
                a,
                p["conv_w"],
                p["A_log"],
                p["dt_bias"],
                nk,
                dk,
                dv,
                expand_qk=False,
            ),
            lambda: tk.gdn_prep(
                mixed, b, a, p["conv_w"], p["A_log"], p["dt_bias"], nk, dk
            ),
        )
    core = F.op("gdn_core", reference, fused)
    out = F.op(
        "gated_rmsnorm",
        lambda: ref.gated_rmsnorm(
            torch, core.reshape(-1, dv), z.reshape(-1, dv), p["gn_w"], p["gn_eps"]
        ),
        lambda: tk.gated_rmsnorm(
            core.reshape(-1, dv), z.reshape(-1, dv), p["gn_w"], p["gn_eps"]
        ),
    )
    return m.out_proj(out.reshape(B, T, -1))


def _attn(
    F: FusedLayers,
    m: Any,
    p: dict[str, Any],
    normed: Any,
    position_embeddings: Any,
    attention_mask: Any,
) -> Any:
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    B, T, _ = normed.shape
    hd = m.head_dim
    qp, kp, vp = m.q_proj(normed), m.k_proj(normed), m.v_proj(normed)
    cos, sin = position_embeddings
    nh = qp.shape[-1] // (2 * hd)
    nkv = kp.shape[-1] // hd
    eps = m.q_norm.eps

    prep = F.op(
        "attn_prep",
        lambda: ref.attn_prep(
            torch, qp, kp, vp, m.q_norm.weight, m.k_norm.weight, cos, sin, hd, eps
        )[:2],
        lambda: tk.attn_prep(qp, kp, p["qw1"], p["kw1"], cos, sin, nh, nkv, hd, eps),
    )
    q, k = prep
    v = vp.view(B, T, nkv, hd).transpose(1, 2)
    attn, _ = sdpa_attention_forward(
        m, q, k, v, attention_mask, dropout=0.0, scaling=m.scaling
    )
    gate_view = qp.view(B, T, nh, 2 * hd)[..., hd:]
    gated = F.op(
        "sigmoid_gate",
        lambda: ref.sigmoid_gate(torch, attn, gate_view.reshape(B, T, -1)),
        lambda: tk.sigmoid_gate(attn, gate_view),
    )
    return m.o_proj(gated)
