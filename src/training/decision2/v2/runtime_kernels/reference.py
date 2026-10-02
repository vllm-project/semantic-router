"""Reference op sequences of the shipped runtime, one function per fused-kernel candidate.

The released runtime runs Transformers' Qwen3.5 modules under BF16 autocast
with every parameter in FP32 except the BF16-exact Linear weights (held in
BF16), so the residual stream, every norm and the gate parameters stay FP32
while each Linear rounds its input to BF16 and returns BF16. These functions
replay exactly the eager ops of Transformers 5.17 (``modeling_qwen3_5``) and
FLA 0.5.2 for one layer, including the BF16 casts autocast inserts in front
of every Linear, so a fused kernel can be timed and compared against them.
Call them under ``torch.autocast("cuda", torch.bfloat16)`` like the runtime;
the casts that autocast would add at the following Linear are written out.
"""

from __future__ import annotations

from typing import Any


def add_rmsnorm(
    torch: Any,
    residual: Any,
    delta: Any | None,
    weight: Any,
    eps: float,
    gemm_inputs: int = 1,
):
    """``hidden = residual + delta`` then ``Qwen3_5RMSNorm(hidden)`` and the Linear-input cast(s).

    Returns (hidden FP32, normed BF16). ``gemm_inputs`` is how many Linears read the
    normed tensor (autocast casts it once per Linear: 4 in a Gated DeltaNet
    layer, 3 in full attention, 2 in the MLP).
    """
    hidden = residual if delta is None else residual + delta
    x = hidden.float()
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    out = (out * (1.0 + weight.float())).type_as(hidden)
    normed = None
    for _ in range(gemm_inputs):
        normed = out.to(torch.bfloat16)
    return hidden, normed


def silu_mul(torch: Any, gate: Any, up: Any):
    """``act_fn(gate_proj(x)) * up_proj(x)`` on the two BF16 GEMM outputs."""
    return torch.nn.functional.silu(gate) * up


def causal_conv1d_silu(torch: Any, mixed_qkv: Any, conv_weight: Any):
    """Transformers' ``causal_conv1d_fn`` dispatch: the causal-conv1d package when importable.

    ``mixed_qkv`` is the [B, T, C] BF16 output of ``in_proj_qkv``; returns [B, T, C] BF16.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    x = mixed_qkv.transpose(1, 2)
    out = m.causal_conv1d_fn(x, conv_weight, None, activation="silu")
    return out.transpose(1, 2)


def gdn_prep(
    torch: Any,
    mixed_qkv: Any,
    b: Any,
    a: Any,
    conv_weight: Any,
    A_log: Any,
    dt_bias: Any,
    k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    expand_qk: bool = True,
    l2norm: bool = True,
):
    """Gated DeltaNet input prep up to the inputs of FLA's chunk kernels.

    conv + SiLU, q/k/v split, ``beta = b.sigmoid()``, ``g = -exp(A_log) * softplus(a + dt_bias)``,
    the q/k head expansion (``repeat_interleave``) and FLA's in-kernel q/k L2 norm
    (``use_qk_l2norm_in_kernel=True`` runs ``l2norm_fwd`` on q and k first).
    Returns (q, k, v, g, beta) as FLA's chunk stage receives them.
    """
    import torch.nn.functional as F

    B, T, _ = mixed_qkv.shape
    key_dim = k_heads * head_k_dim
    x = causal_conv1d_silu(torch, mixed_qkv, conv_weight)
    query, key, value = torch.split(
        x, [key_dim, key_dim, x.shape[-1] - 2 * key_dim], dim=-1
    )
    query = query.reshape(B, T, -1, head_k_dim)
    key = key.reshape(B, T, -1, head_k_dim)
    value = value.reshape(B, T, -1, head_v_dim)
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    v_heads = value.shape[2]
    if expand_qk and v_heads // k_heads > 1:
        query = query.repeat_interleave(v_heads // k_heads, dim=2)
        key = key.repeat_interleave(v_heads // k_heads, dim=2)
    # FLA's ``input_guard`` makes every tensor argument contiguous before the kernels run
    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    if l2norm:
        from fla.modules.l2norm import l2norm_fwd

        query, _ = l2norm_fwd(query)
        key, _ = l2norm_fwd(key)
    return query, key, value, g, beta


def gated_rmsnorm(torch: Any, core: Any, z: Any, weight: Any, eps: float):
    """``Qwen3_5RMSNormGated(core, z)`` on [N, head_v_dim] rows, then the out_proj input cast."""
    input_dtype = core.dtype
    hidden = core.to(torch.float32)
    variance = hidden.pow(2).mean(-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + eps)
    hidden = weight * hidden.to(input_dtype)
    hidden = hidden * torch.nn.functional.silu(z.to(torch.float32))
    return hidden.to(input_dtype).to(torch.bfloat16)


def head_rmsnorm(torch: Any, x: Any, weight: Any, eps: float):
    """``Qwen3_5RMSNorm`` on the head dim (q_norm / k_norm): FP32 math, result in x's dtype."""
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out.type_as(x)


def rotary_cos_sin(torch: Any, inv_freq: Any, positions: Any, dtype: Any):
    """``Qwen3_5TextRotaryEmbedding`` for text positions ([B, T] int): cos/sin [B, T, rotary_dim].

    For text the three M-RoPE position rows are equal, so the section
    recomposition is the identity and this is plain partial RoPE.
    """
    with torch.autocast("cuda", enabled=False):
        freqs = (
            inv_freq[None, :, None].float() @ positions[:, None, :].float()
        ).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def attn_prep(
    torch: Any,
    q_proj_out: Any,
    k_proj_out: Any,
    v_proj_out: Any,
    q_norm_w: Any,
    k_norm_w: Any,
    cos: Any,
    sin: Any,
    head_dim: int,
    eps: float,
):
    """``Qwen3_5Attention`` from the three projections to SDPA's BF16 inputs.

    q/gate split, q_norm / k_norm, the [B, H, T, D] transposes, partial RoPE in FP32
    (cos/sin are FP32 because the rotary module sees the FP32 hidden states) and the
    BF16 casts autocast applies at SDPA. Returns (q, k, v, gate) with gate [B, T, H*D].
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    input_shape = q_proj_out.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)
    query, gate = torch.chunk(
        q_proj_out.view(*input_shape, -1, head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)
    query = head_rmsnorm(torch, query.view(hidden_shape), q_norm_w, eps).transpose(1, 2)
    key = head_rmsnorm(torch, k_proj_out.view(hidden_shape), k_norm_w, eps).transpose(
        1, 2
    )
    value = v_proj_out.view(hidden_shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    return query.to(torch.bfloat16), key.to(torch.bfloat16), value, gate


def sdpa(torch: Any, q: Any, k: Any, v: Any, mask: Any | None, scale: float):
    """Transformers' ``sdpa_attention_forward`` core: GQA SDPA, causal when no mask, [B, T, H, D] out."""
    groups = q.shape[1] // k.shape[1]
    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        scale=scale,
        is_causal=mask is None,
        enable_gqa=groups > 1,
    )
    return out.transpose(1, 2).contiguous()


def sigmoid_gate(torch: Any, attn_out: Any, gate: Any):
    """``attn_output.reshape(*input_shape, -1).contiguous() * torch.sigmoid(gate)``.

    ``attn_out`` is SDPA's [B, T, H, D] output after Transformers' transpose.
    """
    B, T = attn_out.shape[:2]
    return attn_out.reshape(B, T, -1).contiguous() * torch.sigmoid(gate)
