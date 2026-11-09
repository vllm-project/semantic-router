"""Attention prep agrees with the existing decoder norm and rotary path."""

from __future__ import annotations

import pytest
import torch
from vllm_srun.accel.kernels import attn_prep_ref
from vllm_srun.engines.native.models.common import (
    RMSNorm,
    ZeroCenteredRMSNorm,
    apply_partial_rotary,
    apply_rotary,
)


@pytest.mark.parametrize(
    "gated,dim,rot", [(True, 128, 32), (True, 256, 64), (False, 128, 128)]
)
def test_attn_prep_ref_matches_decoder_operations(gated, dim, rot):
    torch.manual_seed(0)
    batch, length, heads, kv = 2, 4, 4, 2
    width = dim * (2 if gated else 1)
    q = torch.randn(batch, length, heads * width, dtype=torch.bfloat16)
    k = torch.randn(batch, length, kv * dim, dtype=torch.bfloat16)
    norm_type = ZeroCenteredRMSNorm if gated else RMSNorm
    q_norm, k_norm = norm_type(dim, 1e-6), norm_type(dim, 1e-6)
    with torch.no_grad():
        q_norm.weight.normal_()
        k_norm.weight.normal_()
    qw = 1.0 + q_norm.weight if gated else q_norm.weight
    kw = 1.0 + k_norm.weight if gated else k_norm.weight
    cos = torch.randn(batch, length, rot, dtype=torch.bfloat16)
    sin = torch.randn(batch, length, rot, dtype=torch.bfloat16)
    query, key = attn_prep_ref(
        q,
        k,
        qw,
        kw,
        cos,
        sin,
        heads,
        kv,
        dim,
        1e-6,
        gated=gated,
        zero_centred=gated,
    )
    projected_q = q.reshape(batch, length, heads, width)
    if gated:
        projected_q, _ = projected_q.chunk(2, dim=-1)
    expected_q = q_norm(projected_q).transpose(1, 2)
    expected_k = k_norm(k.reshape(batch, length, kv, dim)).transpose(1, 2)
    rotary = apply_partial_rotary if gated else apply_rotary
    expected_q, expected_k = rotary(expected_q, expected_k, cos, sin)
    assert query.shape == (batch, heads, length, dim)
    assert key.shape == (batch, kv, length, dim)
    assert query.dtype == key.dtype == torch.bfloat16
    # Qwen3 converts to the value dtype at the attention boundary.
    assert torch.equal(query, expected_q.to(q.dtype))
    assert torch.equal(key, expected_k.to(k.dtype))
