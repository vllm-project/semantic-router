"""Released GatedDeltaNet prep matches the documented q/k/v/g/beta layout."""

from __future__ import annotations

import torch
from vllm_srun.accel.kernels import causal_conv1d_ref, gdn_prep_ref


def test_gdn_prep_ref_splits_conv_silu_and_gates():
    torch.manual_seed(0)
    batch, length, k_heads, head_dim, nv, kernel = 2, 8, 2, 4, 3, 4
    channels = (2 * k_heads + nv) * head_dim
    mixed = torch.randn(batch, length, channels)
    b = torch.randn(batch, length, nv)
    a = torch.randn(batch, length, nv)
    weight = torch.randn(channels, kernel)
    a_log = torch.randn(nv)
    dt = torch.randn(nv)
    query, key, value, gate, beta = gdn_prep_ref(
        mixed, b, a, weight, a_log, dt, k_heads, head_dim
    )
    conv = causal_conv1d_ref(
        mixed.transpose(1, 2).contiguous(), weight, activation="silu"
    ).transpose(1, 2)
    key_dim = k_heads * head_dim
    expect_q, expect_k, expect_v = torch.split(
        conv, [key_dim, key_dim, nv * head_dim], dim=-1
    )
    assert query.shape == (batch, length, k_heads, head_dim)
    assert value.shape == (batch, length, nv, head_dim)
    assert torch.equal(query, expect_q.reshape_as(query))
    assert torch.equal(key, expect_k.reshape_as(key))
    assert torch.equal(value, expect_v.reshape_as(value))
    assert torch.equal(beta, b.sigmoid())
    expect_g = -a_log.float().exp() * torch.nn.functional.softplus(a.float() + dt)
    assert torch.equal(gate, expect_g)
