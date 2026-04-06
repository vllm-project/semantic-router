"""Unit tests for ProfileBuilder, ComputedProfile, ManualProfile, and Protocol."""

import math

import pytest

from fleet_sim.gpu_profiles import (
    A10G,
    A100_80GB,
    H100_80GB,
    GpuProfile,
    ProfileBuilder,
    ServingConfig,
)
from fleet_sim.hardware import A100_SXM, B200_SXM, H100_SXM, H200_SXM
from fleet_sim.models import DEEPSEEK_V3, LLAMA_3_1_8B, LLAMA_3_1_70B, QWEN3_235B_A22B

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def builder():
    return ProfileBuilder()


@pytest.fixture
def cfg_tp8_fp16():
    return ServingConfig(tp=8, dtype_bytes=2, mean_ctx_tokens=2048)


@pytest.fixture
def h100_llama70b(builder, cfg_tp8_fp16):
    return builder.build(H100_SXM, LLAMA_3_1_70B, cfg_tp8_fp16)


@pytest.fixture
def h100_deepseek_fp8(builder):
    cfg = ServingConfig(tp=8, ep=8, dtype_bytes=1, mean_ctx_tokens=2048)
    return builder.build(H100_SXM, DEEPSEEK_V3, cfg)


# ── Protocol compliance ───────────────────────────────────────────────────────


class TestGpuProfileProtocol:
    def test_manual_profile_satisfies_protocol(self):
        assert isinstance(A100_80GB, GpuProfile)
        assert isinstance(H100_80GB, GpuProfile)
        assert isinstance(A10G, GpuProfile)

    def test_computed_profile_satisfies_protocol(self, h100_llama70b):
        assert isinstance(h100_llama70b, GpuProfile)

    def test_protocol_methods_present(self, h100_llama70b):
        assert hasattr(h100_llama70b, "iter_latency")
        assert hasattr(h100_llama70b, "n_slots")
        assert hasattr(h100_llama70b, "service_time")
        assert hasattr(h100_llama70b, "throughput")
        assert hasattr(h100_llama70b, "cost_per_hr")
        assert hasattr(h100_llama70b, "name")


# ── ManualProfile ─────────────────────────────────────────────────────────────


class TestManualProfile:
    def test_iter_latency_linear(self):
        p = A100_80GB
        assert p.iter_latency(0) == pytest.approx(p.W)
        assert p.iter_latency(10) == pytest.approx(p.W + 10 * p.H)

    def test_iter_latency_scales_with_mean_seq_len(self):
        """H_eff = H × (mean_seq_len / calibration_ctx): halving seq length halves H term."""
        p = A100_80GB
        n = 64
        full = p.iter_latency(n, mean_seq_len=float(p.calibration_ctx))
        half = p.iter_latency(n, mean_seq_len=float(p.calibration_ctx) / 2)
        # Full-ctx iter_t = W + H*n ; half-ctx = W + (H/2)*n
        expected_full = p.W + p.H * n
        expected_half = p.W + (p.H / 2) * n
        assert full == pytest.approx(expected_full)
        assert half == pytest.approx(expected_half)

    def test_iter_latency_no_seq_len_uses_calibration(self):
        """Omitting mean_seq_len should match passing calibration_ctx explicitly."""
        p = A100_80GB
        assert p.iter_latency(32) == pytest.approx(
            p.iter_latency(32, mean_seq_len=float(p.calibration_ctx))
        )

    def test_n_slots_equals_kv_limit_at_calibration_ctx(self):
        """At calibration_ctx both limits are equal by construction."""
        p = A100_80GB
        blks_per_seq = math.ceil(p.calibration_ctx / p.blk_size)
        kv_limit = p.total_kv_blks // blks_per_seq
        compute_cap = p.max_slots  # calibration_ctx / calibration_ctx == 1
        assert p.n_slots(p.calibration_ctx) == min(kv_limit, compute_cap)

    def test_n_slots_scales_inversely_below_calibration(self):
        """At max_ctx = calibration_ctx/2, n_slots should be 2× the calibration value."""
        p = A100_80GB
        slots_calib = p.n_slots(p.calibration_ctx)
        slots_half = p.n_slots(p.calibration_ctx // 2)
        # Both limits double when max_ctx halves
        assert slots_half == pytest.approx(slots_calib * 2, abs=1)

    def test_n_slots_decreases_with_larger_ctx(self):
        s8k = A100_80GB.n_slots(8192)
        s32k = A100_80GB.n_slots(32768)
        assert s8k > s32k

    def test_short_pool_has_higher_throughput_than_homo(self):
        """A GPU configured for short ctx should serve more req/s than homo ctx."""
        p = A100_80GB
        # 500-token request
        tput_short = p.throughput(max_ctx=2048, mean_l_in=400.0, mean_l_out=100.0)
        tput_homo = p.throughput(
            max_ctx=p.calibration_ctx, mean_l_in=400.0, mean_l_out=100.0
        )
        assert tput_short > tput_homo

    def test_service_time_positive(self):
        st = A100_80GB.service_time(l_in=512, l_out=256, max_ctx=4096)
        assert st > 0

    def test_throughput_positive(self):
        tp = A100_80GB.throughput(max_ctx=4096, mean_l_in=512.0, mean_l_out=256.0)
        assert tp > 0


# ── ProfileBuilder — dense models ────────────────────────────────────────────


class TestProfileBuilderDense:
    def test_W_faster_on_h100_than_a100(self, builder, cfg_tp8_fp16):
        a100 = builder.build(A100_SXM, LLAMA_3_1_70B, cfg_tp8_fp16)
        h100 = builder.build(H100_SXM, LLAMA_3_1_70B, cfg_tp8_fp16)
        # H100 is faster → lower W
        assert h100.W < a100.W

    def test_W_faster_on_h200_than_h100(self, builder, cfg_tp8_fp16):
        h100 = builder.build(H100_SXM, LLAMA_3_1_70B, cfg_tp8_fp16)
        h200 = builder.build(H200_SXM, LLAMA_3_1_70B, cfg_tp8_fp16)
        assert h200.W < h100.W

    def test_W_scales_with_model_size(self, builder, cfg_tp8_fp16):
        p8b = builder.build(H100_SXM, LLAMA_3_1_8B, cfg_tp8_fp16)
        p70b = builder.build(H100_SXM, LLAMA_3_1_70B, cfg_tp8_fp16)
        # 70B has ~8.75× more params → W should be proportionally larger
        assert p70b.W > p8b.W

    def test_W_fp8_lower_than_fp16(self, builder):
        cfg_fp16 = ServingConfig(tp=8, dtype_bytes=2, mean_ctx_tokens=2048)
        cfg_fp8 = ServingConfig(tp=8, dtype_bytes=1, mean_ctx_tokens=2048)
        fp16 = builder.build(H100_SXM, LLAMA_3_1_70B, cfg_fp16)
        fp8 = builder.build(H100_SXM, LLAMA_3_1_70B, cfg_fp8)
        assert fp8.W < fp16.W
        # Should be roughly half (fp8 weights are half the size)
        assert pytest.approx(0.5, rel=0.05) == fp8.W / fp16.W

    def test_H_positive(self, h100_llama70b):
        assert h100_llama70b.H > 0

    def test_H_higher_tp_lower(self, builder):
        # Higher TP splits KV reads across more GPUs → lower per-GPU H
        cfg_tp4 = ServingConfig(tp=4, dtype_bytes=2, mean_ctx_tokens=2048)
        cfg_tp8 = ServingConfig(tp=8, dtype_bytes=2, mean_ctx_tokens=2048)
        p_tp4 = builder.build(H100_SXM, LLAMA_3_1_70B, cfg_tp4)
        p_tp8 = builder.build(H100_SXM, LLAMA_3_1_70B, cfg_tp8)
        assert p_tp8.H < p_tp4.H

    def test_W_in_plausible_range_h100_70b(self, h100_llama70b):
        # H100 Llama-70B TP=8: should be 3–15ms
        assert 0.003 <= h100_llama70b.W <= 0.015

    def test_W_in_plausible_range_a100_70b(self, builder, cfg_tp8_fp16):
        a100 = builder.build(A100_SXM, LLAMA_3_1_70B, cfg_tp8_fp16)
        # A100 is slower: 5–20ms
        assert 0.005 <= a100.W <= 0.020

    def test_kv_blks_positive(self, h100_llama70b):
        assert h100_llama70b.total_kv_blks > 0

    def test_kv_blks_more_on_h200_due_to_vram(self, builder, cfg_tp8_fp16):
        h100 = builder.build(H100_SXM, LLAMA_3_1_70B, cfg_tp8_fp16)
        h200 = builder.build(H200_SXM, LLAMA_3_1_70B, cfg_tp8_fp16)
        assert h200.total_kv_blks > h100.total_kv_blks

    def test_cost_scales_with_tp(self, builder):
        cfg_tp4 = ServingConfig(tp=4, dtype_bytes=2, mean_ctx_tokens=2048)
        cfg_tp8 = ServingConfig(tp=8, dtype_bytes=2, mean_ctx_tokens=2048)
        p_tp4 = builder.build(H100_SXM, LLAMA_3_1_70B, cfg_tp4)
        p_tp8 = builder.build(H100_SXM, LLAMA_3_1_70B, cfg_tp8)
        assert p_tp8.cost_per_hr / p_tp4.cost_per_hr == pytest.approx(2.0, rel=0.01)


# ── ProfileBuilder — MoE models ──────────────────────────────────────────────


class TestProfileBuilderMoE:
    def test_moe_W_larger_than_comparable_dense(self, builder):
        cfg = ServingConfig(tp=8, ep=8, dtype_bytes=1, mean_ctx_tokens=2048)
        # DeepSeek-V3 fp8 vs Llama-70B fp8 — MoE has more weight to dispatch
        moe = builder.build(H100_SXM, DEEPSEEK_V3, cfg)
        dense = builder.build(H100_SXM, LLAMA_3_1_70B, cfg)
        assert moe.W > dense.W

    def test_moe_w_in_plausible_range_deepseek_fp8(self, h100_deepseek_fp8):
        # DeepSeek-V3 fp8 on H100: 61 layers × ~0.7ms/layer ≈ 40–80ms
        assert 0.030 <= h100_deepseek_fp8.W <= 0.120

    def test_moe_fp8_faster_than_fp16(self, builder):
        cfg_fp16 = ServingConfig(tp=8, ep=8, dtype_bytes=2, mean_ctx_tokens=2048)
        cfg_fp8 = ServingConfig(tp=8, ep=8, dtype_bytes=1, mean_ctx_tokens=2048)
        fp16 = builder.build(H100_SXM, DEEPSEEK_V3, cfg_fp16)
        fp8 = builder.build(H100_SXM, DEEPSEEK_V3, cfg_fp8)
        assert fp8.W < fp16.W

    def test_moe_b200_faster_than_h100(self, builder):
        cfg = ServingConfig(tp=8, ep=8, dtype_bytes=1, mean_ctx_tokens=2048)
        h100 = builder.build(H100_SXM, DEEPSEEK_V3, cfg)
        b200 = builder.build(B200_SXM, DEEPSEEK_V3, cfg)
        assert b200.W < h100.W

    def test_qwen3_235b_moe_faster_than_deepseek_v3(self, builder):
        # Qwen3-235B has fewer layers and smaller hidden → faster per step
        cfg = ServingConfig(tp=8, ep=8, dtype_bytes=1, mean_ctx_tokens=2048)
        qwen = builder.build(H100_SXM, QWEN3_235B_A22B, cfg)
        dsv3 = builder.build(H100_SXM, DEEPSEEK_V3, cfg)
        assert qwen.W < dsv3.W

    def test_moe_protocol_compliance(self, h100_deepseek_fp8):
        assert isinstance(h100_deepseek_fp8, GpuProfile)


# ── ComputedProfile methods ───────────────────────────────────────────────────


class TestComputedProfileMethods:
    def test_iter_latency_increases_with_batch(self, h100_llama70b):
        lat1 = h100_llama70b.iter_latency(1)
        lat64 = h100_llama70b.iter_latency(64)
        assert lat64 > lat1

    def test_n_slots_decreases_with_ctx(self, h100_llama70b):
        s4k = h100_llama70b.n_slots(4096)
        s32k = h100_llama70b.n_slots(32768)
        assert s4k > s32k

    def test_n_slots_at_least_1(self, h100_llama70b):
        assert h100_llama70b.n_slots(512) >= 1

    def test_service_time_positive(self, h100_llama70b):
        st = h100_llama70b.service_time(l_in=512, l_out=256, max_ctx=4096)
        assert st > 0

    def test_service_time_longer_output_takes_longer(self, h100_llama70b):
        st_short = h100_llama70b.service_time(512, 64, 4096)
        st_long = h100_llama70b.service_time(512, 512, 4096)
        assert st_long > st_short

    def test_throughput_positive(self, h100_llama70b):
        tp = h100_llama70b.throughput(4096, 512.0, 256.0)
        assert tp > 0

    def test_chunk_attribute_exposed(self, h100_llama70b):
        assert h100_llama70b.chunk == 512

    def test_name_encodes_hardware_model_tp(self, h100_llama70b):
        name = h100_llama70b.name
        assert "H100" in name
        assert "70B" in name or "Llama" in name
        assert "TP8" in name

    def test_summary_returns_string(self, h100_llama70b):
        s = h100_llama70b.summary()
        assert isinstance(s, str)
        assert "ms" in s

    def test_frozen_immutable(self, h100_llama70b):
        with pytest.raises((AttributeError, TypeError)):
            h100_llama70b.W = 0.0

    def test_prefill_compute_bound_on_a100_large_model(self, builder, cfg_tp8_fp16):
        """A100 has lower FLOP/byte ratio than H100.

        For Llama-3.1-70B with chunk=512, the complete FLOPs (FFN + projections
        + attention) should make prefill compute-bound: prefill_iter > decode_iter.
        The old attention-only formula gave ~0.7ms compute vs ~12ms memory,
        always returning the (incorrect) memory-bound answer.
        """
        a100 = builder.build(A100_SXM, LLAMA_3_1_70B, cfg_tp8_fp16)
        n = a100.n_slots(8192)
        decode_t = a100.iter_latency(n)
        prefill_t = a100.prefill_iter_latency(
            chunk_tokens=512, kv_history_tokens=1024, n_active=n
        )
        # Prefill must be more expensive than a single decode step
        assert prefill_t > decode_t, (
            f"A100 prefill ({prefill_t*1000:.1f}ms) should exceed decode "
            f"({decode_t*1000:.1f}ms) for chunk=512 on 70B model"
        )

    def test_prefill_compute_bound_on_h100_large_model(self, builder, cfg_tp8_fp16):
        """H100 is closer to the compute/memory boundary than A100."""
        h100 = builder.build(H100_SXM, LLAMA_3_1_70B, cfg_tp8_fp16)
        n = h100.n_slots(8192)
        decode_t = h100.iter_latency(n)
        prefill_t = h100.prefill_iter_latency(
            chunk_tokens=512, kv_history_tokens=1024, n_active=n
        )
        assert prefill_t > decode_t, (
            f"H100 prefill ({prefill_t*1000:.1f}ms) should exceed decode "
            f"({decode_t*1000:.1f}ms) for chunk=512 on 70B model"
        )

    def test_int4_profile_builds_without_zero_w(self, builder):
        """int4 dtype (bytes=0.5) must not zero out W via int(0.5)=0."""
        cfg_int4 = ServingConfig(tp=4, dtype_bytes=0.5, mean_ctx_tokens=2048)
        cfg_fp16 = ServingConfig(tp=4, dtype_bytes=2.0, mean_ctx_tokens=2048)
        p_int4 = builder.build(H100_SXM, LLAMA_3_1_8B, cfg_int4)
        p_fp16 = builder.build(H100_SXM, LLAMA_3_1_8B, cfg_fp16)
        assert p_int4.W > 0, "W must be non-zero for int4"
        # int4 weights are 1/4 the size → W must be meaningfully less than fp16 W.
        # W = streaming_time + layer_overhead; only the streaming term scales with
        # dtype_bytes, so the ratio is not exactly 0.25 but W_int4 < W_fp16/2.
        assert p_int4.W < p_fp16.W / 2
        assert p_int4.H > 0, "H must be non-zero for int4"
        assert p_int4.total_kv_blks > 0, "KV blocks must be positive for int4"

    def test_int4_kv_blks_greater_than_fp16(self, builder):
        """int4 KV cache uses 1/4 the bytes → more KV blocks fit."""
        cfg_int4 = ServingConfig(tp=8, dtype_bytes=0.5, mean_ctx_tokens=2048)
        cfg_fp16 = ServingConfig(tp=8, dtype_bytes=2.0, mean_ctx_tokens=2048)
        p_int4 = builder.build(H100_SXM, LLAMA_3_1_70B, cfg_int4)
        p_fp16 = builder.build(H100_SXM, LLAMA_3_1_70B, cfg_fp16)
        assert p_int4.total_kv_blks > p_fp16.total_kv_blks
