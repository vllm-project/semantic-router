"""Unit tests for DisaggFleetOptimizer."""

import pytest

from fleet_sim import (
    ALPHA_DEC,
    ALPHA_PRE,
    BETA_TTFT,
    H100_SXM,
    LLAMA_3_1_70B,
    DisaggFleetOptimizer,
    DisaggResult,
    DisaggSweepPoint,
)
from fleet_sim.gpu_profiles import H100_80GB, ProfileBuilder, ServingConfig

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def builder():
    return ProfileBuilder()


@pytest.fixture
def llama_prefill(builder):
    return builder.build(
        H100_SXM,
        LLAMA_3_1_70B,
        ServingConfig(tp=4, dtype_bytes=2, mean_ctx_tokens=1024, phase="prefill"),
    )


@pytest.fixture
def llama_decode(builder):
    return builder.build(
        H100_SXM,
        LLAMA_3_1_70B,
        ServingConfig(tp=8, dtype_bytes=2, mean_ctx_tokens=1024, phase="decode"),
    )


@pytest.fixture
def disagg_llama(llama_prefill, llama_decode):
    return DisaggFleetOptimizer(
        prefill_profile=llama_prefill,
        decode_profile=llama_decode,
        mean_isl=1024,
        mean_osl=256,
        slo_ttft_ms=2000,
        slo_tpot_ms=100,
        max_ctx=4096,
    )


# ── Published constants ───────────────────────────────────────────────────────


class TestPublishedConstants:
    def test_alpha_pre_value(self):
        assert pytest.approx(0.90) == ALPHA_PRE

    def test_alpha_dec_value(self):
        assert pytest.approx(0.92) == ALPHA_DEC

    def test_beta_ttft_value(self):
        assert pytest.approx(1.80) == BETA_TTFT


# ── Optimizer construction ────────────────────────────────────────────────────


class TestDisaggFleetOptimizer:
    def test_accepts_manual_profiles(self):
        opt = DisaggFleetOptimizer(
            prefill_profile=H100_80GB,
            decode_profile=H100_80GB,
            mean_isl=512,
            mean_osl=128,
            slo_ttft_ms=5000,
            slo_tpot_ms=200,
            max_ctx=4096,
        )
        assert opt is not None

    def test_accepts_computed_profiles(self, disagg_llama):
        assert disagg_llama is not None

    def test_optimize_returns_result(self, disagg_llama):
        r = disagg_llama.optimize(max_prefill=4, max_decode=4)
        assert r is not None
        assert isinstance(r, DisaggResult)

    def test_result_has_positive_gpus(self, disagg_llama):
        r = disagg_llama.optimize(max_prefill=4, max_decode=4)
        assert r.total_gpus > 0
        assert r.n_prefill >= 1
        assert r.n_decode >= 1

    def test_result_total_gpus_consistent(
        self, disagg_llama, llama_prefill, llama_decode
    ):
        r = disagg_llama.optimize(max_prefill=4, max_decode=4)
        expected = r.n_prefill * r.prefill_gpus + r.n_decode * r.decode_gpus
        assert r.total_gpus == expected

    def test_result_ttft_includes_beta_correction(self, disagg_llama):
        r = disagg_llama.optimize(max_prefill=4, max_decode=4)
        # Effective TTFT must be at most the SLO
        assert r.ttft_ms <= disagg_llama.slo_ttft_ms + 1e-6

    def test_result_tpot_within_slo(self, disagg_llama):
        r = disagg_llama.optimize(max_prefill=4, max_decode=4)
        assert r.tpot_ms <= disagg_llama.slo_tpot_ms + 1e-6

    def test_result_system_rate_positive(self, disagg_llama):
        r = disagg_llama.optimize(max_prefill=4, max_decode=4)
        assert r.system_rate > 0

    def test_result_cost_positive(self, disagg_llama):
        r = disagg_llama.optimize(max_prefill=4, max_decode=4)
        assert r.cost_per_hr > 0

    def test_infeasible_slo_returns_none(self, llama_prefill, llama_decode):
        # SLO too tight to satisfy
        opt = DisaggFleetOptimizer(
            prefill_profile=llama_prefill,
            decode_profile=llama_decode,
            mean_isl=1024,
            mean_osl=256,
            slo_ttft_ms=1,  # 1ms — impossible
            slo_tpot_ms=0.1,  # 0.1ms — impossible
            max_ctx=4096,
        )
        r = opt.optimize(max_prefill=4, max_decode=4)
        assert r is None

    def test_more_workers_increases_system_rate(self, disagg_llama):
        r_small = disagg_llama.optimize(max_prefill=2, max_decode=2)
        r_large = disagg_llama.optimize(max_prefill=8, max_decode=8)
        assert r_large.system_rate >= r_small.system_rate

    def test_custom_alpha_beta(self, llama_prefill, llama_decode):
        opt_default = DisaggFleetOptimizer(
            llama_prefill,
            llama_decode,
            mean_isl=1024,
            mean_osl=256,
            slo_ttft_ms=2000,
            slo_tpot_ms=100,
            max_ctx=4096,
        )
        opt_optimistic = DisaggFleetOptimizer(
            llama_prefill,
            llama_decode,
            mean_isl=1024,
            mean_osl=256,
            slo_ttft_ms=2000,
            slo_tpot_ms=100,
            max_ctx=4096,
            alpha_pre=1.0,
            alpha_dec=1.0,
            beta_ttft=1.0,
        )
        r_default = opt_default.optimize(max_prefill=4, max_decode=4)
        r_optimistic = opt_optimistic.optimize(max_prefill=4, max_decode=4)
        # With no degradation, system rate should be higher
        assert r_optimistic.system_rate >= r_default.system_rate


class TestDisaggSweep:
    def test_sweep_returns_list(self, disagg_llama):
        pts = disagg_llama.sweep(max_prefill=3, max_decode=3)
        assert isinstance(pts, list)
        assert len(pts) > 0

    def test_sweep_points_have_correct_type(self, disagg_llama):
        pts = disagg_llama.sweep(max_prefill=2, max_decode=2)
        for p in pts:
            assert isinstance(p, DisaggSweepPoint)

    def test_sweep_respects_valid_gpu_counts(self, llama_prefill, llama_decode):
        opt = DisaggFleetOptimizer(
            llama_prefill,
            llama_decode,
            mean_isl=1024,
            mean_osl=256,
            slo_ttft_ms=2000,
            slo_tpot_ms=100,
            max_ctx=4096,
        )
        valid = [32, 64]
        pts = opt.sweep(max_prefill=4, max_decode=4, valid_total_gpus=valid)
        for p in pts:
            assert p.total_gpus in valid

    def test_sweep_print_report_runs(self, disagg_llama):
        r = disagg_llama.optimize(max_prefill=4, max_decode=4)
        if r:
            r.print_report()  # should not raise
