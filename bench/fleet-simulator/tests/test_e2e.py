"""End-to-end tests covering the full simulation and optimization pipelines.

These tests exercise complete workflows from profile construction through
fleet sizing and DES verification.  They are slower than unit tests but
validate the integrated system.
"""

import json
from pathlib import Path

import pytest

# Path helpers
_DATA = Path(__file__).parent.parent / "data"
_AZURE_CDF = _DATA / "azure_cdf.json"


def _load_cdf(name="azure"):
    path = _DATA / f"{name}_cdf.json"
    raw = json.loads(path.read_text())
    return raw["cdf"] if isinstance(raw, dict) else raw


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def azure_cdf():
    return _load_cdf("azure")


@pytest.fixture(scope="module")
def builder():
    from fleet_sim.gpu_profiles import ProfileBuilder

    return ProfileBuilder()


@pytest.fixture(scope="module")
def h100_llama70b(builder):
    from fleet_sim import H100_SXM, LLAMA_3_1_70B
    from fleet_sim.gpu_profiles import ServingConfig

    return builder.build(
        H100_SXM,
        LLAMA_3_1_70B,
        ServingConfig(tp=8, dtype_bytes=2, mean_ctx_tokens=2048),
    )


# ── E2E: Aggregated optimizer ─────────────────────────────────────────────────


class TestAggregatedOptimizerE2E:
    def test_optimize_a100_returns_slo_met_result(self, azure_cdf):
        from fleet_sim import A100_80GB, FleetOptimizer

        opt = FleetOptimizer(
            gpu_short=A100_80GB, gpu_long=A100_80GB, B_short=4096, t_slo_ms=500
        )
        report = opt.optimize(
            cdf=azure_cdf, lam=50, gammas=[1.0], n_sim_requests=0, verbose=False
        )
        best = report.best_analytical
        assert best is not None
        assert best.slo_met

    def test_higher_load_needs_more_gpus(self, azure_cdf):
        from fleet_sim import A100_80GB, FleetOptimizer

        opt = FleetOptimizer(
            gpu_short=A100_80GB, gpu_long=A100_80GB, B_short=4096, t_slo_ms=500
        )
        r_low = opt.optimize(
            cdf=azure_cdf, lam=20, gammas=[1.0], n_sim_requests=0, verbose=False
        )
        r_high = opt.optimize(
            cdf=azure_cdf, lam=100, gammas=[1.0], n_sim_requests=0, verbose=False
        )
        total_low = r_low.best_analytical.n_s + r_low.best_analytical.n_l
        total_high = r_high.best_analytical.n_s + r_high.best_analytical.n_l
        assert total_high > total_low

    def test_h100_needs_fewer_gpus_than_a100(self, azure_cdf):
        from fleet_sim import A100_80GB, H100_80GB, FleetOptimizer

        opt_a100 = FleetOptimizer(
            gpu_short=A100_80GB, gpu_long=A100_80GB, B_short=4096, t_slo_ms=500
        )
        opt_h100 = FleetOptimizer(
            gpu_short=H100_80GB, gpu_long=H100_80GB, B_short=4096, t_slo_ms=500
        )
        r_a100 = opt_a100.optimize(
            cdf=azure_cdf, lam=50, gammas=[1.0], n_sim_requests=0, verbose=False
        )
        r_h100 = opt_h100.optimize(
            cdf=azure_cdf, lam=50, gammas=[1.0], n_sim_requests=0, verbose=False
        )
        total_a100 = r_a100.best_analytical.n_s + r_a100.best_analytical.n_l
        total_h100 = r_h100.best_analytical.n_s + r_h100.best_analytical.n_l
        assert total_h100 <= total_a100

    def test_computed_profile_gives_reasonable_sizing(self, azure_cdf, h100_llama70b):
        from fleet_sim import FleetOptimizer

        opt = FleetOptimizer(
            gpu_short=h100_llama70b, gpu_long=h100_llama70b, B_short=4096, t_slo_ms=500
        )
        report = opt.optimize(
            cdf=azure_cdf, lam=50, gammas=[1.0], n_sim_requests=0, verbose=False
        )
        best = report.best_analytical
        assert best is not None
        # Reasonable fleet size: not 0 and not absurdly large
        total = best.n_s + best.n_l
        assert 1 <= total <= 500

    def test_optimize_with_des_verification(self, azure_cdf):
        from fleet_sim import A100_80GB, FleetOptimizer

        opt = FleetOptimizer(
            gpu_short=A100_80GB, gpu_long=A100_80GB, B_short=4096, t_slo_ms=500
        )
        report = opt.optimize(
            cdf=azure_cdf,
            lam=30,
            gammas=[1.0],
            n_sim_requests=5000,
            verify_top_n=1,
            verbose=False,
        )
        # DES may or may not produce results at low n_sim, but should not crash
        assert report is not None


# ── E2E: Disaggregated optimizer ──────────────────────────────────────────────


class TestDisaggOptimizerE2E:
    @pytest.fixture
    def disagg_h100_llama(self, builder):
        from fleet_sim import H100_SXM, LLAMA_3_1_70B, DisaggFleetOptimizer
        from fleet_sim.gpu_profiles import ServingConfig

        prefill = builder.build(
            H100_SXM,
            LLAMA_3_1_70B,
            ServingConfig(tp=4, dtype_bytes=2, mean_ctx_tokens=1024, phase="prefill"),
        )
        decode = builder.build(
            H100_SXM,
            LLAMA_3_1_70B,
            ServingConfig(tp=8, dtype_bytes=2, mean_ctx_tokens=1024, phase="decode"),
        )
        return DisaggFleetOptimizer(
            prefill,
            decode,
            mean_isl=1024,
            mean_osl=256,
            slo_ttft_ms=2000,
            slo_tpot_ms=100,
            max_ctx=4096,
        )

    def test_disagg_finds_valid_config(self, disagg_h100_llama):
        r = disagg_h100_llama.optimize(max_prefill=6, max_decode=6)
        assert r is not None

    def test_disagg_gpu_count_consistent(self, disagg_h100_llama):
        r = disagg_h100_llama.optimize(max_prefill=6, max_decode=6)
        assert r.total_gpus == r.n_prefill * r.prefill_gpus + r.n_decode * r.decode_gpus

    def test_disagg_rate_matching_uses_alpha(self, disagg_h100_llama):
        from fleet_sim import ALPHA_DEC, ALPHA_PRE

        r = disagg_h100_llama.optimize(max_prefill=6, max_decode=6)
        # The system rate should equal min(R_pre, R_dec) with degradation
        p = disagg_h100_llama.prefill_profile
        d = disagg_h100_llama.decode_profile
        max_ctx = disagg_h100_llama.max_ctx
        r_pre = (
            p.throughput(max_ctx, disagg_h100_llama.mean_isl, 1.0)
            * r.n_prefill
            * ALPHA_PRE
        )
        r_dec = (
            d.throughput(max_ctx, 1.0, disagg_h100_llama.mean_osl)
            * r.n_decode
            * ALPHA_DEC
        )
        expected = min(r_pre, r_dec)
        assert r.system_rate == pytest.approx(expected, rel=1e-6)

    def test_disagg_sweep_covers_all_combos(self, disagg_h100_llama):
        pts = disagg_h100_llama.sweep(max_prefill=3, max_decode=3)
        # 3×3 = 9 combinations
        assert len(pts) == 9

    def test_disagg_beta_applied_to_ttft(self, disagg_h100_llama):
        from fleet_sim import BETA_TTFT

        r = disagg_h100_llama.optimize(max_prefill=4, max_decode=4)
        # Effective TTFT = base_TTFT × β_TTFT
        base = disagg_h100_llama._prefill_ttft_ms()
        assert r.ttft_ms == pytest.approx(base * BETA_TTFT, rel=1e-6)


# ── E2E: DES simulation ───────────────────────────────────────────────────────


class TestDESSimulationE2E:
    def test_fleet_simulation_runs(self, azure_cdf):
        from fleet_sim import A100_80GB, Fleet, FleetConfig, PoolConfig
        from fleet_sim.workload.synthetic import CdfWorkload, PoissonWorkload

        workload = CdfWorkload(azure_cdf)
        arrivals = PoissonWorkload(
            lam=20, length_gen=workload, n_requests=500
        ).generate()

        cfg = FleetConfig(
            pools=[
                PoolConfig(pool_id="main", gpu=A100_80GB, n_gpus=8, max_ctx=4096),
            ]
        )
        result = Fleet(cfg).run(arrivals)

        assert len(result.completed) == 500
        assert result.p99_ttft_ms() > 0
        assert result.p50_ttft_ms() > 0

    def test_more_gpus_reduces_p99(self, azure_cdf):
        from fleet_sim import A100_80GB, Fleet, FleetConfig, PoolConfig
        from fleet_sim.workload.synthetic import CdfWorkload, PoissonWorkload

        workload = CdfWorkload(azure_cdf)
        arrivals = PoissonWorkload(
            lam=30, length_gen=workload, n_requests=1000
        ).generate()

        def run_p99(n):
            cfg = FleetConfig(
                pools=[
                    PoolConfig(pool_id="main", gpu=A100_80GB, n_gpus=n, max_ctx=4096),
                ]
            )
            return Fleet(cfg).run(arrivals).p99_ttft_ms()

        p99_4 = run_p99(4)
        p99_16 = run_p99(16)
        assert p99_16 <= p99_4

    def test_computed_profile_in_des_simulation(self, azure_cdf, h100_llama70b):
        from fleet_sim import Fleet, FleetConfig, PoolConfig
        from fleet_sim.workload.synthetic import CdfWorkload, PoissonWorkload

        workload = CdfWorkload(azure_cdf)
        arrivals = PoissonWorkload(
            lam=20, length_gen=workload, n_requests=500
        ).generate()

        cfg = FleetConfig(
            pools=[
                PoolConfig(pool_id="main", gpu=h100_llama70b, n_gpus=4, max_ctx=4096),
            ]
        )
        result = Fleet(cfg).run(arrivals)
        assert len(result.completed) == 500
        assert result.p99_ttft_ms() > 0

    def test_two_pool_fleet_simulation(self, azure_cdf):
        from fleet_sim import A100_80GB, H100_80GB, Fleet, FleetConfig, PoolConfig
        from fleet_sim.workload.synthetic import CdfWorkload, PoissonWorkload

        workload = CdfWorkload(azure_cdf)
        arrivals = PoissonWorkload(
            lam=30, length_gen=workload, n_requests=800
        ).generate()

        cfg = FleetConfig(
            pools=[
                PoolConfig(pool_id="short", gpu=A100_80GB, n_gpus=6, max_ctx=2048),
                PoolConfig(pool_id="long", gpu=H100_80GB, n_gpus=4, max_ctx=32768),
            ],
            router_type="LengthRouter",
            router_kwargs={"short_threshold": 2048},
        )
        result = Fleet(cfg).run(arrivals)
        assert len(result.completed) == 800


# ── E2E: All GPU types produce valid profiles ─────────────────────────────────


class TestAllGPUTypesE2E:
    @pytest.mark.parametrize(
        "hw_name",
        [
            "a100",
            "h100",
            "h200",
            "b200",
            "gb200",
            "l40s",
        ],
    )
    def test_profile_build_all_gpu_types(self, hw_name, builder):
        from fleet_sim import LLAMA_3_1_70B, get_hardware
        from fleet_sim.gpu_profiles import ServingConfig

        hw = get_hardware(hw_name)
        cfg = ServingConfig(tp=8, dtype_bytes=2, mean_ctx_tokens=2048)
        prof = builder.build(hw, LLAMA_3_1_70B, cfg)
        assert prof.W > 0
        assert prof.H > 0
        assert prof.total_kv_blks > 0
        assert prof.cost_per_hr > 0

    @pytest.mark.parametrize(
        "model_name,expected_moe",
        [
            ("llama-3.1-8b", False),
            ("llama-3.1-70b", False),
            ("qwen3-32b", False),
            ("qwen3-235b", True),
            ("deepseek-v3", True),
        ],
    )
    def test_profile_build_all_models(self, model_name, expected_moe, builder):
        from fleet_sim import H100_SXM, get_model
        from fleet_sim.gpu_profiles import ServingConfig

        model = get_model(model_name)
        cfg = ServingConfig(tp=8, dtype_bytes=1, mean_ctx_tokens=2048)
        prof = builder.build(H100_SXM, model, cfg)
        assert prof.W > 0
        assert prof.H > 0
        assert model.is_moe == expected_moe

    def test_moe_W_much_larger_than_dense_at_same_scale(self, builder):
        from fleet_sim import DEEPSEEK_V3, H100_SXM, LLAMA_3_1_70B
        from fleet_sim.gpu_profiles import ServingConfig

        cfg = ServingConfig(tp=8, dtype_bytes=1, mean_ctx_tokens=2048)
        dense = builder.build(H100_SXM, LLAMA_3_1_70B, cfg)
        moe = builder.build(H100_SXM, DEEPSEEK_V3, cfg)
        # DeepSeek-V3 has much more MoE dispatch overhead
        assert moe.W > dense.W * 3
