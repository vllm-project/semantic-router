"""Tests for all public import paths and hand-calibrated profile behavior."""

import pytest


class TestImports:
    def test_import_gpu_profile_from_fleet_sim(self):
        from fleet_sim import GpuProfile

        assert GpuProfile is not None

    def test_import_predefined_profiles_from_fleet_sim(self):
        from fleet_sim import A10G, A100_80GB, H100_80GB

        assert A100_80GB.name == "A100-80GB"
        assert H100_80GB.name == "H100-80GB"
        assert A10G.name == "A10G"

    def test_import_profiles_from_gpu_profiles(self):
        from fleet_sim.gpu_profiles import A100_80GB

        assert A100_80GB is not None

    def test_import_profiles_from_profiles_module(self):
        from fleet_sim.gpu_profiles.profiles import (
            A100_80GB,
        )

        assert A100_80GB is not None

    def test_import_fleet_config_pool_config_request(self):
        from fleet_sim import Fleet

        assert Fleet is not None

    def test_import_fleet_optimizer(self):
        from fleet_sim import FleetOptimizer

        assert FleetOptimizer is not None

    def test_import_core_submodule(self):
        from fleet_sim.core import Fleet

        assert Fleet is not None

    def test_import_hardware_catalog(self):
        from fleet_sim import H100_SXM

        assert H100_SXM is not None

    def test_import_model_catalog(self):
        from fleet_sim import LLAMA_3_1_70B

        assert LLAMA_3_1_70B is not None

    def test_import_profile_builder(self):
        from fleet_sim.gpu_profiles import (
            ProfileBuilder,
        )

        assert ProfileBuilder is not None

    def test_import_disagg_optimizer(self):
        from fleet_sim import DisaggFleetOptimizer

        assert DisaggFleetOptimizer is not None

    def test_import_optimizer_analysis_submodules(self):
        from fleet_sim.optimizer.grid_flex import GridFlexPoint, grid_flex_analysis
        from fleet_sim.optimizer.tpw import FleetTpwResult, TpwPoint, _split_cdf

        assert GridFlexPoint is not None
        assert grid_flex_analysis is not None
        assert TpwPoint is not None
        assert FleetTpwResult is not None
        assert _split_cdf is not None

    def test_root_optimizer_surface_is_curated(self):
        import fleet_sim
        from fleet_sim import optimizer

        assert hasattr(fleet_sim, "FleetOptimizer")
        assert hasattr(fleet_sim, "grid_flex_analysis")
        assert hasattr(optimizer, "_split_cdf")
        assert not hasattr(fleet_sim, "_split_cdf")

    def test_custom_factory(self):
        from fleet_sim.gpu_profiles import CUSTOM

        p = CUSTOM("test", W=0.005, H=0.0005)
        assert p.name == "test"
        assert p.W == 0.005


class TestManualProfile:
    """ManualProfile: hand-calibrated W/H constants."""

    def test_construct_directly(self):
        from fleet_sim.gpu_profiles import ManualProfile

        p = ManualProfile(
            name="test-gpu",
            W=0.008,
            H=0.0006,
            chunk=512,
            blk_size=16,
            total_kv_blks=65536,
            max_slots=128,
            cost_per_hr=2.0,
        )
        assert p.name == "test-gpu"
        assert p.iter_latency(10) == pytest.approx(0.008 + 10 * 0.0006)

    def test_a100_iter_latency(self):
        from fleet_sim import A100_80GB

        lat = A100_80GB.iter_latency(64)
        assert lat == pytest.approx(A100_80GB.W + 64 * A100_80GB.H)

    def test_a100_n_slots(self):
        from fleet_sim import A100_80GB

        slots = A100_80GB.n_slots(4096)
        assert isinstance(slots, int)
        assert slots > 0

    def test_a100_service_time(self):
        from fleet_sim import A100_80GB

        st = A100_80GB.service_time(512, 256, 4096)
        assert st > 0

    def test_a100_throughput(self):
        from fleet_sim import A100_80GB

        tp = A100_80GB.throughput(4096, 512.0, 256.0)
        assert tp > 0

    def test_satisfies_gpu_profile_protocol(self):
        from fleet_sim import A100_80GB, GpuProfile

        assert isinstance(A100_80GB, GpuProfile)

    def test_pool_config_accepts_manual_profile(self):
        from fleet_sim import A100_80GB, PoolConfig

        pc = PoolConfig(pool_id="test", gpu=A100_80GB, n_gpus=4, max_ctx=4096)
        assert pc.gpu is A100_80GB


class TestFleetOptimizerWithProfiles:
    """FleetOptimizer works with both ManualProfile and ComputedProfile."""

    @pytest.fixture
    def azure_cdf(self):
        import json
        from pathlib import Path

        raw = json.loads(
            (Path(__file__).parent.parent / "data" / "azure_cdf.json").read_text()
        )
        return raw["cdf"] if isinstance(raw, dict) else raw

    def test_optimize_with_manual_profile(self, azure_cdf):
        from fleet_sim import A100_80GB, FleetOptimizer

        opt = FleetOptimizer(
            gpu_short=A100_80GB,
            gpu_long=A100_80GB,
            B_short=4096,
            t_slo_ms=500,
        )
        report = opt.optimize(
            cdf=azure_cdf, lam=30, gammas=[1.0], n_sim_requests=0, verbose=False
        )
        best = report.best_analytical
        assert best is not None
        assert best.n_s >= 1

    def test_optimize_with_computed_profile(self, azure_cdf):
        from fleet_sim import H100_SXM, LLAMA_3_1_70B, FleetOptimizer
        from fleet_sim.gpu_profiles import ProfileBuilder, ServingConfig

        profile = ProfileBuilder().build(
            H100_SXM,
            LLAMA_3_1_70B,
            ServingConfig(tp=8, dtype_bytes=2, mean_ctx_tokens=2048),
        )
        opt = FleetOptimizer(
            gpu_short=profile, gpu_long=profile, B_short=4096, t_slo_ms=500
        )
        report = opt.optimize(
            cdf=azure_cdf, lam=30, gammas=[1.0], n_sim_requests=0, verbose=False
        )
        assert report.best_analytical is not None

    def test_computed_and_manual_give_same_interface(self, azure_cdf):
        from fleet_sim import A100_80GB, H100_SXM, LLAMA_3_1_70B, FleetOptimizer
        from fleet_sim.gpu_profiles import ProfileBuilder, ServingConfig

        computed = ProfileBuilder().build(
            H100_SXM,
            LLAMA_3_1_70B,
            ServingConfig(tp=8, dtype_bytes=2, mean_ctx_tokens=2048),
        )
        for profile in (A100_80GB, computed):
            opt = FleetOptimizer(
                gpu_short=profile, gpu_long=profile, B_short=4096, t_slo_ms=500
            )
            r = opt.optimize(
                cdf=azure_cdf, lam=30, gammas=[1.0], n_sim_requests=0, verbose=False
            )
            assert r.best_analytical.cost_per_hr > 0
