"""Runner and internal simulator contract tests for fleet-sim API flows."""

from __future__ import annotations

import pytest

from .api_test_support import _MINIMAL_CSV, _MINIMAL_JSONL, _load_cdf


class TestRunnerGpuResolver:
    def test_a100_resolves_to_manual_profile(self):
        from fleet_sim.api.runner import _gpu
        from fleet_sim.gpu_profiles.profiles import A100_80GB

        assert _gpu("a100") is A100_80GB

    def test_h100_resolves_to_manual_profile(self):
        from fleet_sim.api.runner import _gpu
        from fleet_sim.gpu_profiles.profiles import H100_80GB

        assert _gpu("h100") is H100_80GB

    def test_a10g_resolves_to_manual_profile(self):
        from fleet_sim.api.runner import _gpu
        from fleet_sim.gpu_profiles.profiles import A10G

        assert _gpu("a10g") is A10G

    def test_uppercase_and_hyphens_normalised(self):
        from fleet_sim.api.runner import _gpu
        from fleet_sim.gpu_profiles.profiles import A100_80GB

        assert _gpu("A100") is A100_80GB
        assert _gpu("a100-80gb") is A100_80GB

    def test_h200_resolves_via_catalog(self):
        from fleet_sim.api.runner import _gpu

        profile = _gpu("h200")
        assert profile.W > 0
        assert profile.cost_per_hr > 0

    def test_l40s_resolves_via_catalog(self):
        from fleet_sim.api.runner import _gpu

        assert _gpu("l40s").W > 0

    def test_unknown_gpu_raises_value_error(self):
        from fleet_sim.api.runner import _gpu

        with pytest.raises(ValueError, match="Unknown GPU profile"):
            _gpu("imaginary_gpu_xyz")


class TestRunnerCdfLoader:
    def test_load_builtin_azure(self):
        from fleet_sim.api.models import WorkloadRef
        from fleet_sim.api.runner import _load_cdf

        cdf = _load_cdf(WorkloadRef(type="builtin", name="azure"))
        assert len(cdf) > 0
        assert cdf[-1][1] == pytest.approx(1.0, abs=0.01)

    def test_load_builtin_lmsys(self):
        from fleet_sim.api.models import WorkloadRef
        from fleet_sim.api.runner import _load_cdf

        assert len(_load_cdf(WorkloadRef(type="builtin", name="lmsys"))) > 0

    def test_load_builtin_not_found_raises(self):
        from fleet_sim.api.models import WorkloadRef
        from fleet_sim.api.runner import _load_cdf

        with pytest.raises(FileNotFoundError):
            _load_cdf(WorkloadRef(type="builtin", name="does_not_exist_xyz"))

    def test_load_trace_without_id_raises(self):
        from fleet_sim.api.models import WorkloadRef
        from fleet_sim.api.runner import _load_cdf

        with pytest.raises(ValueError, match="trace_id required"):
            _load_cdf(WorkloadRef(type="trace", trace_id=None))

    def test_cdf_from_jsonl_trace(self, tmp_path):
        from fleet_sim.api.runner import _cdf_from_trace

        path = tmp_path / "trace.jsonl"
        path.write_text(_MINIMAL_JSONL)
        cdf = _cdf_from_trace(path, "jsonl")
        assert len(cdf) > 0
        assert cdf[-1][1] == pytest.approx(1.0)
        thresholds = [row[0] for row in cdf]
        assert thresholds == sorted(thresholds)

    def test_cdf_from_csv_trace(self, tmp_path):
        from fleet_sim.api.runner import _cdf_from_trace

        path = tmp_path / "trace.csv"
        path.write_text(_MINIMAL_CSV)
        cdf = _cdf_from_trace(path, "csv")
        assert len(cdf) > 0
        assert cdf[-1][1] == pytest.approx(1.0)

    def test_cdf_from_empty_file_raises(self, tmp_path):
        from fleet_sim.api.runner import _cdf_from_trace

        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            _cdf_from_trace(path, "jsonl")


class TestRunnerOptimize:
    @pytest.fixture(scope="class")
    def opt_result(self):
        from fleet_sim.api.models import OptimizeParams, WorkloadRef
        from fleet_sim.api.runner import _run_optimize

        params = OptimizeParams(
            workload=WorkloadRef(type="builtin", name="azure"),
            lam=30.0,
            slo_ms=500.0,
            b_short=4096,
            gpu_short="a100",
            gpu_long="a100",
            long_max_ctx=65536,
            gamma_min=1.0,
            gamma_max=1.0,
            gamma_step=0.5,
            n_sim_requests=1000,
        )
        return _run_optimize(params)

    def test_returns_opt_result_type(self, opt_result):
        from fleet_sim.api.models import OptResult

        assert isinstance(opt_result, OptResult)

    def test_best_point_present(self, opt_result):
        assert opt_result.best is not None

    def test_best_has_required_fields(self, opt_result):
        best = opt_result.best
        assert best.gamma >= 1.0
        assert best.n_s >= 0
        assert best.n_l >= 0
        assert best.total_gpus > 0
        assert best.annual_cost_kusd > 0
        assert isinstance(best.slo_met, bool)
        assert best.source in ("analytical", "des", "simulated")

    def test_sweep_nonempty(self, opt_result):
        assert len(opt_result.sweep) > 0

    def test_sweep_points_have_required_fields(self, opt_result):
        for point in opt_result.sweep:
            assert point.total_gpus > 0
            assert point.annual_cost_kusd > 0
            assert point.source in ("analytical", "des", "simulated")

    def test_baseline_cost_positive(self, opt_result):
        assert opt_result.baseline_annual_cost_kusd > 0

    def test_savings_nonnegative(self, opt_result):
        assert opt_result.savings_pct >= 0.0


class TestRunnerSimulate:
    @pytest.fixture(scope="class")
    def sim_result(self):
        from fleet_sim.api.models import (
            FleetConfigIn,
            PoolConfigIn,
            SimulateParams,
            WorkloadRef,
        )
        from fleet_sim.api.runner import _run_simulate

        params = SimulateParams(
            workload=WorkloadRef(type="builtin", name="azure"),
            fleet=FleetConfigIn(
                name="test",
                pools=[
                    PoolConfigIn(pool_id="main", gpu="a100", n_gpus=8, max_ctx=4096)
                ],
                router="length",
            ),
            lam=20.0,
            slo_ms=500.0,
            n_requests=500,
        )
        return _run_simulate(params)

    def test_returns_sim_result_type(self, sim_result):
        from fleet_sim.api.models import SimResult

        assert isinstance(sim_result, SimResult)

    def test_total_gpus_matches_fleet(self, sim_result):
        assert sim_result.total_gpus == 8

    def test_fleet_p99_positive(self, sim_result):
        assert sim_result.fleet_p99_ttft_ms > 0

    def test_fleet_p50_positive(self, sim_result):
        assert sim_result.fleet_p50_ttft_ms > 0

    def test_slo_compliance_in_range(self, sim_result):
        assert 0.0 <= sim_result.fleet_slo_compliance <= 1.0

    def test_utilisation_in_range(self, sim_result):
        assert 0.0 <= sim_result.fleet_mean_utilisation <= 1.0

    def test_pool_results_present(self, sim_result):
        assert len(sim_result.pools) == 1
        pool = sim_result.pools[0]
        assert pool.pool_id == "main"
        assert pool.n_gpus == 8
        assert pool.p99_ttft_ms > 0
        assert pool.cost_per_hr > 0

    def test_histogram_nonempty(self, sim_result):
        assert len(sim_result.ttft_histogram) > 0

    def test_histogram_counts_approximately_n_requests(self, sim_result):
        assert sum(bin_.count for bin_ in sim_result.ttft_histogram) >= 495

    def test_two_pool_fleet_produces_two_pool_results(self):
        from fleet_sim.api.models import (
            FleetConfigIn,
            PoolConfigIn,
            SimulateParams,
            WorkloadRef,
        )
        from fleet_sim.api.runner import _run_simulate

        params = SimulateParams(
            workload=WorkloadRef(type="builtin", name="azure"),
            fleet=FleetConfigIn(
                name="hetero",
                pools=[
                    PoolConfigIn(pool_id="short", gpu="a100", n_gpus=6, max_ctx=4096),
                    PoolConfigIn(pool_id="long", gpu="h100", n_gpus=4, max_ctx=32768),
                ],
                router="length",
            ),
            lam=20.0,
            slo_ms=500.0,
            n_requests=500,
        )
        result = _run_simulate(params)
        assert len(result.pools) == 2
        assert {pool.pool_id for pool in result.pools} == {"short", "long"}


class TestRunnerWhatif:
    @pytest.fixture(scope="class")
    def whatif_result(self):
        from fleet_sim.api.models import (
            FleetConfigIn,
            PoolConfigIn,
            WhatifParams,
            WorkloadRef,
        )
        from fleet_sim.api.runner import _run_whatif

        params = WhatifParams(
            workload=WorkloadRef(type="builtin", name="azure"),
            fleet=FleetConfigIn(
                name="test",
                pools=[
                    PoolConfigIn(pool_id="main", gpu="a100", n_gpus=4, max_ctx=4096)
                ],
                router="length",
            ),
            lam_range=[5.0, 15.0, 30.0],
            slo_ms=500.0,
            n_requests=300,
        )
        return _run_whatif(params)

    def test_returns_whatif_result_type(self, whatif_result):
        from fleet_sim.api.models import WhatifResult

        assert isinstance(whatif_result, WhatifResult)

    def test_points_match_lam_range(self, whatif_result):
        assert len(whatif_result.points) == 3

    def test_points_ordered_by_lam(self, whatif_result):
        assert [point.lam for point in whatif_result.points] == sorted(
            point.lam for point in whatif_result.points
        )

    def test_points_have_positive_p99(self, whatif_result):
        for point in whatif_result.points:
            assert point.fleet_p99_ttft_ms > 0

    def test_slo_compliance_in_range(self, whatif_result):
        for point in whatif_result.points:
            assert 0.0 <= point.fleet_slo_compliance <= 1.0

    def test_cost_positive(self, whatif_result):
        for point in whatif_result.points:
            assert point.annual_cost_kusd > 0

    def test_slo_break_detected_under_strict_slo(self):
        from fleet_sim.api.models import (
            FleetConfigIn,
            PoolConfigIn,
            WhatifParams,
            WorkloadRef,
        )
        from fleet_sim.api.runner import _run_whatif

        params = WhatifParams(
            workload=WorkloadRef(type="builtin", name="azure"),
            fleet=FleetConfigIn(
                name="tiny",
                pools=[
                    PoolConfigIn(pool_id="main", gpu="a100", n_gpus=4, max_ctx=4096)
                ],
                router="length",
            ),
            lam_range=[5.0, 20.0],
            slo_ms=1.0,
            n_requests=200,
        )
        result = _run_whatif(params)
        assert result.slo_break_lam == pytest.approx(5.0)


class TestSimulatorApiContract:
    @pytest.fixture(scope="class")
    def azure_cdf(self):
        return _load_cdf("azure")

    @pytest.fixture(scope="class")
    def opt_report(self, azure_cdf):
        from fleet_sim import A100_80GB, FleetOptimizer

        optimizer = FleetOptimizer(
            gpu_short=A100_80GB, gpu_long=A100_80GB, B_short=4096, t_slo_ms=500
        )
        return optimizer.optimize(
            cdf=azure_cdf, lam=30, gammas=[1.0], n_sim_requests=0, verbose=False
        )

    def test_report_has_analytical_list(self, opt_report):
        assert hasattr(opt_report, "analytical")
        assert isinstance(opt_report.analytical, list)
        assert len(opt_report.analytical) > 0

    def test_report_has_simulated_list(self, opt_report):
        assert hasattr(opt_report, "simulated")
        assert isinstance(opt_report.simulated, list)

    def test_report_has_best_analytical(self, opt_report):
        assert hasattr(opt_report, "best_analytical")
        assert opt_report.best_analytical is not None

    def test_report_has_best_simulated_attr(self, opt_report):
        assert hasattr(opt_report, "best_simulated")

    def test_sweep_result_gamma(self, opt_report):
        result = opt_report.analytical[0]
        assert hasattr(result, "gamma")
        assert result.gamma >= 1.0

    def test_sweep_result_n_s_n_l(self, opt_report):
        result = opt_report.analytical[0]
        assert hasattr(result, "n_s")
        assert hasattr(result, "n_l")
        assert result.n_s >= 0
        assert result.n_l >= 0

    def test_sweep_result_total_gpus(self, opt_report):
        result = opt_report.analytical[0]
        assert hasattr(result, "total_gpus")
        assert result.total_gpus > 0

    def test_sweep_result_annualised_cost_kusd(self, opt_report):
        result = opt_report.analytical[0]
        assert hasattr(result, "annualised_cost_kusd")
        assert result.annualised_cost_kusd > 0

    def test_sweep_result_p99_ttft_fields(self, opt_report):
        result = opt_report.analytical[0]
        assert hasattr(result, "p99_ttft_short_ms")
        assert hasattr(result, "p99_ttft_long_ms")

    def test_sweep_result_slo_met(self, opt_report):
        result = opt_report.analytical[0]
        assert hasattr(result, "slo_met")
        assert isinstance(result.slo_met, bool)

    def test_sweep_result_source(self, opt_report):
        result = opt_report.analytical[0]
        assert hasattr(result, "source")
        assert result.source in ("analytical", "des", "simulated")

    def test_fleet_sim_result_p99_ttft_ms(self, azure_cdf):
        from fleet_sim import A100_80GB, Fleet, FleetConfig, PoolConfig
        from fleet_sim.workload.synthetic import CdfWorkload, PoissonWorkload

        arrivals = PoissonWorkload(
            lam=20, length_gen=CdfWorkload(azure_cdf), n_requests=300
        ).generate()
        result = Fleet(
            FleetConfig(
                pools=[
                    PoolConfig(pool_id="main", gpu=A100_80GB, n_gpus=8, max_ctx=4096)
                ]
            )
        ).run(arrivals)
        assert hasattr(result, "p99_ttft_ms")
        assert hasattr(result, "p50_ttft_ms")
        assert hasattr(result, "p99_queue_wait_ms")
        assert hasattr(result, "slo_compliance")
        assert hasattr(result, "mean_utilisation")
        assert hasattr(result, "annualised_cost_usd")
        assert hasattr(result, "total_gpus")
        assert hasattr(result, "completed")
        assert hasattr(result, "pools")
        assert result.p99_ttft_ms() > 0
        assert result.total_gpus() == 8

    def test_pool_object_has_cost_per_hr(self, azure_cdf):
        from fleet_sim import A100_80GB, Fleet, FleetConfig, PoolConfig
        from fleet_sim.workload.synthetic import CdfWorkload, PoissonWorkload

        arrivals = PoissonWorkload(
            lam=20, length_gen=CdfWorkload(azure_cdf), n_requests=200
        ).generate()
        result = Fleet(
            FleetConfig(
                pools=[
                    PoolConfig(pool_id="main", gpu=A100_80GB, n_gpus=4, max_ctx=4096)
                ]
            )
        ).run(arrivals)
        for pool in result.pools.values():
            assert hasattr(pool, "cost_per_hr")
            assert callable(pool.cost_per_hr)
            assert pool.cost_per_hr() > 0

    def test_pool_object_has_gpu_with_name(self, azure_cdf):
        from fleet_sim import A100_80GB, Fleet, FleetConfig, PoolConfig
        from fleet_sim.workload.synthetic import CdfWorkload, PoissonWorkload

        arrivals = PoissonWorkload(
            lam=20, length_gen=CdfWorkload(azure_cdf), n_requests=200
        ).generate()
        result = Fleet(
            FleetConfig(
                pools=[
                    PoolConfig(pool_id="main", gpu=A100_80GB, n_gpus=4, max_ctx=4096)
                ]
            )
        ).run(arrivals)
        for pool in result.pools.values():
            assert hasattr(pool, "gpu")
            assert hasattr(pool.gpu, "name")

    def test_calibrate_returns_four_tuple(self, azure_cdf):
        from fleet_sim import A100_80GB
        from fleet_sim.optimizer.base import _calibrate

        mu_gpu, cv2, n_slots, mean_prefill = _calibrate(
            azure_cdf, pool_max=4096, gpu=A100_80GB
        )
        assert mu_gpu > 0
        assert cv2 >= 0
        assert n_slots > 0
        assert mean_prefill >= 0

    def test_disagg_result_total_gpus_consistent(self):
        from fleet_sim import H100_SXM, LLAMA_3_1_70B, DisaggFleetOptimizer
        from fleet_sim.gpu_profiles import ProfileBuilder, ServingConfig

        builder = ProfileBuilder()
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
        result = DisaggFleetOptimizer(
            prefill,
            decode,
            mean_isl=1024,
            mean_osl=256,
            slo_ttft_ms=2000,
            slo_tpot_ms=100,
            max_ctx=4096,
        ).optimize(max_prefill=4, max_decode=4)
        assert (
            result.total_gpus
            == result.n_prefill * result.prefill_gpus
            + result.n_decode * result.decode_gpus
        )
