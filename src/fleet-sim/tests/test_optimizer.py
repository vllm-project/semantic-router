"""Unit and integration tests for FleetOptimizer, threshold_pareto, and GPU comparison."""

import json
from pathlib import Path

import pytest

_DATA = Path(__file__).parent.parent / "data"


def _load_cdf(name: str) -> list:
    raw = json.loads((_DATA / f"{name}_cdf.json").read_text())
    return raw["cdf"] if isinstance(raw, dict) else raw


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def lmsys_cdf():
    return _load_cdf("lmsys")


@pytest.fixture(scope="module")
def azure_cdf():
    return _load_cdf("azure")


@pytest.fixture(scope="module")
def agent_cdf():
    return _load_cdf("agent_heavy")


# ── _calibrate uses seq-len-aware service_time ────────────────────────────────


class TestCalibrateSeqLenAware:
    def test_short_pool_higher_mu_than_long_pool(self, lmsys_cdf):
        """Short pool should have higher service rate (μ) than long pool
        for the same workload slice, because requests are shorter."""
        from fleet_sim.gpu_profiles.profiles import A100_80GB
        from fleet_sim.optimizer.base import _calibrate

        mu_short, _, _, _ = _calibrate(lmsys_cdf, pool_max=2048, gpu=A100_80GB)
        mu_long, _, _, _ = _calibrate(lmsys_cdf, pool_max=65536, gpu=A100_80GB)
        assert mu_short > mu_long

    def test_cv2_positive(self, azure_cdf):
        from fleet_sim.gpu_profiles.profiles import A100_80GB
        from fleet_sim.optimizer.base import _calibrate

        _, cv2, _, _ = _calibrate(azure_cdf, pool_max=8192, gpu=A100_80GB)
        assert cv2 > 0


# ── threshold_pareto ──────────────────────────────────────────────────────────


class TestThresholdPareto:
    def test_returns_non_empty_list(self, lmsys_cdf):
        from fleet_sim.gpu_profiles.profiles import A100_80GB
        from fleet_sim.optimizer import threshold_pareto

        results = threshold_pareto(
            lmsys_cdf,
            lam=20,
            gpu_short=A100_80GB,
            gpu_long=A100_80GB,
            t_slo_ms=500,
            long_max_ctx=65536,
        )
        assert len(results) > 0

    def test_at_least_one_pareto_optimal(self, lmsys_cdf):
        from fleet_sim.gpu_profiles.profiles import A100_80GB
        from fleet_sim.optimizer import threshold_pareto

        results = threshold_pareto(
            lmsys_cdf,
            lam=20,
            gpu_short=A100_80GB,
            gpu_long=A100_80GB,
            t_slo_ms=500,
            long_max_ctx=65536,
        )
        assert any(r.pareto for r in results)

    def test_pareto_sorted_by_b_short(self, lmsys_cdf):
        from fleet_sim.gpu_profiles.profiles import A100_80GB
        from fleet_sim.optimizer import threshold_pareto

        results = threshold_pareto(
            lmsys_cdf,
            lam=20,
            gpu_short=A100_80GB,
            gpu_long=A100_80GB,
            t_slo_ms=500,
            long_max_ctx=65536,
        )
        thresholds = [r.b_short for r in results]
        assert thresholds == sorted(thresholds)

    def test_pareto_point_not_dominated(self, lmsys_cdf):
        """Every Pareto-optimal point should have no other point that is both
        cheaper and has lower worst-case P99."""
        from fleet_sim.gpu_profiles.profiles import A100_80GB
        from fleet_sim.optimizer import threshold_pareto

        results = threshold_pareto(
            lmsys_cdf,
            lam=20,
            gpu_short=A100_80GB,
            gpu_long=A100_80GB,
            t_slo_ms=500,
            long_max_ctx=65536,
        )
        pareto_pts = [r for r in results if r.pareto]
        for p in pareto_pts:
            dominated = any(
                other.cost_kusd_yr < p.cost_kusd_yr
                and other.worst_p99_ms < p.worst_p99_ms
                for other in results
                if other is not p
            )
            assert not dominated, f"Pareto point B={p.b_short} is dominated"

    def test_savings_nonnegative_for_lmsys(self, lmsys_cdf):
        """For LMSYS (short-dominated), the best Pareto point should not cost more than homo.
        At higher traffic levels the two-pool layout saves GPUs; at low traffic both tie.
        """
        from fleet_sim.gpu_profiles.profiles import A100_80GB
        from fleet_sim.optimizer import threshold_pareto

        results = threshold_pareto(
            lmsys_cdf,
            lam=100,
            gpu_short=A100_80GB,
            gpu_long=A100_80GB,
            t_slo_ms=500,
            long_max_ctx=65536,
        )
        best = min(
            (r for r in results if r.pareto and r.slo_met),
            key=lambda r: r.cost_kusd_yr,
            default=None,
        )
        assert best is not None
        assert (
            best.savings_vs_homo_pct >= 0
        ), f"Expected non-negative savings for LMSYS at lam=100, got {best.savings_vs_homo_pct:.1f}%"

    def test_threshold_result_fields(self, azure_cdf):
        from fleet_sim.gpu_profiles.profiles import A100_80GB
        from fleet_sim.optimizer import ThresholdResult, threshold_pareto

        results = threshold_pareto(
            azure_cdf,
            lam=50,
            gpu_short=A100_80GB,
            gpu_long=A100_80GB,
            t_slo_ms=500,
            long_max_ctx=8192,
        )
        assert len(results) > 0
        r = results[0]
        assert isinstance(r, ThresholdResult)
        assert r.b_short > 0
        assert 0 < r.alpha <= 1.0
        assert r.n_s >= 0
        assert r.n_l >= 0
        assert r.total_gpus == r.n_s + r.n_l
        assert r.cost_kusd_yr > 0
        assert r.worst_p99_ms == max(r.p99_short_ms, r.p99_long_ms)


# ── Two-pool vs homo cost comparison ─────────────────────────────────────────


class TestTwoPoolVsHomo:
    def test_two_pool_cheaper_for_lmsys(self, lmsys_cdf):
        """For LMSYS (98% short), two-pool at B_short=4096 should beat homo."""
        from fleet_sim.gpu_profiles.profiles import A100_80GB
        from fleet_sim.optimizer import FleetOptimizer

        homo = FleetOptimizer(
            gpu_short=A100_80GB,
            gpu_long=A100_80GB,
            B_short=65536,
            t_slo_ms=500,
            long_max_ctx=65536,
        )
        hetero = FleetOptimizer(
            gpu_short=A100_80GB,
            gpu_long=A100_80GB,
            B_short=4096,
            t_slo_ms=500,
            long_max_ctx=65536,
        )
        h = homo.sweep_analytical(lmsys_cdf, 20, gammas=[1.0], verbose=False)[0]
        s = hetero.sweep_analytical(lmsys_cdf, 20, gammas=[1.0], verbose=False)[0]
        assert s.total_gpus <= h.total_gpus, (
            f"Two-pool ({s.total_gpus} GPUs) should need ≤ homo ({h.total_gpus} GPUs) "
            f"for LMSYS workload"
        )

    def test_two_pool_cheaper_for_agent_heavy(self, agent_cdf):
        """For agent workload (65K ctx), B_short=16384 pool should beat homo (65K ctx)."""
        from fleet_sim.gpu_profiles.profiles import A100_80GB
        from fleet_sim.optimizer import FleetOptimizer

        homo = FleetOptimizer(
            gpu_short=A100_80GB,
            gpu_long=A100_80GB,
            B_short=65536,
            t_slo_ms=500,
            long_max_ctx=65536,
        )
        hetero = FleetOptimizer(
            gpu_short=A100_80GB,
            gpu_long=A100_80GB,
            B_short=16384,
            t_slo_ms=500,
            long_max_ctx=65536,
        )
        h = homo.sweep_analytical(agent_cdf, 200, gammas=[1.0], verbose=False)[0]
        s = hetero.sweep_analytical(agent_cdf, 200, gammas=[1.0], verbose=False)[0]
        assert s.total_gpus < h.total_gpus, (
            f"Two-pool B=16384 ({s.total_gpus} GPUs) should need < homo ({h.total_gpus} GPUs) "
            f"for agent-heavy workload"
        )

    def test_h100_fleet_smaller_than_a100(self, azure_cdf):
        """H100 has more throughput so the optimally-sized fleet needs fewer GPUs."""
        from fleet_sim.gpu_profiles.profiles import A100_80GB, H100_80GB
        from fleet_sim.optimizer import FleetOptimizer

        a100_opt = FleetOptimizer(
            gpu_short=A100_80GB,
            gpu_long=A100_80GB,
            B_short=4096,
            t_slo_ms=500,
            long_max_ctx=8192,
        )
        h100_opt = FleetOptimizer(
            gpu_short=H100_80GB,
            gpu_long=H100_80GB,
            B_short=4096,
            t_slo_ms=500,
            long_max_ctx=8192,
        )
        a100_r = a100_opt.sweep_analytical(azure_cdf, 100, gammas=[1.0], verbose=False)[
            0
        ]
        h100_r = h100_opt.sweep_analytical(azure_cdf, 100, gammas=[1.0], verbose=False)[
            0
        ]
        assert h100_r.total_gpus < a100_r.total_gpus


# ── CLI integration tests ─────────────────────────────────────────────────────


class TestCLISubcommands:
    """Smoke-tests: every subcommand must run without error and produce output."""

    @pytest.fixture(autouse=True)
    def _run_sim(self):
        """Return a helper that calls run_sim.py and asserts exit code 0."""
        import subprocess
        import sys

        self._root = Path(__file__).parent.parent

        def run(*extra_args):
            cmd = [sys.executable, str(self._root / "run_sim.py")] + list(extra_args)
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(self._root)
            )
            assert (
                result.returncode == 0
            ), f"Command failed:\n{' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            return result.stdout

        self._run = run

    def test_optimize(self):
        out = self._run(
            "optimize",
            "--cdf",
            "data/lmsys_cdf.json",
            "--lam",
            "20",
            "--slo",
            "500",
            "--b-short",
            "4096",
            "--long-max-ctx",
            "65536",
            "--gpu-short",
            "a100",
            "--gpu-long",
            "a100",
            "--n-sim-req",
            "0",
        )
        assert "n_s" in out or "total" in out.lower()

    def test_simulate(self):
        out = self._run(
            "simulate",
            "--cdf",
            "data/azure_cdf.json",
            "--lam",
            "10",
            "--n-s",
            "3",
            "--n-l",
            "2",
            "--b-short",
            "4096",
            "--long-max-ctx",
            "8192",
            "--n-req",
            "500",
            "--seed",
            "42",
        )
        assert "P99" in out or "p99" in out.lower()

    def test_whatif_lambda_sweep(self):
        out = self._run(
            "whatif",
            "--cdf",
            "data/azure_cdf.json",
            "--lam-range",
            "50",
            "100",
            "--slo",
            "500",
            "--b-short",
            "4096",
            "--long-max-ctx",
            "8192",
            "--gpu-short",
            "a100",
        )
        assert "50" in out and "100" in out

    def test_whatif_gpu_compare(self):
        out = self._run(
            "whatif",
            "--cdf",
            "data/azure_cdf.json",
            "--lam-range",
            "50",
            "--slo",
            "500",
            "--b-short",
            "4096",
            "--long-max-ctx",
            "8192",
            "--gpu-compare",
            "a100",
            "h100",
        )
        assert "A100" in out.upper() and "H100" in out.upper()
        assert "cost ratio" in out.lower() or "1.00" in out

    def test_whatif_gpu_compare_three_types(self):
        out = self._run(
            "whatif",
            "--cdf",
            "data/azure_cdf.json",
            "--lam-range",
            "50",
            "--slo",
            "500",
            "--b-short",
            "4096",
            "--long-max-ctx",
            "8192",
            "--gpu-compare",
            "a100",
            "h100",
            "a10g",
        )
        assert "A10G" in out.upper()

    def test_pareto(self):
        out = self._run(
            "pareto",
            "--cdf",
            "data/lmsys_cdf.json",
            "--lam",
            "20",
            "--slo",
            "500",
            "--long-max-ctx",
            "65536",
            "--gpu-short",
            "a100",
        )
        assert "B_short" in out
        assert "★" in out  # at least one Pareto-optimal point marked

    def test_pareto_azure(self):
        out = self._run(
            "pareto",
            "--cdf",
            "data/azure_cdf.json",
            "--lam",
            "50",
            "--slo",
            "500",
            "--long-max-ctx",
            "8192",
            "--gpu-short",
            "a100",
        )
        assert "saving" in out.lower() or "Saving" in out

    def test_compare_routers(self):
        out = self._run(
            "compare-routers",
            "--cdf",
            "data/azure_cdf.json",
            "--lam",
            "10",
            "--n-s",
            "3",
            "--n-l",
            "2",
            "--b-short",
            "4096",
            "--long-max-ctx",
            "8192",
            "--n-req",
            "300",
            "--seed",
            "42",
        )
        assert "LengthRouter" in out or "P99" in out

    def test_simulate_fleet(self):
        out = self._run(
            "simulate-fleet",
            "examples/multi_model_fleet.json",
            "--lam",
            "10",
            "--slo",
            "500",
            "--n-req",
            "300",
            "--seed",
            "42",
        )
        assert "pool" in out.lower() or "P99" in out or "completed" in out.lower()

    def test_whatif_saves_json(self, tmp_path):
        out_file = tmp_path / "result.json"
        self._run(
            "whatif",
            "--cdf",
            "data/azure_cdf.json",
            "--lam-range",
            "50",
            "--slo",
            "500",
            "--b-short",
            "4096",
            "--long-max-ctx",
            "8192",
            "--gpu-compare",
            "a100",
            "h100",
            "--out",
            str(out_file),
        )
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert len(data) == 1  # one λ
        assert "a100" in data[0]
        assert "h100" in data[0]
