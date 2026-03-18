"""API and runner integration tests.

These tests verify that:
1.  Every REST endpoint returns the correct HTTP status code and response shape.
2.  The runner functions (_run_optimize, _run_simulate, _run_whatif) correctly
    interface with the fleet_sim simulator — catching any breakage caused by
    internal simulator API changes.
3.  The simulator's internal objects (OptimizationReport, SweepResult,
    FleetSimResult) expose the attributes the runner depends on.

Storage is redirected to a temporary directory for each test so nothing
touches the real api_store on disk.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# ── Data helpers ──────────────────────────────────────────────────────────────

_DATA = Path(__file__).parent.parent / "data"


def _load_cdf(name: str = "azure") -> list:
    path = _DATA / f"{name}_cdf.json"
    raw = json.loads(path.read_text())
    return raw["cdf"] if isinstance(raw, dict) else raw


_MINIMAL_JSONL = (
    '{"prompt_tokens": 128, "generated_tokens": 64, "timestamp": 0.0}\n'
    '{"prompt_tokens": 256, "generated_tokens": 128, "timestamp": 1.0}\n'
    '{"prompt_tokens": 512, "generated_tokens": 64, "timestamp": 2.0}\n'
)

_MINIMAL_CSV = (
    "prompt_tokens,generated_tokens,timestamp\n"
    "128,64,0.0\n"
    "256,128,1.0\n"
    "512,64,2.0\n"
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def isolated_storage(tmp_path, monkeypatch) -> None:
    """Redirect all storage I/O to a throw-away temp directory."""
    import api.storage as stor

    base = tmp_path / "api_store"
    traces_dir = base / "traces"
    jobs_dir = base / "jobs"
    traces_dir.mkdir(parents=True)
    jobs_dir.mkdir(parents=True)
    trace_meta = base / "traces_meta.json"
    fleets_file = base / "fleets.json"
    trace_meta.write_text("{}")
    fleets_file.write_text("{}")

    monkeypatch.setattr(stor, "_TRACES_DIR", traces_dir)
    monkeypatch.setattr(stor, "_TRACE_META", trace_meta)
    monkeypatch.setattr(stor, "_FLEETS_FILE", fleets_file)
    monkeypatch.setattr(stor, "_JOBS_DIR", jobs_dir)


@pytest.fixture()
def client(isolated_storage) -> Generator[TestClient, None, None]:
    from api.app import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _wait_job_done(client: TestClient, job_id: str,
                   max_wait: float = 60.0) -> dict:
    """Poll until a job reaches terminal status or max_wait is exceeded."""
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        data = r.json()
        if data["status"] in ("done", "failed"):
            return data
        time.sleep(0.2)
    return client.get(f"/api/jobs/{job_id}").json()


# ── Workload routes ───────────────────────────────────────────────────────────


class TestWorkloadRoutes:
    def test_list_workloads_returns_four(self, client):
        r = client.get("/api/workloads")
        assert r.status_code == 200
        items = r.json()
        assert len(items) == 4
        names = {w["name"] for w in items}
        assert names == {"azure", "lmsys", "lmsys_multiturn", "agent_heavy"}

    def test_builtin_workload_has_required_fields(self, client):
        r = client.get("/api/workloads")
        w = r.json()[0]
        assert "name" in w
        assert "description" in w
        assert "path" in w

    def test_get_cdf_azure(self, client):
        r = client.get("/api/workloads/azure/cdf")
        assert r.status_code == 200
        pts = r.json()
        assert len(pts) > 0
        first = pts[0]
        assert "threshold" in first
        assert "cumulative_frac" in first
        assert 0.0 < first["cumulative_frac"] <= 1.0

    @pytest.mark.parametrize("name", ["azure", "lmsys", "lmsys_multiturn", "agent_heavy"])
    def test_get_cdf_all_builtins(self, client, name):
        r = client.get(f"/api/workloads/{name}/cdf")
        assert r.status_code == 200
        assert len(r.json()) > 0

    def test_get_cdf_unknown_returns_404(self, client):
        r = client.get("/api/workloads/nonexistent/cdf")
        assert r.status_code == 404

    def test_get_stats_returns_trace_stats_shape(self, client):
        r = client.get("/api/workloads/azure/stats")
        assert r.status_code == 200
        s = r.json()
        for field in ("n_requests", "p50_prompt_tokens", "p99_prompt_tokens",
                      "p50_output_tokens", "p99_output_tokens",
                      "prompt_histogram"):
            assert field in s, f"Missing field: {field}"
        assert s["p50_prompt_tokens"] > 0
        assert len(s["prompt_histogram"]) > 0

    def test_get_stats_unknown_returns_404(self, client):
        r = client.get("/api/workloads/nonexistent/stats")
        assert r.status_code == 404


# ── GPU-profile routes ────────────────────────────────────────────────────────


class TestGpuProfileRoutes:
    def test_list_profiles_returns_three(self, client):
        r = client.get("/api/gpu-profiles")
        assert r.status_code == 200
        profiles = r.json()
        assert len(profiles) == 3

    def test_profile_has_required_fields(self, client):
        r = client.get("/api/gpu-profiles")
        for p in r.json():
            for field in ("name", "W_ms", "H_ms_per_slot", "chunk",
                          "blk_size", "total_kv_blks", "max_slots", "cost_per_hr"):
                assert field in p
            assert p["W_ms"] > 0
            assert p["cost_per_hr"] > 0


# ── Fleet CRUD routes ─────────────────────────────────────────────────────────


class TestFleetRoutes:
    _FLEET_BODY = {
        "name": "test-fleet",
        "pools": [
            {"pool_id": "short", "gpu": "a100", "n_gpus": 4, "max_ctx": 4096},
            {"pool_id": "long",  "gpu": "h100", "n_gpus": 2, "max_ctx": 32768},
        ],
        "router": "length",
        "compress_gamma": None,
    }

    def test_list_fleets_empty_initially(self, client):
        r = client.get("/api/fleets")
        assert r.status_code == 200
        assert r.json() == []

    def test_create_fleet_returns_created(self, client):
        r = client.post("/api/fleets", json=self._FLEET_BODY)
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "test-fleet"
        assert "id" in data
        assert data["total_gpus"] == 6
        assert data["estimated_cost_per_hr"] > 0
        assert data["estimated_annual_cost_kusd"] > 0

    def test_create_fleet_appears_in_list(self, client):
        client.post("/api/fleets", json=self._FLEET_BODY)
        r = client.get("/api/fleets")
        assert len(r.json()) == 1

    def test_get_fleet_by_id(self, client):
        created = client.post("/api/fleets", json=self._FLEET_BODY).json()
        r = client.get(f"/api/fleets/{created['id']}")
        assert r.status_code == 200
        assert r.json()["id"] == created["id"]

    def test_get_fleet_not_found(self, client):
        r = client.get("/api/fleets/doesnotexist")
        assert r.status_code == 404

    def test_delete_fleet(self, client):
        fleet_id = client.post("/api/fleets", json=self._FLEET_BODY).json()["id"]
        r = client.delete(f"/api/fleets/{fleet_id}")
        assert r.status_code == 200
        assert client.get(f"/api/fleets/{fleet_id}").status_code == 404

    def test_delete_fleet_not_found(self, client):
        r = client.delete("/api/fleets/doesnotexist")
        assert r.status_code == 404

    def test_fleet_cost_calculation(self, client):
        # A10G: $1.01/hr × 8 = $8.08/hr → $70.78k/yr
        body = {**self._FLEET_BODY,
                "pools": [{"pool_id": "main", "gpu": "a10g",
                           "n_gpus": 8, "max_ctx": 4096}]}
        r = client.post("/api/fleets", json=body)
        data = r.json()
        assert abs(data["estimated_cost_per_hr"] - 8.08) < 0.02
        assert data["estimated_annual_cost_kusd"] == pytest.approx(70.78, abs=0.5)


# ── Trace routes ──────────────────────────────────────────────────────────────


class TestTraceRoutes:
    def test_list_traces_empty_initially(self, client):
        r = client.get("/api/traces")
        assert r.status_code == 200
        assert r.json() == []

    def test_upload_jsonl_trace(self, client):
        r = client.post(
            "/api/traces?fmt=jsonl",
            files={"file": ("test.jsonl", io.BytesIO(_MINIMAL_JSONL.encode()), "text/plain")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "id" in data
        assert data["n_requests"] == 3
        assert data["format"] == "jsonl"
        assert data["stats"] is not None

    def test_upload_csv_trace(self, client):
        r = client.post(
            "/api/traces?fmt=csv",
            files={"file": ("test.csv", io.BytesIO(_MINIMAL_CSV.encode()), "text/plain")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["n_requests"] == 3
        assert data["format"] == "csv"

    def test_upload_trace_stats_have_correct_fields(self, client):
        r = client.post(
            "/api/traces?fmt=jsonl",
            files={"file": ("test.jsonl", io.BytesIO(_MINIMAL_JSONL.encode()), "text/plain")},
        )
        s = r.json()["stats"]
        for field in ("n_requests", "p50_prompt_tokens", "p99_prompt_tokens",
                      "p50_output_tokens", "p99_output_tokens",
                      "prompt_histogram", "output_histogram"):
            assert field in s

    def test_uploaded_trace_appears_in_list(self, client):
        client.post(
            "/api/traces?fmt=jsonl",
            files={"file": ("test.jsonl", io.BytesIO(_MINIMAL_JSONL.encode()), "text/plain")},
        )
        r = client.get("/api/traces")
        assert len(r.json()) == 1

    def test_get_trace_by_id(self, client):
        trace_id = client.post(
            "/api/traces?fmt=jsonl",
            files={"file": ("test.jsonl", io.BytesIO(_MINIMAL_JSONL.encode()), "text/plain")},
        ).json()["id"]
        r = client.get(f"/api/traces/{trace_id}")
        assert r.status_code == 200
        assert r.json()["id"] == trace_id

    def test_get_trace_not_found(self, client):
        r = client.get("/api/traces/doesnotexist")
        assert r.status_code == 404

    def test_sample_trace(self, client):
        trace_id = client.post(
            "/api/traces?fmt=jsonl",
            files={"file": ("test.jsonl", io.BytesIO(_MINIMAL_JSONL.encode()), "text/plain")},
        ).json()["id"]
        r = client.get(f"/api/traces/{trace_id}/sample?limit=2")
        assert r.status_code == 200
        data = r.json()
        assert "records" in data
        assert "total" in data
        assert len(data["records"]) <= 2
        assert data["total"] == 3

    def test_sample_trace_not_found(self, client):
        r = client.get("/api/traces/doesnotexist/sample")
        assert r.status_code == 404

    def test_delete_trace(self, client):
        trace_id = client.post(
            "/api/traces?fmt=jsonl",
            files={"file": ("test.jsonl", io.BytesIO(_MINIMAL_JSONL.encode()), "text/plain")},
        ).json()["id"]
        r = client.delete(f"/api/traces/{trace_id}")
        assert r.status_code == 200
        assert client.get(f"/api/traces/{trace_id}").status_code == 404

    def test_delete_trace_not_found(self, client):
        r = client.delete("/api/traces/doesnotexist")
        assert r.status_code == 404


# ── Job routes (status / CRUD, no heavy computation) ─────────────────────────


class TestJobRoutes:
    _OPT_BODY = {
        "type": "optimize",
        "optimize": {
            "workload": {"type": "builtin", "name": "azure"},
            "lam": 30.0,
            "slo_ms": 500.0,
            "b_short": 4096,
            "gpu_short": "a100",
            "gpu_long": "a100",
            "long_max_ctx": 65536,
            "gamma_min": 1.0,
            "gamma_max": 1.0,
            "gamma_step": 0.5,
            "n_sim_requests": 1000,
        },
    }

    def test_list_jobs_empty_initially(self, client):
        r = client.get("/api/jobs")
        assert r.status_code == 200
        assert r.json() == []

    def test_submit_optimize_missing_params_returns_422(self, client):
        r = client.post("/api/jobs", json={"type": "optimize"})
        assert r.status_code == 422

    def test_submit_simulate_missing_params_returns_422(self, client):
        r = client.post("/api/jobs", json={"type": "simulate"})
        assert r.status_code == 422

    def test_submit_whatif_missing_params_returns_422(self, client):
        r = client.post("/api/jobs", json={"type": "whatif"})
        assert r.status_code == 422

    def test_submit_job_creates_record(self, client):
        r = client.post("/api/jobs", json=self._OPT_BODY)
        assert r.status_code == 200
        data = r.json()
        assert "id" in data
        assert data["type"] == "optimize"
        assert data["status"] in ("pending", "running", "done", "failed")

    def test_get_job_not_found(self, client):
        r = client.get("/api/jobs/doesnotexist")
        assert r.status_code == 404

    def test_delete_job(self, client):
        job_id = client.post("/api/jobs", json=self._OPT_BODY).json()["id"]
        _wait_job_done(client, job_id)
        r = client.delete(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        assert client.get(f"/api/jobs/{job_id}").status_code == 404

    def test_delete_job_not_found(self, client):
        r = client.delete("/api/jobs/doesnotexist")
        assert r.status_code == 404

    def test_job_appears_in_list_after_submission(self, client):
        client.post("/api/jobs", json=self._OPT_BODY)
        r = client.get("/api/jobs")
        assert len(r.json()) >= 1


# ── Runner unit tests ─────────────────────────────────────────────────────────


class TestRunnerGpuResolver:
    def test_a100_resolves_to_manual_profile(self):
        from api.runner import _gpu
        from fleet_sim.gpu_profiles.profiles import A100_80GB
        assert _gpu("a100") is A100_80GB

    def test_h100_resolves_to_manual_profile(self):
        from api.runner import _gpu
        from fleet_sim.gpu_profiles.profiles import H100_80GB
        assert _gpu("h100") is H100_80GB

    def test_a10g_resolves_to_manual_profile(self):
        from api.runner import _gpu
        from fleet_sim.gpu_profiles.profiles import A10G
        assert _gpu("a10g") is A10G

    def test_uppercase_and_hyphens_normalised(self):
        from api.runner import _gpu
        from fleet_sim.gpu_profiles.profiles import A100_80GB
        assert _gpu("A100") is A100_80GB
        assert _gpu("a100-80gb") is A100_80GB

    def test_h200_resolves_via_catalog(self):
        from api.runner import _gpu
        p = _gpu("h200")
        assert p.W > 0
        assert p.cost_per_hr > 0

    def test_l40s_resolves_via_catalog(self):
        from api.runner import _gpu
        p = _gpu("l40s")
        assert p.W > 0

    def test_unknown_gpu_raises_value_error(self):
        from api.runner import _gpu
        with pytest.raises(ValueError, match="Unknown GPU profile"):
            _gpu("imaginary_gpu_xyz")


class TestRunnerCdfLoader:
    def test_load_builtin_azure(self):
        from api.models import WorkloadRef
        from api.runner import _load_cdf
        cdf = _load_cdf(WorkloadRef(type="builtin", name="azure"))
        assert len(cdf) > 0
        assert cdf[-1][1] == pytest.approx(1.0, abs=0.01)

    def test_load_builtin_lmsys(self):
        from api.models import WorkloadRef
        from api.runner import _load_cdf
        cdf = _load_cdf(WorkloadRef(type="builtin", name="lmsys"))
        assert len(cdf) > 0

    def test_load_builtin_not_found_raises(self):
        from api.models import WorkloadRef
        from api.runner import _load_cdf
        with pytest.raises(FileNotFoundError):
            _load_cdf(WorkloadRef(type="builtin", name="does_not_exist_xyz"))

    def test_load_trace_without_id_raises(self):
        from api.models import WorkloadRef
        from api.runner import _load_cdf
        with pytest.raises(ValueError, match="trace_id required"):
            _load_cdf(WorkloadRef(type="trace", trace_id=None))

    def test_cdf_from_jsonl_trace(self, tmp_path):
        from api.runner import _cdf_from_trace
        p = tmp_path / "trace.jsonl"
        p.write_text(_MINIMAL_JSONL)
        cdf = _cdf_from_trace(p, "jsonl")
        assert len(cdf) > 0
        assert cdf[-1][1] == pytest.approx(1.0)
        # Thresholds should be ascending
        thresholds = [row[0] for row in cdf]
        assert thresholds == sorted(thresholds)

    def test_cdf_from_csv_trace(self, tmp_path):
        from api.runner import _cdf_from_trace
        p = tmp_path / "trace.csv"
        p.write_text(_MINIMAL_CSV)
        cdf = _cdf_from_trace(p, "csv")
        assert len(cdf) > 0
        assert cdf[-1][1] == pytest.approx(1.0)

    def test_cdf_from_empty_file_raises(self, tmp_path):
        from api.runner import _cdf_from_trace
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        with pytest.raises(ValueError, match="empty"):
            _cdf_from_trace(p, "jsonl")


class TestRunnerOptimize:
    """Verify _run_optimize correctly maps OptimizationReport to OptResult."""

    @pytest.fixture(scope="class")
    def opt_result(self):
        from api.models import OptimizeParams, WorkloadRef
        from api.runner import _run_optimize
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
            n_sim_requests=1000,  # minimum allowed; DES runs quickly at this size
        )
        return _run_optimize(params)

    def test_returns_opt_result_type(self, opt_result):
        from api.models import OptResult
        assert isinstance(opt_result, OptResult)

    def test_best_point_present(self, opt_result):
        assert opt_result.best is not None

    def test_best_has_required_fields(self, opt_result):
        b = opt_result.best
        assert b.gamma >= 1.0
        assert b.n_s >= 0
        assert b.n_l >= 0
        assert b.total_gpus > 0
        assert b.annual_cost_kusd > 0
        assert isinstance(b.slo_met, bool)
        assert b.source in ("analytical", "des", "simulated")

    def test_sweep_nonempty(self, opt_result):
        assert len(opt_result.sweep) > 0

    def test_sweep_points_have_required_fields(self, opt_result):
        for pt in opt_result.sweep:
            assert pt.total_gpus > 0
            assert pt.annual_cost_kusd > 0
            assert pt.source in ("analytical", "des", "simulated")

    def test_baseline_cost_positive(self, opt_result):
        assert opt_result.baseline_annual_cost_kusd > 0

    def test_savings_nonnegative(self, opt_result):
        assert opt_result.savings_pct >= 0.0


class TestRunnerSimulate:
    """Verify _run_simulate correctly maps FleetSimResult to SimResult."""

    @pytest.fixture(scope="class")
    def sim_result(self):
        from api.models import FleetConfigIn, PoolConfigIn, SimulateParams, WorkloadRef
        from api.runner import _run_simulate
        params = SimulateParams(
            workload=WorkloadRef(type="builtin", name="azure"),
            fleet=FleetConfigIn(
                name="test",
                pools=[PoolConfigIn(pool_id="main", gpu="a100",
                                    n_gpus=8, max_ctx=4096)],
                router="length",
            ),
            lam=20.0,
            slo_ms=500.0,
            n_requests=500,
        )
        return _run_simulate(params)

    def test_returns_sim_result_type(self, sim_result):
        from api.models import SimResult
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
        p = sim_result.pools[0]
        assert p.pool_id == "main"
        assert p.n_gpus == 8
        assert p.p99_ttft_ms > 0
        assert p.cost_per_hr > 0

    def test_histogram_nonempty(self, sim_result):
        assert len(sim_result.ttft_histogram) > 0

    def test_histogram_counts_approximately_n_requests(self, sim_result):
        # The last bin uses an exclusive upper bound, so the maximum value may
        # not be counted. Allow for up to 1% of requests to be uncounted.
        total = sum(b.count for b in sim_result.ttft_histogram)
        assert total >= 495

    def test_two_pool_fleet_produces_two_pool_results(self):
        from api.models import FleetConfigIn, PoolConfigIn, SimulateParams, WorkloadRef
        from api.runner import _run_simulate
        params = SimulateParams(
            workload=WorkloadRef(type="builtin", name="azure"),
            fleet=FleetConfigIn(
                name="hetero",
                pools=[
                    PoolConfigIn(pool_id="short", gpu="a100", n_gpus=6, max_ctx=4096),
                    PoolConfigIn(pool_id="long",  gpu="h100", n_gpus=4, max_ctx=32768),
                ],
                router="length",
            ),
            lam=20.0,
            slo_ms=500.0,
            n_requests=500,
        )
        result = _run_simulate(params)
        assert len(result.pools) == 2
        pool_ids = {p.pool_id for p in result.pools}
        assert pool_ids == {"short", "long"}


class TestRunnerWhatif:
    """Verify _run_whatif correctly sweeps lambda and detects SLO breaks."""

    @pytest.fixture(scope="class")
    def whatif_result(self):
        from api.models import FleetConfigIn, PoolConfigIn, WhatifParams, WorkloadRef
        from api.runner import _run_whatif
        params = WhatifParams(
            workload=WorkloadRef(type="builtin", name="azure"),
            fleet=FleetConfigIn(
                name="test",
                pools=[PoolConfigIn(pool_id="main", gpu="a100",
                                    n_gpus=4, max_ctx=4096)],
                router="length",
            ),
            lam_range=[5.0, 15.0, 30.0],
            slo_ms=500.0,
            n_requests=300,
        )
        return _run_whatif(params)

    def test_returns_whatif_result_type(self, whatif_result):
        from api.models import WhatifResult
        assert isinstance(whatif_result, WhatifResult)

    def test_points_match_lam_range(self, whatif_result):
        assert len(whatif_result.points) == 3

    def test_points_ordered_by_lam(self, whatif_result):
        lams = [p.lam for p in whatif_result.points]
        assert lams == sorted(lams)

    def test_points_have_positive_p99(self, whatif_result):
        for p in whatif_result.points:
            assert p.fleet_p99_ttft_ms > 0

    def test_slo_compliance_in_range(self, whatif_result):
        for p in whatif_result.points:
            assert 0.0 <= p.fleet_slo_compliance <= 1.0

    def test_cost_positive(self, whatif_result):
        for p in whatif_result.points:
            assert p.annual_cost_kusd > 0

    def test_slo_break_detected_under_strict_slo(self):
        """A 1 ms SLO is physically impossible; slo_break_lam must be set."""
        from api.models import FleetConfigIn, PoolConfigIn, WhatifParams, WorkloadRef
        from api.runner import _run_whatif
        params = WhatifParams(
            workload=WorkloadRef(type="builtin", name="azure"),
            fleet=FleetConfigIn(
                name="tiny",
                pools=[PoolConfigIn(pool_id="main", gpu="a100",
                                    n_gpus=4, max_ctx=4096)],
                router="length",
            ),
            lam_range=[5.0, 20.0],
            slo_ms=1.0,   # 1 ms is physically impossible; always violated
            n_requests=200,
        )
        result = _run_whatif(params)
        assert result.slo_break_lam is not None
        assert result.slo_break_lam == pytest.approx(5.0)


# ── Simulator API contract tests ──────────────────────────────────────────────
#
# These tests verify that the simulator's internal objects expose exactly the
# attributes that api/runner.py depends on.  A change in a simulator class
# (e.g. renaming an attribute) that would crash the runner is caught here
# without requiring a full HTTP round-trip.


class TestSimulatorApiContract:
    """Regression tests for the internal simulator API surface used by runner."""

    @pytest.fixture(scope="class")
    def azure_cdf(self):
        return _load_cdf("azure")

    @pytest.fixture(scope="class")
    def opt_report(self, azure_cdf):
        from fleet_sim import A100_80GB, FleetOptimizer
        opt = FleetOptimizer(gpu_short=A100_80GB, gpu_long=A100_80GB,
                             B_short=4096, t_slo_ms=500)
        return opt.optimize(cdf=azure_cdf, lam=30, gammas=[1.0],
                            n_sim_requests=0, verbose=False)

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
        # best_simulated may be None when n_sim_requests=0
        assert hasattr(opt_report, "best_simulated")

    def test_sweep_result_gamma(self, opt_report):
        r = opt_report.analytical[0]
        assert hasattr(r, "gamma")
        assert r.gamma >= 1.0

    def test_sweep_result_n_s_n_l(self, opt_report):
        r = opt_report.analytical[0]
        assert hasattr(r, "n_s")
        assert hasattr(r, "n_l")
        assert r.n_s >= 0
        assert r.n_l >= 0

    def test_sweep_result_total_gpus(self, opt_report):
        r = opt_report.analytical[0]
        assert hasattr(r, "total_gpus")
        assert r.total_gpus > 0

    def test_sweep_result_annualised_cost_kusd(self, opt_report):
        r = opt_report.analytical[0]
        assert hasattr(r, "annualised_cost_kusd")
        assert r.annualised_cost_kusd > 0

    def test_sweep_result_p99_ttft_fields(self, opt_report):
        r = opt_report.analytical[0]
        assert hasattr(r, "p99_ttft_short_ms")
        assert hasattr(r, "p99_ttft_long_ms")

    def test_sweep_result_slo_met(self, opt_report):
        r = opt_report.analytical[0]
        assert hasattr(r, "slo_met")
        assert isinstance(r.slo_met, bool)

    def test_sweep_result_source(self, opt_report):
        r = opt_report.analytical[0]
        assert hasattr(r, "source")
        assert r.source in ("analytical", "des", "simulated")

    def test_fleet_sim_result_p99_ttft_ms(self, azure_cdf):
        from fleet_sim import A100_80GB, Fleet, FleetConfig, PoolConfig
        from fleet_sim.workload.synthetic import CdfWorkload, PoissonWorkload
        workload = CdfWorkload(azure_cdf)
        arrivals = PoissonWorkload(lam=20, length_gen=workload,
                                   n_requests=300).generate()
        cfg = FleetConfig(pools=[PoolConfig(pool_id="main", gpu=A100_80GB,
                                            n_gpus=8, max_ctx=4096)])
        result = Fleet(cfg).run(arrivals)
        # Attributes used in runner._fleet_sim_result_to_model
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
        workload = CdfWorkload(azure_cdf)
        arrivals = PoissonWorkload(lam=20, length_gen=workload,
                                   n_requests=200).generate()
        cfg = FleetConfig(pools=[PoolConfig(pool_id="main", gpu=A100_80GB,
                                            n_gpus=4, max_ctx=4096)])
        result = Fleet(cfg).run(arrivals)
        for pool in result.pools.values():
            assert hasattr(pool, "cost_per_hr")
            assert callable(pool.cost_per_hr)
            assert pool.cost_per_hr() > 0

    def test_pool_object_has_gpu_with_name(self, azure_cdf):
        from fleet_sim import A100_80GB, Fleet, FleetConfig, PoolConfig
        from fleet_sim.workload.synthetic import CdfWorkload, PoissonWorkload
        workload = CdfWorkload(azure_cdf)
        arrivals = PoissonWorkload(lam=20, length_gen=workload,
                                   n_requests=200).generate()
        cfg = FleetConfig(pools=[PoolConfig(pool_id="main", gpu=A100_80GB,
                                            n_gpus=4, max_ctx=4096)])
        result = Fleet(cfg).run(arrivals)
        for pool in result.pools.values():
            assert hasattr(pool, "gpu")
            assert hasattr(pool.gpu, "name")

    def test_calibrate_returns_four_tuple(self, azure_cdf):
        from fleet_sim import A100_80GB
        from fleet_sim.optimizer.base import _calibrate
        result = _calibrate(azure_cdf, pool_max=4096, gpu=A100_80GB)
        assert len(result) == 4
        mu_gpu, cv2, n_slots, mean_prefill = result
        assert mu_gpu > 0
        assert cv2 >= 0
        assert n_slots > 0
        assert mean_prefill >= 0

    def test_disagg_result_total_gpus_consistent(self):
        from fleet_sim import LLAMA_3_1_70B, DisaggFleetOptimizer, H100_SXM
        from fleet_sim.gpu_profiles import ProfileBuilder, ServingConfig
        builder = ProfileBuilder()
        prefill = builder.build(H100_SXM, LLAMA_3_1_70B,
                                ServingConfig(tp=4, dtype_bytes=2,
                                              mean_ctx_tokens=1024,
                                              phase="prefill"))
        decode = builder.build(H100_SXM, LLAMA_3_1_70B,
                               ServingConfig(tp=8, dtype_bytes=2,
                                             mean_ctx_tokens=1024,
                                             phase="decode"))
        opt = DisaggFleetOptimizer(prefill, decode,
                                   mean_isl=1024, mean_osl=256,
                                   slo_ttft_ms=2000, slo_tpot_ms=100,
                                   max_ctx=4096)
        r = opt.optimize(max_prefill=4, max_decode=4)
        assert r.total_gpus == r.n_prefill * r.prefill_gpus + r.n_decode * r.decode_gpus


# ── Full job lifecycle via HTTP (background tasks) ────────────────────────────


class TestJobLifecycle:
    """Submit jobs through the HTTP API and verify that results are stored
    correctly with the expected shape.  These exercise the full stack:
    route → runner → fleet_sim → storage."""

    def test_optimize_job_completes_and_has_result(self, client):
        body = {
            "type": "optimize",
            "optimize": {
                "workload": {"type": "builtin", "name": "azure"},
                "lam": 20.0,
                "slo_ms": 500.0,
                "b_short": 4096,
                "gpu_short": "a100",
                "gpu_long": "a100",
                "long_max_ctx": 65536,
                "gamma_min": 1.0,
                "gamma_max": 1.0,
                "gamma_step": 0.5,
                "n_sim_requests": 1000,
            },
        }
        r = client.post("/api/jobs", json=body)
        assert r.status_code == 200
        job_id = r.json()["id"]
        job = _wait_job_done(client, job_id)

        assert job["status"] == "done", f"Job failed: {job.get('error')}"
        assert job["result_optimize"] is not None
        opt = job["result_optimize"]
        assert "best" in opt
        assert "sweep" in opt
        assert opt["best"]["total_gpus"] > 0
        assert opt["savings_pct"] >= 0.0

    def test_simulate_job_completes_and_has_result(self, client):
        body = {
            "type": "simulate",
            "simulate": {
                "workload": {"type": "builtin", "name": "azure"},
                "fleet": {
                    "name": "test",
                    "pools": [{"pool_id": "main", "gpu": "a100",
                               "n_gpus": 4, "max_ctx": 4096}],
                    "router": "length",
                },
                "lam": 10.0,
                "slo_ms": 500.0,
                "n_requests": 500,
            },
        }
        r = client.post("/api/jobs", json=body)
        assert r.status_code == 200
        job_id = r.json()["id"]
        job = _wait_job_done(client, job_id)

        assert job["status"] == "done", f"Job failed: {job.get('error')}"
        assert job["result_simulate"] is not None
        sim = job["result_simulate"]
        assert sim["total_gpus"] == 4
        assert sim["fleet_p99_ttft_ms"] > 0
        assert len(sim["pools"]) == 1
        assert len(sim["ttft_histogram"]) > 0

    def test_simulate_job_with_saved_fleet(self, client):
        # Save a fleet first
        fleet = client.post("/api/fleets", json={
            "name": "saved",
            "pools": [{"pool_id": "main", "gpu": "a100",
                       "n_gpus": 4, "max_ctx": 4096}],
            "router": "length",
        }).json()
        body = {
            "type": "simulate",
            "simulate": {
                "workload": {"type": "builtin", "name": "azure"},
                "fleet_id": fleet["id"],
                "lam": 10.0,
                "slo_ms": 500.0,
                "n_requests": 300,
            },
        }
        r = client.post("/api/jobs", json=body)
        job = _wait_job_done(client, r.json()["id"])
        assert job["status"] == "done", f"Job failed: {job.get('error')}"
        assert job["result_simulate"]["total_gpus"] == 4

    def test_whatif_job_completes_and_has_result(self, client):
        body = {
            "type": "whatif",
            "whatif": {
                "workload": {"type": "builtin", "name": "azure"},
                "fleet": {
                    "name": "test",
                    "pools": [{"pool_id": "main", "gpu": "a100",
                               "n_gpus": 4, "max_ctx": 4096}],
                    "router": "length",
                },
                "lam_range": [5.0, 20.0],
                "slo_ms": 500.0,
                "n_requests": 300,
            },
        }
        r = client.post("/api/jobs", json=body)
        assert r.status_code == 200
        job_id = r.json()["id"]
        job = _wait_job_done(client, job_id)

        assert job["status"] == "done", f"Job failed: {job.get('error')}"
        assert job["result_whatif"] is not None
        wf = job["result_whatif"]
        assert len(wf["points"]) == 2

    def test_optimize_job_with_trace_workload(self, client):
        # Upload a trace
        trace_id = client.post(
            "/api/traces?fmt=jsonl",
            files={"file": ("t.jsonl",
                            io.BytesIO((_MINIMAL_JSONL * 30).encode()),
                            "text/plain")},
        ).json()["id"]

        body = {
            "type": "simulate",
            "simulate": {
                "workload": {"type": "trace", "trace_id": trace_id},
                "fleet": {
                    "name": "test",
                    "pools": [{"pool_id": "main", "gpu": "a100",
                               "n_gpus": 4, "max_ctx": 4096}],
                    "router": "length",
                },
                "lam": 5.0,
                "slo_ms": 500.0,
                "n_requests": 200,
            },
        }
        r = client.post("/api/jobs", json=body)
        job = _wait_job_done(client, r.json()["id"])
        assert job["status"] == "done", f"Job failed: {job.get('error')}"

    def test_failed_job_stores_error_message(self, client):
        body = {
            "type": "simulate",
            "simulate": {
                "workload": {"type": "trace", "trace_id": "nonexistent_trace"},
                "fleet": {
                    "name": "test",
                    "pools": [{"pool_id": "main", "gpu": "a100",
                               "n_gpus": 4, "max_ctx": 4096}],
                    "router": "length",
                },
                "lam": 10.0,
                "slo_ms": 500.0,
                "n_requests": 500,
            },
        }
        r = client.post("/api/jobs", json=body)
        job = _wait_job_done(client, r.json()["id"])
        assert job["status"] == "failed"
        assert job["error"] is not None
        assert len(job["error"]) > 0
