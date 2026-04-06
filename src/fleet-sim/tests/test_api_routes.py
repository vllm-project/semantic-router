"""HTTP route tests for the fleet-sim API."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from .api_test_support import _MINIMAL_CSV, _MINIMAL_JSONL, _wait_job_done


class TestSystemRoutes:
    def test_healthz(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_metadata(self, client):
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["service"] == "vllm-sr-sim"
        assert body["docs"] == "/api/docs"

    def test_docs_resolve_openapi_under_forwarded_prefix(self, client):
        response = client.get(
            "/api/docs", headers={"x-forwarded-prefix": "/api/fleet-sim"}
        )
        assert response.status_code == 200
        assert "/api/fleet-sim/api/openapi.json" in response.text


class TestWorkloadRoutes:
    def test_list_workloads_returns_four(self, client):
        response = client.get("/api/workloads")
        assert response.status_code == 200
        items = response.json()
        assert len(items) == 4
        assert {item["name"] for item in items} == {
            "azure",
            "lmsys",
            "lmsys_multiturn",
            "agent_heavy",
        }

    def test_builtin_workload_has_required_fields(self, client):
        response = client.get("/api/workloads")
        workload = response.json()[0]
        assert "name" in workload
        assert "description" in workload
        assert "path" in workload

    def test_get_cdf_azure(self, client):
        response = client.get("/api/workloads/azure/cdf")
        assert response.status_code == 200
        points = response.json()
        assert len(points) > 0
        first = points[0]
        assert "threshold" in first
        assert "cumulative_frac" in first
        assert 0.0 < first["cumulative_frac"] <= 1.0

    @pytest.mark.parametrize(
        "name", ["azure", "lmsys", "lmsys_multiturn", "agent_heavy"]
    )
    def test_get_cdf_all_builtins(self, client, name):
        response = client.get(f"/api/workloads/{name}/cdf")
        assert response.status_code == 200
        assert len(response.json()) > 0

    def test_get_cdf_unknown_returns_404(self, client):
        assert client.get("/api/workloads/nonexistent/cdf").status_code == 404

    def test_get_stats_returns_trace_stats_shape(self, client):
        response = client.get("/api/workloads/azure/stats")
        assert response.status_code == 200
        stats = response.json()
        for field in (
            "n_requests",
            "p50_prompt_tokens",
            "p99_prompt_tokens",
            "p50_output_tokens",
            "p99_output_tokens",
            "prompt_histogram",
        ):
            assert field in stats, f"Missing field: {field}"
        assert stats["p50_prompt_tokens"] > 0
        assert len(stats["prompt_histogram"]) > 0

    def test_get_stats_unknown_returns_404(self, client):
        assert client.get("/api/workloads/nonexistent/stats").status_code == 404


class TestGpuProfileRoutes:
    def test_list_profiles_returns_three(self, client):
        response = client.get("/api/gpu-profiles")
        assert response.status_code == 200
        assert len(response.json()) == 3

    def test_profile_has_required_fields(self, client):
        response = client.get("/api/gpu-profiles")
        for profile in response.json():
            for field in (
                "name",
                "W_ms",
                "H_ms_per_slot",
                "chunk",
                "blk_size",
                "total_kv_blks",
                "max_slots",
                "cost_per_hr",
            ):
                assert field in profile
            assert profile["W_ms"] > 0
            assert profile["cost_per_hr"] > 0


class TestFleetRoutes:
    _FLEET_BODY = {
        "name": "test-fleet",
        "pools": [
            {"pool_id": "short", "gpu": "a100", "n_gpus": 4, "max_ctx": 4096},
            {"pool_id": "long", "gpu": "h100", "n_gpus": 2, "max_ctx": 32768},
        ],
        "router": "length",
        "compress_gamma": None,
    }

    def test_list_fleets_empty_initially(self, client):
        response = client.get("/api/fleets")
        assert response.status_code == 200
        assert response.json() == []

    def test_create_fleet_returns_created(self, client):
        response = client.post("/api/fleets", json=self._FLEET_BODY)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-fleet"
        assert "id" in data
        assert data["total_gpus"] == 6
        assert data["estimated_cost_per_hr"] > 0
        assert data["estimated_annual_cost_kusd"] > 0

    def test_create_fleet_appears_in_list(self, client):
        client.post("/api/fleets", json=self._FLEET_BODY)
        assert len(client.get("/api/fleets").json()) == 1

    def test_get_fleet_by_id(self, client):
        created = client.post("/api/fleets", json=self._FLEET_BODY).json()
        response = client.get(f"/api/fleets/{created['id']}")
        assert response.status_code == 200
        assert response.json()["id"] == created["id"]

    def test_get_fleet_not_found(self, client):
        assert client.get("/api/fleets/doesnotexist").status_code == 404

    def test_delete_fleet(self, client):
        fleet_id = client.post("/api/fleets", json=self._FLEET_BODY).json()["id"]
        response = client.delete(f"/api/fleets/{fleet_id}")
        assert response.status_code == 200
        assert client.get(f"/api/fleets/{fleet_id}").status_code == 404

    def test_delete_fleet_not_found(self, client):
        assert client.delete("/api/fleets/doesnotexist").status_code == 404

    def test_fleet_cost_calculation(self, client):
        body = {
            **self._FLEET_BODY,
            "pools": [{"pool_id": "main", "gpu": "a10g", "n_gpus": 8, "max_ctx": 4096}],
        }
        data = client.post("/api/fleets", json=body).json()
        assert abs(data["estimated_cost_per_hr"] - 8.08) < 0.02
        assert data["estimated_annual_cost_kusd"] == pytest.approx(70.78, abs=0.5)


class TestTraceRoutes:
    def test_list_traces_empty_initially(self, client):
        response = client.get("/api/traces")
        assert response.status_code == 200
        assert response.json() == []

    def test_upload_jsonl_trace(self, client):
        response = client.post(
            "/api/traces?fmt=jsonl",
            files={
                "file": (
                    "test.jsonl",
                    io.BytesIO(_MINIMAL_JSONL.encode()),
                    "text/plain",
                )
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["n_requests"] == 3
        assert data["format"] == "jsonl"
        assert data["stats"] is not None

    def test_upload_csv_trace(self, client):
        response = client.post(
            "/api/traces?fmt=csv",
            files={
                "file": ("test.csv", io.BytesIO(_MINIMAL_CSV.encode()), "text/plain")
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["n_requests"] == 3
        assert data["format"] == "csv"

    def test_upload_trace_stats_have_correct_fields(self, client):
        response = client.post(
            "/api/traces?fmt=jsonl",
            files={
                "file": (
                    "test.jsonl",
                    io.BytesIO(_MINIMAL_JSONL.encode()),
                    "text/plain",
                )
            },
        )
        stats = response.json()["stats"]
        for field in (
            "n_requests",
            "p50_prompt_tokens",
            "p99_prompt_tokens",
            "p50_output_tokens",
            "p99_output_tokens",
            "prompt_histogram",
            "output_histogram",
        ):
            assert field in stats

    def test_uploaded_trace_appears_in_list(self, client):
        client.post(
            "/api/traces?fmt=jsonl",
            files={
                "file": (
                    "test.jsonl",
                    io.BytesIO(_MINIMAL_JSONL.encode()),
                    "text/plain",
                )
            },
        )
        assert len(client.get("/api/traces").json()) == 1

    def test_get_trace_by_id(self, client):
        trace_id = client.post(
            "/api/traces?fmt=jsonl",
            files={
                "file": (
                    "test.jsonl",
                    io.BytesIO(_MINIMAL_JSONL.encode()),
                    "text/plain",
                )
            },
        ).json()["id"]
        response = client.get(f"/api/traces/{trace_id}")
        assert response.status_code == 200
        assert response.json()["id"] == trace_id

    def test_get_trace_not_found(self, client):
        assert client.get("/api/traces/doesnotexist").status_code == 404

    def test_sample_trace(self, client):
        trace_id = client.post(
            "/api/traces?fmt=jsonl",
            files={
                "file": (
                    "test.jsonl",
                    io.BytesIO(_MINIMAL_JSONL.encode()),
                    "text/plain",
                )
            },
        ).json()["id"]
        response = client.get(f"/api/traces/{trace_id}/sample?limit=2")
        assert response.status_code == 200
        data = response.json()
        assert "records" in data
        assert "total" in data
        assert len(data["records"]) <= 2
        assert data["total"] == 3

    def test_sample_trace_not_found(self, client):
        assert client.get("/api/traces/doesnotexist/sample").status_code == 404

    def test_delete_trace(self, client):
        trace_id = client.post(
            "/api/traces?fmt=jsonl",
            files={
                "file": (
                    "test.jsonl",
                    io.BytesIO(_MINIMAL_JSONL.encode()),
                    "text/plain",
                )
            },
        ).json()["id"]
        response = client.delete(f"/api/traces/{trace_id}")
        assert response.status_code == 200
        assert client.get(f"/api/traces/{trace_id}").status_code == 404

    def test_delete_trace_not_found(self, client):
        assert client.delete("/api/traces/doesnotexist").status_code == 404

    def test_startup_seeds_example_traces_when_enabled(
        self, isolated_storage, monkeypatch, tmp_path
    ):
        seed_dir = tmp_path / "trace_samples"
        seed_dir.mkdir()
        (seed_dir / "router_decisions.semantic_router.jsonl").write_text(_MINIMAL_JSONL)
        (seed_dir / "generic_chat_mix.jsonl").write_text(_MINIMAL_JSONL)
        (seed_dir / "batch_spike_requests.csv").write_text(_MINIMAL_CSV)
        monkeypatch.setenv("VLLM_SR_SIM_SEED_EXAMPLE_TRACES", "true")
        monkeypatch.setenv("VLLM_SR_SIM_SEED_TRACE_DIR", str(seed_dir))

        from fleet_sim.api.app import app

        with TestClient(app, raise_server_exceptions=True) as seeded_client:
            response = seeded_client.get("/api/traces")

        assert response.status_code == 200
        traces = response.json()
        assert len(traces) == 3
        assert {trace["name"] for trace in traces} == {
            "Router decisions sample",
            "Generic chat mix",
            "Batch spike requests",
        }

    def test_seed_example_traces_uses_runtime_workdir_candidates(
        self, isolated_storage, monkeypatch, tmp_path
    ):
        runtime_root = tmp_path / "runtime"
        seed_dir = runtime_root / "examples" / "trace_samples"
        seed_dir.mkdir(parents=True)
        (seed_dir / "router_decisions.semantic_router.jsonl").write_text(_MINIMAL_JSONL)
        (seed_dir / "generic_chat_mix.jsonl").write_text(_MINIMAL_JSONL)
        (seed_dir / "batch_spike_requests.csv").write_text(_MINIMAL_CSV)
        monkeypatch.chdir(runtime_root)
        monkeypatch.setenv("VLLM_SR_SIM_SEED_EXAMPLE_TRACES", "true")
        monkeypatch.delenv("VLLM_SR_SIM_SEED_TRACE_DIR", raising=False)

        from fleet_sim.api import storage, trace_ingest

        assert trace_ingest.resolve_seed_trace_dir() == seed_dir
        assert trace_ingest.seed_example_traces_if_enabled() == 3
        assert len(storage.list_traces()) == 3


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
        response = client.get("/api/jobs")
        assert response.status_code == 200
        assert response.json() == []

    def test_submit_optimize_missing_params_returns_422(self, client):
        assert client.post("/api/jobs", json={"type": "optimize"}).status_code == 422

    def test_submit_simulate_missing_params_returns_422(self, client):
        assert client.post("/api/jobs", json={"type": "simulate"}).status_code == 422

    def test_submit_whatif_missing_params_returns_422(self, client):
        assert client.post("/api/jobs", json={"type": "whatif"}).status_code == 422

    def test_submit_job_creates_record(self, client):
        response = client.post("/api/jobs", json=self._OPT_BODY)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["type"] == "optimize"
        assert data["status"] in ("pending", "running", "done", "failed")

    def test_get_job_not_found(self, client):
        assert client.get("/api/jobs/doesnotexist").status_code == 404

    def test_delete_job(self, client):
        job_id = client.post("/api/jobs", json=self._OPT_BODY).json()["id"]
        _wait_job_done(client, job_id)
        response = client.delete(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        assert client.get(f"/api/jobs/{job_id}").status_code == 404

    def test_delete_job_not_found(self, client):
        assert client.delete("/api/jobs/doesnotexist").status_code == 404

    def test_job_appears_in_list_after_submission(self, client):
        client.post("/api/jobs", json=self._OPT_BODY)
        assert len(client.get("/api/jobs").json()) >= 1
