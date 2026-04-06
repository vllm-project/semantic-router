"""Background job lifecycle tests for the fleet-sim API."""

from __future__ import annotations

import io

from .api_test_support import _MINIMAL_JSONL, _wait_job_done


class TestJobLifecycle:
    """Submit jobs through the HTTP API and verify stored result shapes."""

    def test_optimize_job_completes_and_has_result(self, client):
        response = client.post(
            "/api/jobs",
            json={
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
            },
        )
        assert response.status_code == 200
        job = _wait_job_done(client, response.json()["id"])
        assert job["status"] == "done", f"Job failed: {job.get('error')}"
        assert job["result_optimize"] is not None
        result = job["result_optimize"]
        assert "best" in result
        assert "sweep" in result
        assert result["best"]["total_gpus"] > 0
        assert result["savings_pct"] >= 0.0

    def test_simulate_job_completes_and_has_result(self, client):
        response = client.post(
            "/api/jobs",
            json={
                "type": "simulate",
                "simulate": {
                    "workload": {"type": "builtin", "name": "azure"},
                    "fleet": {
                        "name": "test",
                        "pools": [
                            {
                                "pool_id": "main",
                                "gpu": "a100",
                                "n_gpus": 4,
                                "max_ctx": 4096,
                            }
                        ],
                        "router": "length",
                    },
                    "lam": 10.0,
                    "slo_ms": 500.0,
                    "n_requests": 500,
                },
            },
        )
        assert response.status_code == 200
        job = _wait_job_done(client, response.json()["id"])
        assert job["status"] == "done", f"Job failed: {job.get('error')}"
        result = job["result_simulate"]
        assert result is not None
        assert result["total_gpus"] == 4
        assert result["fleet_p99_ttft_ms"] > 0
        assert len(result["pools"]) == 1
        assert len(result["ttft_histogram"]) > 0

    def test_simulate_job_with_saved_fleet(self, client):
        fleet = client.post(
            "/api/fleets",
            json={
                "name": "saved",
                "pools": [
                    {"pool_id": "main", "gpu": "a100", "n_gpus": 4, "max_ctx": 4096}
                ],
                "router": "length",
            },
        ).json()
        response = client.post(
            "/api/jobs",
            json={
                "type": "simulate",
                "simulate": {
                    "workload": {"type": "builtin", "name": "azure"},
                    "fleet_id": fleet["id"],
                    "lam": 10.0,
                    "slo_ms": 500.0,
                    "n_requests": 300,
                },
            },
        )
        job = _wait_job_done(client, response.json()["id"])
        assert job["status"] == "done", f"Job failed: {job.get('error')}"
        assert job["result_simulate"]["total_gpus"] == 4

    def test_whatif_job_completes_and_has_result(self, client):
        response = client.post(
            "/api/jobs",
            json={
                "type": "whatif",
                "whatif": {
                    "workload": {"type": "builtin", "name": "azure"},
                    "fleet": {
                        "name": "test",
                        "pools": [
                            {
                                "pool_id": "main",
                                "gpu": "a100",
                                "n_gpus": 4,
                                "max_ctx": 4096,
                            }
                        ],
                        "router": "length",
                    },
                    "lam_range": [5.0, 20.0],
                    "slo_ms": 500.0,
                    "n_requests": 300,
                },
            },
        )
        assert response.status_code == 200
        job = _wait_job_done(client, response.json()["id"])
        assert job["status"] == "done", f"Job failed: {job.get('error')}"
        assert len(job["result_whatif"]["points"]) == 2

    def test_optimize_job_with_trace_workload(self, client):
        trace_id = client.post(
            "/api/traces?fmt=jsonl",
            files={
                "file": (
                    "t.jsonl",
                    io.BytesIO((_MINIMAL_JSONL * 30).encode()),
                    "text/plain",
                )
            },
        ).json()["id"]
        response = client.post(
            "/api/jobs",
            json={
                "type": "simulate",
                "simulate": {
                    "workload": {"type": "trace", "trace_id": trace_id},
                    "fleet": {
                        "name": "test",
                        "pools": [
                            {
                                "pool_id": "main",
                                "gpu": "a100",
                                "n_gpus": 4,
                                "max_ctx": 4096,
                            }
                        ],
                        "router": "length",
                    },
                    "lam": 5.0,
                    "slo_ms": 500.0,
                    "n_requests": 200,
                },
            },
        )
        job = _wait_job_done(client, response.json()["id"])
        assert job["status"] == "done", f"Job failed: {job.get('error')}"

    def test_failed_job_stores_error_message(self, client):
        response = client.post(
            "/api/jobs",
            json={
                "type": "simulate",
                "simulate": {
                    "workload": {"type": "trace", "trace_id": "nonexistent_trace"},
                    "fleet": {
                        "name": "test",
                        "pools": [
                            {
                                "pool_id": "main",
                                "gpu": "a100",
                                "n_gpus": 4,
                                "max_ctx": 4096,
                            }
                        ],
                        "router": "length",
                    },
                    "lam": 10.0,
                    "slo_ms": 500.0,
                    "n_requests": 500,
                },
            },
        )
        job = _wait_job_done(client, response.json()["id"])
        assert job["status"] == "failed"
        assert job["error"] is not None
        assert len(job["error"]) > 0
