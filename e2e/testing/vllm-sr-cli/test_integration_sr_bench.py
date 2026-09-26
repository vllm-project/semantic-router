"""Actual managed-worker lifecycle through the public CLI, without inference.

The setup-mode Router and Envoy stay stopped. A deliberately missing preview
credential creates a durable failed run before any upstream HTTP request. This
is lifecycle evidence, not a benchmark quality or model-capability evaluation.
"""

import hashlib
import json
import os
import sqlite3
import time
import unittest
from pathlib import Path

from cli_test_base import CLITestBase
from serve_session import ServeSessionMixin


@unittest.skipUnless(
    os.getenv("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Requires real local runtime images and a container daemon",
)
class TestManagedBenchmarkIntegration(ServeSessionMixin, CLITestBase):
    """Bootstrap, reuse, stop, and explicitly reopen the same durable store."""

    def _isolated_environment(self):
        environment = os.environ.copy()
        for name in tuple(environment):
            if name.startswith("SR_BENCH_") or name in {
                "VLLM_SR_PLATFORM",
                "VLLM_SR_STATE_ROOT_DIR",
                "DISABLE_DASHBOARD",
            }:
                del environment[name]
        environment["VLLM_SR_STATE_ROOT_DIR"] = self.test_dir
        return environment

    def _cli_json(self, *arguments, expected_status=0):
        result = self._run_subprocess(
            ["vllm-sr", *arguments],
            env=self.environment,
            cwd=self.test_dir,
            timeout=self.DEFAULT_TIMEOUT,
        )
        self.assertEqual(result.returncode, expected_status, result.stderr[:500])
        return json.loads(result.stdout)

    def _start_setup(self):
        process = self._start_serve_background(env=self.environment)
        try:
            self._wait_for_serve_success(process)
        finally:
            self._stop_serve_process(process)
        self.assertEqual(
            self._explicit_container_status(self.ROUTER_CONTAINER_NAME), "created"
        )
        self.assertEqual(
            self._explicit_container_status(self.ENVOY_CONTAINER_NAME), "created"
        )
        return self._worker_identity()

    def _worker_identity(self):
        result = self._run_subprocess(
            [self.container_runtime, "inspect", self.SR_BENCH_CONTAINER_NAME],
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr[:500])
        container = json.loads(result.stdout)[0]
        self.assertEqual(container["State"]["Status"], "running")
        self.assertFalse(container["HostConfig"].get("DeviceRequests"))
        self.assertFalse(container["HostConfig"].get("Devices"))
        self.assertFalse(container["HostConfig"].get("Privileged"))
        self.assertNotIn(
            "/var/run/docker.sock",
            [mount["Destination"] for mount in container["Mounts"]],
        )
        self.assertNotIn(
            "SR_BENCH_LIFECYCLE_MISSING_TOKEN",
            [item.split("=", 1)[0] for item in container["Config"]["Env"]],
        )
        self.assertIn(
            str(self.store), [mount["Source"] for mount in container["Mounts"]]
        )
        for bindings in container["HostConfig"]["PortBindings"].values():
            for binding in bindings or []:
                self.assertEqual(binding["HostIp"], "127.0.0.1")
        return {
            "id": container["Id"],
            "pid": container["State"]["Pid"],
            "started_at": container["State"]["StartedAt"],
            "identity": container["Config"]["Labels"]["io.vllm-sr.sr-bench.identity"],
            "token_sha256": hashlib.sha256(self.token_file.read_bytes()).hexdigest(),
        }

    def _prepare_failure_fixture(self):
        source = Path(self.test_dir) / "lifecycle-fixture.json"
        source.write_text(
            json.dumps(
                [
                    {
                        "question_id": "synthetic-lifecycle-only",
                        "category": "lifecycle-fixture",
                        "question": "Synthetic lifecycle fixture: select A.",
                        "options": ["A", "B"],
                        "answer": "A",
                    }
                ]
            )
        )
        dataset = self._cli_json(
            "benchmark",
            "dataset",
            "prepare",
            "--local",
            "--benchmark",
            "mmlu-pro",
            "--profile",
            "smoke",
            "--limit",
            "1",
            "--source-path",
            str(source),
            "--revision",
            "synthetic-lifecycle-v1",
        )
        self.assertEqual(dataset["case_count"], 1)
        manifest = {
            "version": "sr-bench-1.0",
            "name": "Synthetic managed-worker lifecycle, no inference",
            "mode": "preview",
            "profile": "smoke",
            "cost_policy": "capability_only",
            "dataset": {"path": dataset["path"], "sha256": dataset["sha256"]},
            "targets": [
                {
                    "id": "missing-credential-fixture",
                    "kind": "mom",
                    "model": "auto",
                    "base_url": "http://127.0.0.1:1/v1",
                    "preview_url": "http://127.0.0.1:1/api/v1/routing/preview",
                    "preview_api_key_env": "SR_BENCH_LIFECYCLE_MISSING_TOKEN",
                }
            ],
            "limits": {
                "concurrency": 1,
                "total_timeout_s": 5,
                "idle_timeout_s": 2,
                "max_run_seconds": 30,
                "case_timeout_s": 10,
            },
        }
        path = Path(self.test_dir) / "lifecycle-manifest.json"
        path.write_text(json.dumps(manifest))
        self.assertEqual(
            self._cli_json("benchmark", "plan", "--manifest", str(path))["total"], 1
        )
        return path, dataset

    def _terminal_run(self, run_id):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            run = self._cli_json("benchmark", "show", run_id)
            if run["status"] in {"failed", "completed", "cancelled", "interrupted"}:
                return run
            time.sleep(0.1)
        self.fail("The no-inference lifecycle fixture did not terminate")

    def test_setup_managed_worker_reuses_identity_and_preserves_evidence(self):
        self.environment = self._isolated_environment()
        self.store = (
            Path(self.test_dir) / ".sr-bench" / self.runtime_stack.stack_name / "store"
        )
        self.token_file = self.store.parent / "service-token"
        first = self._start_setup()
        self.assertEqual(self.token_file.stat().st_mode & 0o077, 0)
        self.assertEqual(self._cli_json("benchmark", "runs")["runs"], [])
        manifest, dataset = self._prepare_failure_fixture()
        submit = (
            "benchmark",
            "preview",
            "--manifest",
            str(manifest),
            "--detach",
            "--idempotency-key",
            "managed-lifecycle-v1",
        )
        run_id = self._cli_json(*submit)["id"]
        run = self._terminal_run(run_id)
        self.assertEqual(run["status"], "failed")
        self.assertEqual(
            run["progress"], {"total": 1, "completed": 0, "failed": 1, "running": 0}
        )
        results = self._cli_json("benchmark", "show", run_id, "--results")
        self.assertIn("credential", json.dumps(results).lower())
        self.assertEqual(
            self._cli_json("benchmark", "show", run_id, "--calls")["total"], 0
        )

        # Re-running serve must keep the existing owner of the journal alive.
        self.assertEqual(self._start_setup(), first)
        self.assertEqual(self._cli_json(*submit)["id"], run_id)
        self.assertEqual(self._cli_json("benchmark", "show", run_id), run)
        self.assertEqual(
            self._cli_json("benchmark", "show", run_id, "--results"), results
        )
        self.assertEqual(
            hashlib.sha256(Path(dataset["path"]).read_bytes()).hexdigest(),
            dataset["sha256"],
        )

        self._stop_setup()
        unavailable = self._run_subprocess(
            ["vllm-sr", "benchmark", "runs"],
            env=self.environment,
            cwd=self.test_dir,
            timeout=10,
        )
        self.assertNotEqual(unavailable.returncode, 0)
        self.assertIn("unavailable", unavailable.stderr.lower())
        self.assertEqual(
            self._explicit_container_status(self.SR_BENCH_CONTAINER_NAME), "not found"
        )
        with sqlite3.connect(
            f"file:{self.store / 'journal.sqlite3'}?mode=ro", uri=True
        ) as db:
            self.assertEqual(db.execute("PRAGMA quick_check").fetchone()[0], "ok")
            self.assertEqual(db.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM calls").fetchone()[0], 0)

        # Only an explicit later serve may reopen this initialized store.
        reopened = self._start_setup()
        self.assertNotEqual(reopened["id"], first["id"])
        self.assertEqual(reopened["token_sha256"], first["token_sha256"])
        self.assertEqual(reopened["identity"], first["identity"])
        self.assertEqual(self._cli_json("benchmark", "show", run_id), run)
        self.assertEqual(
            self._cli_json("benchmark", "show", run_id, "--results"), results
        )
        self.assertEqual(
            self._cli_json("benchmark", "show", run_id, "--calls")["total"], 0
        )
        self._stop_setup()

    def _stop_setup(self):
        result = self._run_subprocess(
            ["vllm-sr", "stop"],
            env=self.environment,
            cwd=self.test_dir,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr[:500])


if __name__ == "__main__":
    unittest.main()
