from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
import unittest
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml
from cli.router_management_client import CONFIG_PATH, RouterManagementClient
from cli.sr_bench import agent_task
from cli.sr_bench.contracts import SESSION_AWARE
from cli.sr_bench.service import Server
from cli.sr_bench.store import Store
from cli.sr_bench.transport import SESSION_PHASES
from cli_test_base import CLITestBase
from mock_upstream import (
    DEFAULT_PROVIDER_MOCKER_IMAGE,
    PROVIDER_MOCKER_IMAGE_ENV,
    PROVIDER_MOCKER_PORT,
    MockUpstreamMixin,
)
from test_integration_engine_mode import available_offset

ROOT: Path = Path(__file__).resolve().parents[3]
LISTENER_PORT: int = 8899
TEST_BUDGET_SECONDS: int = 600
CLEANUP_RESERVE_SECONDS: int = 60
MODEL: str = "test-model"


@unittest.skipUnless(
    os.getenv("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Requires cached CPU Router/provider-mocker images and a container daemon",
)
class TestAgentTaskRouterIntegration(MockUpstreamMixin, CLITestBase):
    _owns_namespace: bool = False

    @classmethod
    def setUpClass(cls) -> None:
        cls._owns_namespace = False
        environment = patch.dict(
            os.environ,
            {
                "VLLM_SR_STACK_NAME": "agent-task-" + secrets.token_hex(6),
                "VLLM_SR_PORT_OFFSET": str(available_offset()),
                "HF_HUB_OFFLINE": "1",
            },
        )
        environment.start()
        cls.addClassCleanup(environment.stop)
        super().setUpClass()
        cls.AUXILIARY_CONTAINER_NAMES += (cls.runtime_stack.stack_name + "-upstream",)
        for resource, template in (
            ("container", "{{.Names}}"),
            ("network", "{{.Name}}"),
            ("volume", "{{.Name}}"),
        ):
            result = cls._run_subprocess(
                command=[
                    cls.container_runtime,
                    resource,
                    "ls",
                    *(["--all"] if resource == "container" else []),
                    "--format",
                    template,
                ],
                timeout=10,
            )
            if result.returncode or any(
                name.startswith(cls.runtime_stack.stack_name)
                for name in result.stdout.splitlines()
            ):
                raise RuntimeError("Agent-task test namespace is not available")
        cls._owns_namespace = True
        image = (
            os.getenv(PROVIDER_MOCKER_IMAGE_ENV)
            or os.getenv("PROVIDER_MOCKER_IMAGE")
            or DEFAULT_PROVIDER_MOCKER_IMAGE
        )
        result = cls._run_subprocess(
            command=[cls.container_runtime, "image", "inspect", image],
            timeout=10,
        )
        if result.returncode:
            raise RuntimeError(
                "The agent-task test requires a cached provider-mocker image"
            )

    @classmethod
    def _cleanup_container(cls) -> None:
        if not cls._owns_namespace:
            return
        super()._cleanup_container()
        for network in (cls.NETWORK_NAME, cls.DATA_NETWORK_NAME):
            exists = cls._run_subprocess(
                command=[cls.container_runtime, "network", "inspect", network],
                timeout=10,
            )
            if exists.returncode == 0:
                removed = cls._run_subprocess(
                    command=[cls.container_runtime, "network", "rm", network],
                    timeout=10,
                )
                if removed.returncode:
                    raise RuntimeError(removed.stderr)
        volumes = cls._run_subprocess(
            command=[cls.container_runtime, "volume", "ls", "--format", "{{.Name}}"],
            timeout=10,
        )
        if volumes.returncode:
            raise RuntimeError(volumes.stderr)
        for volume in volumes.stdout.splitlines():
            if volume.startswith(cls.runtime_stack.stack_name + "-"):
                removed = cls._run_subprocess(
                    command=[cls.container_runtime, "volume", "rm", volume],
                    timeout=10,
                )
                if removed.returncode:
                    raise RuntimeError(removed.stderr)

    def _remaining(self, cap: int) -> int:
        remaining = int(self.deadline - time.monotonic()) - CLEANUP_RESERVE_SECONDS
        self.assertGreater(
            remaining, 0, "Agent-task Router test exceeded its work budget"
        )
        return min(cap, remaining)

    def _write_session_config(self, mock_container: str) -> None:
        path = Path(
            self.write_minimal_canonical_config(
                port=LISTENER_PORT,
                model_name=MODEL,
                base_url=f"http://{mock_container}:{PROVIDER_MOCKER_PORT}/v1",
                provider="openai",
                api_only=True,
            )
        )
        config = yaml.safe_load(path.read_text())
        config["global"].setdefault("router", {})["learning"] = {
            "enabled": True,
            "adaptation": {"enabled": False},
            "protection": {
                "enabled": True,
                "scope": "session",
                "identity": {"headers": {"session": "x-session-id"}},
            },
        }
        path.write_text(yaml.safe_dump(config, sort_keys=False))

    def _router_hash(self) -> str:
        response = RouterManagementClient(
            base_url=f"http://127.0.0.1:{self.runtime_stack.api_port}",
            timeout=self._remaining(5),
            token_env="",
        ).request(method="GET", path=CONFIG_PATH + "/hash")
        value = response.payload
        self.assertEqual(value["activation_status"], "active", value)
        self.assertEqual(
            value["generated_runtime_hash"], value["active_runtime_hash"], value
        )
        self.assertRegex(value["active_runtime_hash"], r"^[0-9a-f]{64}$")
        return value["active_runtime_hash"]

    def _native_run(
        self, base_url: str, config_hash: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        directory = Path(self.test_dir) / "native"
        result = self._run_subprocess(
            command=[
                sys.executable,
                str(ROOT / "bench/agent_task_live_benchmark.py"),
                "--base-url",
                base_url,
                "--model",
                "vllm-sr/auto",
                "--protocol",
                "chat",
                "--continuation-mode",
                "full-history",
                "--suite",
                "smoke",
                "--session-prefix",
                self.runtime_stack.stack_name + "-native",
                "--timeout",
                "5",
                "--max-tokens",
                "96",
                "--extra-header",
                "x-vsr-debug=true",
                "--extra-header",
                f"x-sr-bench-expected-config-hash={config_hash}",
                "--require-router-header",
                "x-vsr-selected-model",
                "--require-router-header",
                "x-vsr-session-phase",
                "--min-success-rate",
                "1",
                "--output-dir",
                str(directory),
            ],
            cwd=self.test_dir,
            timeout=self._remaining(90),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return (
            json.loads((directory / "summary.json").read_text()),
            [
                json.loads(line)
                for line in (directory / "turns.jsonl").read_text().splitlines()
            ],
        )

    def _shared_run(
        self, base_url: str, config_hash: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        store = Store(Path(self.test_dir) / "sr-bench")
        server = Server(("127.0.0.1", 0), store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            manifest = {
                "version": "sr-bench-1.0",
                "profile": "quick",
                "seed": 0,
                "dataset": agent_task.prepare(store=store.root, suite="smoke"),
                "cost_policy": "capability_only",
                "targets": [
                    {
                        "id": "router",
                        "kind": "mom",
                        "model": "vllm-sr/auto",
                        "base_url": base_url,
                        "preview_url": f"http://127.0.0.1:{self.runtime_stack.api_port}/api/v1/routing/preview",
                        "config_hash": config_hash,
                        "session_mode": SESSION_AWARE,
                    }
                ],
                "benchmark_options": {
                    agent_task.BENCHMARK_ID: {
                        "grader_version": agent_task.GRADER_VERSION
                    }
                },
                "sampling": {"temperature": 0, "max_tokens": 96},
                "limits": {
                    "concurrency": 1,
                    "max_calls_per_case": 3,
                    "total_timeout_s": 5,
                    "idle_timeout_s": 2,
                    "case_timeout_s": 15,
                    "max_run_seconds": 60,
                },
            }
            path = Path(self.test_dir) / "run.json"
            path.write_text(json.dumps(manifest))
            code, stdout, stderr = self.run_cli(
                args=[
                    "benchmark",
                    "--url",
                    f"http://127.0.0.1:{server.server_port}",
                    "--store",
                    str(store.root),
                    "--no-autostart",
                    "run",
                    "--manifest",
                    str(path),
                    "--idempotency-key",
                    "agent-task-router-once",
                ],
                timeout=self._remaining(90),
            )
            if code:
                self.fail(
                    stdout
                    + stderr
                    + self._container_log_diagnostics((self.ROUTER_CONTAINER_NAME,))
                )
            report = json.loads(stdout)
            run_id = store.list()[0]["id"]
            return report, store.calls(run_id)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            for worker in server.engine.threads.values():
                worker.join(timeout=10)
                self.assertFalse(worker.is_alive(), "Benchmark worker did not stop")
            store.db.close()

    def test_native_and_shared_smoke_tasks_use_real_router_sessions(self) -> None:
        started = time.monotonic()
        self.deadline = started + TEST_BUDGET_SECONDS
        mock_container = self.runtime_stack.stack_name + "-upstream"
        self._write_session_config(mock_container=mock_container)
        code, stdout, stderr = self.run_cli(
            args=[
                "serve",
                "--config",
                "config.yaml",
                "--gateway",
                "standalone",
                "--minimal",
                "--image-pull-policy",
                "never",
                "--startup-timeout",
                "120",
            ],
            timeout=self._remaining(180),
        )
        if code:
            self.fail(
                stdout
                + stderr
                + self._container_log_diagnostics((self.ROUTER_CONTAINER_NAME,))
            )
        try:
            self.assertEqual(
                self.container_status(self.ENVOY_CONTAINER_NAME), "not found"
            )
            self.assertEqual(
                self.container_status(self.DASHBOARD_CONTAINER_NAME), "not found"
            )
            self.assertEqual(
                self.container_status(self.SR_BENCH_CONTAINER_NAME), "not found"
            )
            with self._running_mock_upstream(mock_container):
                headers = self._send_mock_chat_completion(
                    mock_container,
                    listener_port=LISTENER_PORT,
                    model="vllm-sr/auto",
                    request_headers={
                        "x-vsr-debug": "true",
                        "x-session-id": self.runtime_stack.stack_name + "-readiness",
                    },
                )
                self.assertIn("protection", headers.get("x-vsr-learning-methods", ""))
                self.assertIn(
                    "protection=session", headers.get("x-vsr-learning-scopes", "")
                )
                config_hash = self._router_hash()
                base_url = f"http://127.0.0.1:{LISTENER_PORT + self.runtime_stack.port_offset}/v1"
                native, native_rows = self._native_run(
                    base_url=base_url, config_hash=config_hash
                )
                print(
                    json.dumps(
                        {
                            "stage": "native",
                            "router_config_hash": config_hash,
                            "requests": native["requests"],
                            "task_instances": native["task_instances"],
                            "mean_credit": native["task_score_mean"],
                            "observed_phases": [
                                row["x-vsr-session-phase"] for row in native_rows
                            ],
                        },
                        sort_keys=True,
                    )
                )
                report, calls = self._shared_run(
                    base_url=base_url, config_hash=config_hash
                )
                self.assertEqual(self._router_hash(), config_hash)
        finally:
            code, stdout, stderr = self.run_cli(args=["stop"], timeout=45)
            self.assertEqual(code, 0, stdout + stderr)

        target = report["summary"]["targets"][0]
        quality = target["token_coverage"]
        self.assertEqual((native["requests"], native["task_instances"]), (12, 4))
        self.assertEqual((quality["planned"], quality["graded"]), (4, 4))
        self.assertEqual(target["request_count"], 12)
        self.assertTrue(quality["complete"])
        self.assertEqual(quality["mean_credit"], native["task_score_mean"])
        self.assertEqual(quality["full_completion_rate"], native["task_success_rate"])
        self.assertEqual(
            target["continuity"]["model_switches"], native["model_switches"]
        )
        self.assertEqual(target["selected_models"], {MODEL: 12})
        self.assertIsNone(target["sr_bench_score"])
        self.assertTrue(all(call["config_hash"] == config_hash for call in calls))

        native_sessions: dict[str, set[str]] = defaultdict(set)
        shared_sessions: dict[str, set[str]] = defaultdict(set)
        native_phases: dict[str, list[str]] = defaultdict(list)
        shared_phases: dict[str, list[str]] = defaultdict(list)
        for row in native_rows:
            native_sessions[row["task"]].add(row["session_id"])
            native_phases[row["task"]].append(row["x-vsr-session-phase"])
        for turn, call in zip(quality["turns"], calls, strict=True):
            shared_sessions[turn["task"]].add(call["session_id"])
            shared_phases[turn["task"]].append(turn["observed_phase"])
            self.assertEqual(call["phase_source"], "router")
            self.assertIn(call["phase"], SESSION_PHASES)
            self.assertEqual(turn["selected_model"], MODEL)
            self.assertEqual(turn["session_id"], call["session_id"])
        self.assertEqual(dict(native_phases), dict(shared_phases))
        for sessions in (native_sessions, shared_sessions):
            self.assertEqual(len(sessions), 4)
            self.assertTrue(all(len(values) == 1 for values in sessions.values()))
            self.assertEqual(len(set.union(*sessions.values())), 4)
        self.assertTrue(
            set.union(*native_sessions.values()).isdisjoint(
                set.union(*shared_sessions.values())
            )
        )
        self.assertLess(time.monotonic() - started, TEST_BUDGET_SECONDS)
        print(
            json.dumps(
                {
                    "router_config_hash": config_hash,
                    "requests_per_arm": 12,
                    "graded_tasks": quality["graded"],
                    "native_mean_credit": native["task_score_mean"],
                    "sr_bench_mean_credit": quality["mean_credit"],
                    "full_completion_rate": quality["full_completion_rate"],
                    "native_phases": native_phases,
                    "sr_bench_phases": shared_phases,
                    "native_sessions": {
                        name: sorted(values) for name, values in native_sessions.items()
                    },
                    "sr_bench_sessions": {
                        name: sorted(values) for name, values in shared_sessions.items()
                    },
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
