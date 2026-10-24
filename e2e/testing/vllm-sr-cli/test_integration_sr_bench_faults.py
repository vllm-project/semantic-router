from __future__ import annotations

import copy
import json
import os
import threading
import unittest
from pathlib import Path
from typing import Any

import requests
import yaml
from cli.sr_bench.service import Server
from cli.sr_bench.store import Store
from cli_test_base import CLITestBase
from mock_upstream import PROVIDER_MOCKER_PORT, MockUpstreamMixin
from serve_session import ServeSessionMixin


@unittest.skipUnless(
    os.getenv("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Requires the Router and provider-mocker images and a container daemon",
)
class TestBenchmarkFaultIntegration(MockUpstreamMixin, ServeSessionMixin, CLITestBase):
    SERVE_GATEWAY = "standalone"

    def test_fault_receipt_survives_real_router_fallback(self) -> None:
        mock_container = f"{self.runtime_stack.stack_name}-fault-upstream"
        config_path = Path(
            self.write_minimal_canonical_config(
                model_name="primary",
                base_url=f"http://{mock_container}:{PROVIDER_MOCKER_PORT}/v1",
                provider="openai",
            )
        )
        config = yaml.safe_load(config_path.read_text())
        fallback = copy.deepcopy(config["providers"]["models"][0])
        fallback.update(name="fallback", provider_model_id="fallback")
        config["providers"]["models"].append(fallback)
        config["routing"]["modelCards"].append({"name": "fallback"})
        config["routing"]["decisions"][0]["modelRefs"].append(
            {"model": "fallback", "use_reasoning": False}
        )
        config["global"] = {
            "router": {
                "fallback": {
                    "version": 1,
                    "enabled": True,
                    "max_attempts": 2,
                    "total_timeout": "10s",
                    "per_attempt_timeout": "5s",
                    "retryable_status_codes": [503],
                }
            }
        }
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        process = self._start_serve_background(arguments=("--minimal",))
        try:
            self._wait_for_serve_success(process)
            with self._running_mock_upstream(mock_container):
                self._send_mock_chat_completion(mock_container, model="vllm-sr/auto")
                self._run_fault_cases(mock_container)
        finally:
            if process.poll() not in {None, 0}:
                print(self._container_log_diagnostics((self.ROUTER_CONTAINER_NAME,)))
            self._stop_serve_process(process)

    def _run_fault_cases(self, mock_container: str) -> None:
        schedules = {
            "a-status": [{"call_index": 0, "status": 503}],
            "b-cut": [{"call_index": 0, "stream_cut_short": True}],
            "c-delay": [{"call_index": 0, "delay": 0.01}],
        }
        configured = self._run_subprocess(
            command=[
                self.container_runtime,
                "exec",
                mock_container,
                "python3",
                "-c",
                "import sys, urllib.request; "
                f"url = 'http://127.0.0.1:{PROVIDER_MOCKER_PORT}/debug/fault-schedule'; "
                "req = urllib.request.Request(url, data=sys.argv[1].encode(), "
                "headers={'Content-Type': 'application/json'}, method='POST'); "
                "print(urllib.request.urlopen(req, timeout=5).read().decode())",
                json.dumps(schedules),
            ],
            timeout=10,
        )
        self.assertEqual(configured.returncode, 0, configured.stderr)
        identity = requests.get(
            url=f"http://127.0.0.1:{self.runtime_stack.api_port}/api/v1/config/hash",
            timeout=10,
        )
        identity.raise_for_status()
        manifest = {
            "version": "sr-bench-1.0",
            "cost_policy": "capability_only",
            "targets": [
                {
                    "id": "router",
                    "kind": "mom",
                    "model": "vllm-sr/auto",
                    "base_url": f"http://127.0.0.1:{8888 + self.runtime_stack.port_offset}/v1",
                    "config_hash": identity.json()["active_runtime_hash"],
                    "max_inference_calls": 1,
                }
            ],
            "cases": [
                {
                    "id": name,
                    "benchmark": "mmlu-pro",
                    "messages": [{"role": "user", "content": "hello"}],
                    "answer": "A",
                }
                for name in ("a-status", "b-cut", "c-delay", "z-control")
            ],
            "fault_schedules": schedules,
            "limits": {
                "concurrency": 1,
                "total_timeout_s": 10,
                "idle_timeout_s": 5,
                "max_run_seconds": 60,
            },
        }
        manifest_path = Path(self.test_dir) / "faults.json"
        manifest_path.write_text(json.dumps(manifest))
        store = Store(Path(self.test_dir) / "bench")
        service = Server(address=("127.0.0.1", 0), store=store)
        thread = threading.Thread(target=service.serve_forever)
        thread.start()
        try:
            result = self._run_subprocess(
                command=[
                    "vllm-sr",
                    "benchmark",
                    "--url",
                    f"http://127.0.0.1:{service.server_port}",
                    "--store",
                    str(store.root),
                    "--no-autostart",
                    "run",
                    "--manifest",
                    str(manifest_path),
                ],
                timeout=75,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            report: dict[str, Any] = json.loads(result.stdout)
            faults = report["fault_summary"]
            self.assertEqual(faults["faulted_tasks"]["total"], 3)
            self.assertEqual(faults["unfaulted_tasks"]["completed"], 1)
            tasks = {task["case_id"]: task for task in faults["faulted_tasks"]["tasks"]}
            status_call = tasks["a-status"]["faulted_calls"][0]
            self.assertEqual(tasks["a-status"]["outcome"], "completed")
            self.assertEqual(status_call["selected_model"], "fallback")
            self.assertEqual(status_call["injected_fault"]["provider_model"], "primary")
            self.assertEqual(status_call["injected_fault"]["status"], 503)
            self.assertEqual(status_call["router_response"]["status"], 200)
            self.assertEqual(tasks["b-cut"]["outcome"], "failed")
            print(
                json.dumps(
                    {
                        "faulted_tasks": faults["faulted_tasks"]["total"],
                        "unfaulted_completed": faults["unfaulted_tasks"]["completed"],
                        "recovered_model": status_call["selected_model"],
                        "faulted_provider": status_call["injected_fault"][
                            "provider_model"
                        ],
                        "injected_status": status_call["injected_fault"]["status"],
                        "returned_status": status_call["router_response"]["status"],
                        "cut_outcome": tasks["b-cut"]["outcome"],
                    },
                    sort_keys=True,
                )
            )
        finally:
            service.shutdown()
            service.server_close()
            thread.join()
