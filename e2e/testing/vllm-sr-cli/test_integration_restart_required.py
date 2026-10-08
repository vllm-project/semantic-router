#!/usr/bin/env python3
"""A Dashboard change the running Router can't take waits for `vllm-sr serve`.

A standalone Router binds its listeners at startup, so moving one is refused
as restart_required. The Dashboard saves the change as a pending activation
and answers "Restart required", `vllm-sr status` reports it, the Router keeps
serving the old port, and the next `vllm-sr serve` applies it.
"""

import json
import os
import secrets
import unittest
from urllib import error as urllib_error
from urllib import request as urllib_request

import yaml
from cli_test_base import CLITestBase
from mock_upstream import PROVIDER_MOCKER_PORT, MockUpstreamMixin
from serve_session import ServeSessionMixin


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
)
class TestRestartRequired(MockUpstreamMixin, ServeSessionMixin, CLITestBase):
    """The CLI applies what the Dashboard saved for a restart."""

    SERVE_GATEWAY = "standalone"

    def _dashboard(self, path, body=None, token=None):
        request = urllib_request.Request(
            f"{self.runtime_stack.dashboard_url}{path}",
            data=None if body is None else json.dumps(body).encode(),
            method="GET" if body is None else "POST",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {token}"} if token else {}),
            },
        )
        try:
            with urllib_request.urlopen(request, timeout=60) as response:
                return response.status, json.loads(response.read())
        except urllib_error.HTTPError as error:
            self.fail(f"{path}: HTTP {error.code}: {error.read().decode()[:500]}")

    def _serve(self):
        process = self._start_serve_background(
            env=os.environ.copy(), arguments=("--config", "config.yaml")
        )
        try:
            self._wait_for_serve_success(process)
        finally:
            self._stop_serve_process(process)

    def test_a_listener_move_waits_for_serve(self):
        self.print_test_header(
            "Restart required",
            "Moves the listener in the Dashboard; the next serve applies it",
        )
        mock_container = f"{self.runtime_stack.stack_name}-restart-upstream"
        self.write_minimal_canonical_config(
            provider="openai",
            base_url=f"http://{mock_container}:{PROVIDER_MOCKER_PORT}/v1",
        )
        with open(os.path.join(self.test_dir, "config.yaml"), encoding="utf-8") as f:
            moved = yaml.safe_load(f)
        moved["listeners"][0]["port"] = 8890

        self._serve()
        try:
            with self._running_mock_upstream(mock_container):
                self._send_mock_chat_completion(mock_container)
                self.assert_dashboard_holds_no_container_runtime()
                token = self._dashboard(
                    "/api/auth/bootstrap/register",
                    {
                        "email": "admin@example.com",
                        "name": "Admin",
                        "password": f"Restart-{secrets.token_hex(12)}",
                    },
                )[1]["token"]

                status, saved = self._dashboard(
                    "/api/router/config/update", moved, token
                )
                self.assertEqual(status, 202, saved)
                self.assertEqual(saved["status"], "restart_required", saved)
                self.assertEqual(
                    saved["message"], "Restart required: run `vllm-sr serve` to apply."
                )
                code, stdout, _stderr = self.run_cli(["status"], timeout=60)
                self.assertEqual(code, 0, stdout)
                self.assertIn("Restart required", stdout)
                # The Router keeps the listener it bound at startup.
                self._send_mock_chat_completion(mock_container)

                self._serve()
                headers = self._send_mock_chat_completion(
                    mock_container, listener_port=8890
                )
                self.assertEqual(headers.get("x-vsr-selected-model"), "test-model")
                code, stdout, _stderr = self.run_cli(["status"], timeout=60)
                self.assertNotIn("Restart required", stdout)
        finally:
            self.run_cli(["stop"], timeout=120)
        self.print_test_result(True, "the next serve applied the saved change")


if __name__ == "__main__":
    unittest.main()
