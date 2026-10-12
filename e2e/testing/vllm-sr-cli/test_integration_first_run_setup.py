#!/usr/bin/env python3
"""First-run setup on the docker target, with no container runtime in the Dashboard.

`vllm-sr serve` in an empty directory opens the Dashboard's setup and keeps
waiting. Setup through the Dashboard's API writes the config, and the waiting
CLI then creates the Router itself, from the activated config: the test moves
the listener from setup's port to another one, and a chat request routes
through the Router on the new port. The Dashboard mounts no container socket,
has no container CLI, and reports the Router waiting for setup, then running.
After setup, the Dashboard still saves config changes: one the Router
hot-reloads is accepted while the stack runs.
"""

import json
import os
import secrets
import unittest
from urllib import error as urllib_error
from urllib import request as urllib_request

from cli_test_base import CLITestBase
from mock_upstream import PROVIDER_MOCKER_PORT, MockUpstreamMixin
from serve_session import ServeSessionMixin


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
)
class TestFirstRunSetup(MockUpstreamMixin, ServeSessionMixin, CLITestBase):
    """The CLI, not the Dashboard, starts the Router after setup."""

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
            with urllib_request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib_error.HTTPError as error:
            self.fail(f"{path}: HTTP {error.code}: {error.read().decode()[:500]}")

    def _status_service(self, name):
        services = self._dashboard("/api/status")["services"]
        return next(service for service in services if service["name"] == name)

    def _activation_config(self, mock_container):
        return {
            "listeners": [{"name": "http-8888", "address": "0.0.0.0", "port": 8888}],
            "providers": {
                "defaults": {"model": "test-model"},
                "models": [
                    {
                        "name": "test-model",
                        "provider_model_id": "test-model",
                        "backend_refs": [
                            {
                                "name": "primary",
                                "provider": "openai",
                                "weight": 100,
                                "base_url": f"http://{mock_container}:{PROVIDER_MOCKER_PORT}/v1",
                            }
                        ],
                    }
                ],
            },
            "routing": {
                "modelCards": [{"name": "test-model"}],
                "decisions": [
                    {
                        "name": "default-route",
                        "description": "Default route",
                        "priority": 100,
                        "rules": {"operator": "AND", "conditions": []},
                        "modelRefs": [{"model": "test-model", "use_reasoning": False}],
                    }
                ],
            },
        }

    def test_setup_activation_starts_the_router_from_the_cli(self):
        self.print_test_header(
            "First-run setup",
            "Activates setup through the Dashboard; serve starts the Router",
        )
        serve_process = self._start_serve_background(env=os.environ.copy())
        try:
            self._wait_for_setup_mode(serve_process)
            self.assertEqual(
                self._explicit_container_status(self.ROUTER_CONTAINER_NAME), "created"
            )
            self.assert_dashboard_holds_no_container_runtime()
            # Without a container runtime, the Dashboard reads the stack's
            # state from the setup config and the CLI's heartbeat.
            router = self._status_service("Router")
            self.assertEqual(router["status"], "standby", router)

            token = self._dashboard(
                "/api/auth/bootstrap/register",
                {
                    "email": "admin@example.com",
                    "name": "Admin",
                    "password": f"Setup-{secrets.token_hex(12)}",
                },
            )["token"]
            mock_container = f"{self.runtime_stack.stack_name}-setup-upstream"
            with self._running_mock_upstream(mock_container):
                activated = self._dashboard(
                    "/api/setup/activate",
                    {"config": self._activation_config(mock_container)},
                    token,
                )
                self.assertEqual(activated["status"], "success", activated)
                self.assertEqual(
                    activated["message"], "Setup saved. The Router is starting."
                )
                # serve starts the Router, waits for it and exits.
                output = "".join(self._wait_for_serve_success(serve_process))
                # A standalone first run names no Envoy, and no older stack's
                # orphaned volumes on a host that never ran one.
                self.assertIn("with the Router on standby", output)
                self.assertNotIn("Envoy", output)
                self.assertNotIn("orphaned volumes", output)
                headers = self._send_mock_chat_completion(mock_container)
                self.assertEqual(headers.get("x-vsr-selected-model"), "test-model")
                router = self._status_service("Router")
                self.assertTrue(router["healthy"], router)

                edited = self._activation_config(mock_container)
                edited["routing"]["decisions"][0]["description"] = "Edited after setup"
                saved = self._dashboard("/api/router/config/update", edited, token)
                self.assertEqual(saved, {"status": "success"})
                headers = self._send_mock_chat_completion(mock_container)
            self.assertEqual(headers.get("x-vsr-selected-model"), "test-model")
            self.assertEqual(
                self._explicit_container_status(self.ROUTER_CONTAINER_NAME), "running"
            )
        finally:
            self._stop_serve_process(serve_process)
        self.print_test_result(
            True, "setup activated; the CLI started the Router; a later save applied"
        )


if __name__ == "__main__":
    unittest.main()
