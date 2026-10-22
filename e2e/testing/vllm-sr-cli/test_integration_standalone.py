#!/usr/bin/env python3
"""Standalone mode on the docker target: the Router serves the listener.

`vllm-sr serve` starts no Envoy container by default. A chat request sent to
the listener's host port reaches the mock OpenAI upstream through the Router
container alone, and the Router answers the readiness probe there itself.

Signed-off-by: vLLM-SR Team
"""

import json
import os
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
class TestStandaloneDocker(MockUpstreamMixin, ServeSessionMixin, CLITestBase):
    """standalone-docker: the Router container is the gateway."""

    SERVE_GATEWAY = "standalone"
    CONTAINER_STARTUP_TIMEOUT = 120

    def test_the_router_serves_the_listener_without_envoy(self):
        self.print_test_header(
            "Standalone Docker",
            "Routes one chat request through the Router's own listener",
        )
        mock_container = f"{self.runtime_stack.stack_name}-standalone-upstream"
        upstream = f"http://{mock_container}:{PROVIDER_MOCKER_PORT}/v1"
        with self._running_serve(api_only=True, base_url=upstream, provider="openai"):
            self.assertEqual(
                self.container_status(self.ENVOY_CONTAINER_NAME), "not found"
            )
            self.assert_dashboard_holds_no_container_runtime()
            listener_port = 8888 + self.runtime_stack.port_offset
            with urllib_request.urlopen(
                f"http://localhost:{listener_port}/ready", timeout=10
            ) as ready:
                self.assertEqual(ready.status, 200)
                self.assertEqual(json.loads(ready.read())["status"], "ready")
            self._assert_root_is_not_an_inference_request(listener_port)
            with urllib_request.urlopen(
                f"http://localhost:{listener_port}/v1/models", timeout=10
            ) as models:
                self.assertEqual(models.status, 200)
                self.assertEqual(json.loads(models.read())["object"], "list")
            with self._running_mock_upstream(mock_container):
                headers = self._send_mock_chat_completion(mock_container)
            self.assertNotIn("x-envoy-upstream-service-time", headers)
            self.assertEqual(headers.get("x-vsr-selected-model"), "test-model")

            return_code, stdout, _stderr = self.run_cli(["status"], timeout=60)
            self.assertEqual(return_code, 0)
            self.assertIn("Router", stdout)
        self.print_test_result(True, "the Router served the request with no Envoy")

    def _assert_root_is_not_an_inference_request(self, listener_port: int) -> None:
        # The backend has not started: these requests must be answered locally.
        for method, path in (("GET", "/"), ("GET", "/?trace=1"), ("HEAD", "/")):
            with self.subTest(method=method, path=path):
                request = urllib_request.Request(
                    f"http://localhost:{listener_port}{path}", method=method
                )
                with self.assertRaises(urllib_error.HTTPError) as captured:
                    urllib_request.urlopen(request, timeout=10)
                with captured.exception as response:
                    self.assertEqual(response.code, 404)
                    self.assertEqual(
                        response.headers.get("Content-Type"), "application/json"
                    )
                    body = response.read()
                if method == "HEAD":
                    self.assertEqual(body, b"")
                    continue
                error = json.loads(body)["error"]
                self.assertEqual(error["type"], "invalid_request_error")
                self.assertEqual(error["code"], "not_found")
                self.assertIn("POST /v1/chat/completions", error["message"])
                self.assertIn("GET /v1/models", error["message"])


if __name__ == "__main__":
    unittest.main()
