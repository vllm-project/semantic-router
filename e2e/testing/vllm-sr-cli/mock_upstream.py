"""A mock OpenAI upstream on a live `vllm-sr serve` stack, shared by the
integration modules that send chat requests through the Router.

Signed-off-by: vLLM-SR Team
"""

import json
import os
import time
from contextlib import contextmanager
from urllib import error as urllib_error
from urllib import request as urllib_request

DEFAULT_PROVIDER_MOCKER_IMAGE = "semantic-router-ci/provider-mocker:e2e-test"
PROVIDER_MOCKER_IMAGE_ENV = "E2E_PREBUILT_PROVIDER_MOCKER_IMAGE"
PROVIDER_MOCKER_PORT = 18080


class MockUpstreamMixin:
    """Run the provider mocker on the stack network and send it chat requests."""

    @contextmanager
    def _running_mock_upstream(
        self, container_name: str, *, expected_authorization: str | None = None
    ):
        """Run the mock OpenAI upstream on the active stack network."""
        image = os.getenv(PROVIDER_MOCKER_IMAGE_ENV) or os.getenv(
            "PROVIDER_MOCKER_IMAGE", DEFAULT_PROVIDER_MOCKER_IMAGE
        )
        expected_authorization_env = (
            ["-e", f"PROVIDER_MOCKER_EXPECT_AUTHORIZATION={expected_authorization}"]
            if expected_authorization is not None
            else []
        )
        result = self._run_subprocess(
            [
                self.container_runtime,
                "run",
                "-d",
                "--name",
                container_name,
                "--network",
                self.runtime_stack.network_name,
                "-e",
                "PROVIDER_MOCKER_SCENARIO=cli",
                *expected_authorization_env,
                "--entrypoint",
                "python3",
                image,
                "-u",
                "-m",
                "provider_mocker",
                "--port",
                str(PROVIDER_MOCKER_PORT),
            ],
            timeout=30,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"failed to start mock upstream: {result.stderr}",
        )
        self.assertTrue(
            self.wait_for_container_running(
                timeout=30,
                container_name=container_name,
            ),
            "mock upstream did not reach running state",
        )
        try:
            yield
        finally:
            self._run_subprocess(
                [self.container_runtime, "rm", "-f", container_name],
                timeout=30,
            )

    def _container_log_diagnostics(self, container_names: tuple[str, ...]) -> str:
        """Collect bounded logs for a failed mock request."""
        diagnostics = []
        for container_name in container_names:
            logs = self._run_subprocess(
                [
                    self.container_runtime,
                    "logs",
                    "--tail",
                    "80",
                    container_name,
                ],
                timeout=10,
            )
            diagnostics.append(
                f"{container_name}:\n{(logs.stdout + logs.stderr)[-4000:]}"
            )
        return "\n".join(diagnostics)

    def _send_mock_chat_completion(
        self,
        mock_container: str,
        *,
        request_path: str = "/v1/chat/completions",
        redact_values: tuple[str, ...] = (),
        request_headers: dict[str, str] | None = None,
        model: str = "test-model",
        content: str = "ping",
        listener_port: int = 8888,
    ):
        """Send a chat request, retrying until the local stack is ready.

        Returns the response headers of the first successful request.
        """
        listener_port += self.runtime_stack.port_offset
        request = urllib_request.Request(
            f"http://localhost:{listener_port}{request_path}",
            data=json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                }
            ).encode(),
            headers={"Content-Type": "application/json", **(request_headers or {})},
            method="POST",
        )
        deadline = time.time() + 60
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                with urllib_request.urlopen(request, timeout=10) as response:
                    self.assertEqual(response.status, 200)
                    response.read()
                    return response.headers
            except urllib_error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code}: {body}")
                time.sleep(2)
            except (
                urllib_error.URLError,
                ConnectionError,
                TimeoutError,
            ) as exc:
                last_error = exc
                time.sleep(2)

        diagnostics = self._container_log_diagnostics(
            (
                mock_container,
                self.ROUTER_CONTAINER_NAME,
                self.ENVOY_CONTAINER_NAME,
            )
        )
        for value in redact_values:
            diagnostics = diagnostics.replace(value, "[redacted]")
        self.fail(f"request did not reach mock upstream: {last_error}\n{diagnostics}")
