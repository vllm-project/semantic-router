#!/usr/bin/env python3
"""A listener's apis on the Envoy listener `vllm-sr serve --gateway extproc` generates.

The Router applies a listener's apis to the path the client sent. Its
request-body phase then sets the provider's path and clears Envoy's route
cache, so a grant matched on the routes would read the provider path instead
(#7447): it refused a Messages client in front of a Chat Completions backend,
and passed a Chat Completions client in front of a Messages backend.

The claude listener serves only messages. The chat model's backend has a base
path, so its provider path differs from any client path. The open listener
names no apis and shows the Messages backend answers chat requests. Envoy names
the listener in x-vsr-listener, replacing a client's value.
"""

import json
import os
import unittest
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

import yaml
from cli_test_base import CLITestBase
from mock_upstream import PROVIDER_MOCKER_PORT, MockUpstreamMixin
from serve_session import ServeSessionMixin

CLAUDE_PORT = 8888
OPEN_PORT = 8889
CHAT_MODEL = "chat-model"
MESSAGES_MODEL = "messages-model"
CHAT_BASE_PATH = "/compatible-mode/v1"


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
)
class TestEngineListenerAPIs(MockUpstreamMixin, ServeSessionMixin, CLITestBase):
    """Envoy-mode listener apis read the client's path, not the provider's."""

    def test_listener_apis_read_the_path_the_client_sent(self):
        self.print_test_header(
            "Engine Listener APIs Integration Test",
            "A messages-only listener serves Messages clients and refuses Chat clients",
        )
        mock_container = f"{self.runtime_stack.stack_name}-listener-apis-upstream"
        self._write_config(mock_container)

        serve_process = self._start_serve_background()
        try:
            self._wait_for_serve_success(serve_process)
            self.assertTrue(
                self.wait_for_health(
                    port=self.runtime_stack.api_port,
                    timeout=self.HEALTH_CHECK_TIMEOUT,
                ),
                "router API did not become healthy",
            )
            with self._running_mock_upstream(mock_container):
                # Retries until the stack answers; the Messages backend serves
                # a chat client on a listener without apis.
                self._send_mock_chat_completion(
                    mock_container, model=MESSAGES_MODEL, listener_port=OPEN_PORT
                )
                served = self._post(
                    CLAUDE_PORT,
                    "/v1/messages",
                    {
                        "model": CHAT_MODEL,
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": "ping"}],
                    },
                    {"anthropic-version": "2023-06-01"},
                )
                before_refusal = self._received_paths(mock_container)
                refused = self._post(
                    CLAUDE_PORT,
                    "/v1/chat/completions",
                    {
                        "model": MESSAGES_MODEL,
                        "messages": [{"role": "user", "content": "ping"}],
                    },
                    # Envoy replaces a client's claim to the open listener.
                    {"x-vsr-listener": "open"},
                )
                after_refusal = self._received_paths(mock_container)
        finally:
            self._stop_serve_process(serve_process)

        self.assertIn("/v1/messages", before_refusal)
        self.assertEqual(served[0], 200, served[1])
        self.assertIn(f"{CHAT_BASE_PATH}/chat/completions", before_refusal)
        self.assertEqual(refused[0], 404, refused[1])
        self.assertEqual(
            json.loads(refused[1]),
            {
                "detail": 'listener "claude" does not serve /v1/chat/completions; '
                "it serves messages"
            },
        )
        self.assertEqual(
            after_refusal,
            before_refusal,
            "a refused request reached the backend",
        )
        self.print_test_result(True, "listener apis read the path the client sent")

    def _write_config(self, mock_container: str) -> None:
        upstream = f"http://{mock_container}:{PROVIDER_MOCKER_PORT}"
        config_path = Path(
            self.write_minimal_canonical_config(
                port=CLAUDE_PORT,
                model_name=CHAT_MODEL,
                endpoint=upstream + CHAT_BASE_PATH,
                provider="openai",
                api_only=True,
            )
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        claude = config["listeners"][0]
        claude["name"] = "claude"
        claude["apis"] = {"messages": {}}
        config["listeners"].append(
            {"name": "open", "address": "0.0.0.0", "port": OPEN_PORT, "timeout": "60s"}
        )
        config["providers"]["models"].append(
            {
                "name": MESSAGES_MODEL,
                "provider_model_id": MESSAGES_MODEL,
                "api_format": "messages",
                "backend_refs": [
                    {
                        "name": "primary",
                        "weight": 100,
                        "endpoint": upstream,
                        "provider": "anthropic",
                    }
                ],
            }
        )
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

    def _post(
        self,
        port: int,
        path: str,
        body: dict,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        """Send one request through a listener and return its status and body."""
        request = urllib_request.Request(
            f"http://localhost:{port + self.runtime_stack.port_offset}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=30) as response:
                return response.status, response.read().decode("utf-8", "replace")
        except urllib_error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")

    def _received_paths(self, mock_container: str) -> list[str]:
        """The request paths the mock upstream logged, in order."""
        logs = self._run_subprocess(
            [self.container_runtime, "logs", mock_container], timeout=10
        )
        self.assertEqual(logs.returncode, 0, logs.stderr)
        return [
            line.strip() for line in logs.stdout.splitlines() if line.startswith("/")
        ]


if __name__ == "__main__":
    unittest.main()
