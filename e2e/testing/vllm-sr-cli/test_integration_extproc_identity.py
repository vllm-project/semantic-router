#!/usr/bin/env python3
"""Client identity headers on the Envoy listener `vllm-sr serve --gateway extproc` generates.

A role binding gives the platform-admins group its own decision. The first
listener trusts no identity source, so the identity headers a client sends are
removed before the Router reads them: with no identity, the role binding refuses
the request. The second listener sits behind an authenticating proxy
(`identity.trust_headers`), so the same headers select the bound decision.
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

UNTRUSTED_PORT = 8888
TRUSTED_PORT = 8889
ADMIN_ROUTE = "admin-route"
CLAIMED_ADMIN = {
    "x-authz-user-id": "mallory",
    "x-authz-user-groups": "platform-admins",
    "x-authz-tenant-id": "tenant-a",
}


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
)
class TestExtprocListenerIdentity(MockUpstreamMixin, ServeSessionMixin, CLITestBase):
    """Only a listener that trusts identity headers lets a request assert one."""

    def test_client_identity_counts_only_on_a_trusting_listener(self):
        self.print_test_header(
            "Extproc Listener Identity Integration Test",
            "Client identity headers are removed unless the listener trusts them",
        )
        mock_container = f"{self.runtime_stack.stack_name}-identity-upstream"
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
                trusted = self._send_mock_chat_completion(
                    mock_container,
                    request_headers=CLAIMED_ADMIN,
                    model="vllm-sr/auto",
                    listener_port=TRUSTED_PORT,
                )
                status, body = self._status_of(UNTRUSTED_PORT, CLAIMED_ADMIN)
        finally:
            self._stop_serve_process(serve_process)

        self.assertEqual(trusted.get("x-vsr-selected-decision"), ADMIN_ROUTE)
        self.assertEqual(
            status,
            403,
            f"a client that claimed the admin group on the untrusting listener got {status}: {body}",
        )
        self.print_test_result(True, "client identity counts only where it is trusted")

    def _write_config(self, mock_container: str) -> None:
        config_path = Path(
            self.write_minimal_canonical_config(
                port=UNTRUSTED_PORT,
                base_url=f"http://{mock_container}:{PROVIDER_MOCKER_PORT}",
                provider="openai",
                api_only=True,
            )
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["listeners"].append(
            {
                "name": "behind-proxy",
                "address": "0.0.0.0",
                "port": TRUSTED_PORT,
                "timeout": "60s",
                "identity": {"trust_headers": True},
            }
        )
        default_route = config["recipes"][0]["decisions"][0]
        config["recipes"][0]["signals"] = {
            "role_bindings": [
                {
                    "name": "admin",
                    "subjects": [{"kind": "Group", "name": "platform-admins"}],
                    "role": "admin",
                }
            ]
        }
        config["recipes"][0]["decisions"].insert(
            0,
            {
                "name": ADMIN_ROUTE,
                "priority": default_route["priority"] + 100,
                "rules": {
                    "operator": "AND",
                    "conditions": [{"type": "authz", "name": "admin"}],
                },
                "modelRefs": default_route["modelRefs"],
            },
        )
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

    def _status_of(self, port: int, headers: dict[str, str]) -> tuple[int, str]:
        """Send one chat request through a listener and return its status."""
        request = urllib_request.Request(
            f"http://localhost:{port + self.runtime_stack.port_offset}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "vllm-sr/auto",
                    "messages": [{"role": "user", "content": "ping"}],
                }
            ).encode(),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=30) as response:
                return response.status, response.read().decode("utf-8", "replace")
        except urllib_error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")


if __name__ == "__main__":
    unittest.main()
