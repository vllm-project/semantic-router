#!/usr/bin/env python3
"""Integration coverage for Grafana Live behind the Dashboard proxy."""

import base64
import http.client
import json
import os
import socket
import time
import unittest

from cli_test_base import CLITestBase
from serve_session import ServeSessionMixin

GRAFANA_LIVE_ALLOWED_ORIGINS_ENV = "GF_LIVE_ALLOWED_ORIGINS"
PUBLIC_DASHBOARD_ORIGIN = "https://dashboard.example.test"
TEST_ADMIN_EMAIL = "grafana-live@example.test"
TEST_ADMIN_PASSWORD = "GrafanaLiveTest!3471"
MAX_HANDSHAKE_RESPONSE_BYTES = 65536
MIN_STATUS_LINE_FIELDS = 2

integration_only = unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
)


class TestGrafanaLiveIntegration(ServeSessionMixin, CLITestBase):
    """Exercise the browser-facing WebSocket upgrade through the real stack."""

    @integration_only
    def test_configured_public_origin_can_open_grafana_live_websocket(self):
        with self._running_serve(
            env={
                GRAFANA_LIVE_ALLOWED_ORIGINS_ENV: PUBLIC_DASHBOARD_ORIGIN,
                "DASHBOARD_ADMIN_EMAIL": TEST_ADMIN_EMAIL,
                "DASHBOARD_ADMIN_PASSWORD": TEST_ADMIN_PASSWORD,
            }
        ):
            session_token = self._dashboard_session_token()
            allowed_status = self._wait_for_websocket_status(
                PUBLIC_DASHBOARD_ORIGIN,
                session_token=session_token,
                expected_status=101,
            )
            self.assertEqual(allowed_status, 101)
            self.assertEqual(
                self._websocket_status(
                    "https://untrusted.example.test", session_token=session_token
                ),
                403,
            )

    def _wait_for_websocket_status(
        self,
        origin: str,
        *,
        session_token: str,
        expected_status: int,
        timeout: int = 60,
    ) -> int | None:
        deadline = time.time() + timeout
        last_status = None
        while time.time() < deadline:
            try:
                last_status = self._websocket_status(
                    origin, session_token=session_token
                )
            except OSError:
                last_status = None
            if last_status == expected_status:
                return last_status
            time.sleep(2)
        return last_status

    def _dashboard_session_token(self) -> str:
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.runtime_stack.dashboard_port, timeout=10
        )
        try:
            connection.request(
                "POST",
                "/api/auth/login",
                body=json.dumps(
                    {"email": TEST_ADMIN_EMAIL, "password": TEST_ADMIN_PASSWORD}
                ),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            payload = response.read()
        finally:
            connection.close()
        self.assertEqual(response.status, 200, payload.decode(errors="replace"))
        return json.loads(payload)["token"]

    def _websocket_status(self, origin: str, *, session_token: str) -> int:
        port = self.runtime_stack.dashboard_port
        websocket_key = base64.b64encode(b"grafana-live-key").decode("ascii")
        request = (
            "GET /embedded/grafana/api/live/ws HTTP/1.1\r\n"
            f"Host: localhost:{port}\r\n"
            "Connection: Upgrade\r\n"
            "Upgrade: websocket\r\n"
            f"Origin: {origin}\r\n"
            f"Cookie: vsr_session={session_token}\r\n"
            f"Sec-WebSocket-Key: {websocket_key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        with socket.create_connection(("127.0.0.1", port), timeout=10) as connection:
            connection.sendall(request.encode("ascii"))
            response = bytearray()
            while (
                b"\r\n\r\n" not in response
                and len(response) < MAX_HANDSHAKE_RESPONSE_BYTES
            ):
                chunk = connection.recv(4096)
                if not chunk:
                    break
                response.extend(chunk)

        status_line = (
            bytes(response).partition(b"\r\n")[0].decode("ascii", errors="replace")
        )
        fields = status_line.split()
        if len(fields) < MIN_STATUS_LINE_FIELDS or not fields[1].isdigit():
            raise AssertionError(
                f"invalid WebSocket handshake response: {status_line!r}"
            )
        return int(fields[1])

    def tearDown(self):
        self.run_cli(["stop"], timeout=30)
        self._cleanup_container()
        super().tearDown()


if __name__ == "__main__":
    unittest.main()
