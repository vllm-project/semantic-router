#!/usr/bin/env python3
"""Real-daemon regression: a real Grafana container must boot and become ready
when started exactly like ``container_start_grafana`` with the CLI's mounted
password file and the rendered production ``grafana.serve.ini``.
"""

import base64
import os
import stat
import time
import unittest
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from cli import grafana_credentials as gc
from cli.container_observability import render_observability_template
from cli_test_base import (
    HTTP_STATUS_OK,
    CLITestBase,
    stack_scoped_test_container_name,
)

# Keep in sync with `container_start_grafana` in
# src/vllm-sr/cli/container_support_services.py.
GRAFANA_IMAGE = "docker.io/grafana/grafana:11.5.1"
GRAFANA_SERVE_INI_TEMPLATE = (
    Path(gc.__file__).resolve().parent / "templates" / "grafana.serve.ini"
)
CONTAINER_GRAFANA_INI_PATH = "/etc/grafana/grafana.ini"
# Org-admin API: anonymous Viewer is 403, anonymous disabled is 401, Admin is 200.
GRAFANA_ORG_ADMIN_API = "/api/org/users"
HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_FORBIDDEN = 403

GRAFANA_CONTAINER_SUFFIX = "vllm-sr-cli-test-grafana"
GRAFANA_READY_TIMEOUT = 180
GRAFANA_START_COMMAND_TIMEOUT = 600

integration_only = unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
)


class TestGrafanaPasswordFileContainer(CLITestBase):
    """A real Grafana container must boot with the CLI's mounted password file."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.GRAFANA_CONTAINER_NAME = stack_scoped_test_container_name(
            cls.runtime_stack.stack_name, GRAFANA_CONTAINER_SUFFIX
        )

    def tearDown(self):
        self._run_subprocess(
            [self.container_runtime, "rm", "-f", self.GRAFANA_CONTAINER_NAME],
            timeout=30,
        )
        super().tearDown()

    @integration_only
    def test_grafana_boots_ready_with_the_mounted_password_file(self):
        """Boot Grafana like serve: mounted secret plus rendered production ini.

        Grafana runs as an unprivileged uid, so the file must be
        container-readable (0644) rather than owner-only for the container to
        reach readiness. The production ini must be mounted so anonymous
        Admin cannot bypass the password-file credential.
        """
        self.print_test_header(
            "grafana password file container startup",
            "a real Grafana container becomes ready from the mounted secret "
            "and production ini",
        )

        saved_override = os.environ.pop(gc.GRAFANA_ADMIN_PASSWORD_ENV, None)
        try:
            password_file = gc.ensure_grafana_admin_password_file(self.test_dir)
            password = password_file.read_text(encoding="utf-8")
        finally:
            if saved_override is not None:
                os.environ[gc.GRAFANA_ADMIN_PASSWORD_ENV] = saved_override

        grafana_ini = Path(self.test_dir) / "grafana.serve.ini"
        grafana_ini.write_text(
            render_observability_template(
                GRAFANA_SERVE_INI_TEMPLATE.read_text(encoding="utf-8"),
                self.runtime_stack,
            ),
            encoding="utf-8",
        )

        self.assertEqual(
            stat.S_IMODE(password_file.stat().st_mode),
            0o644,
            "the mounted secret must be readable by the unprivileged Grafana uid",
        )
        self.assertEqual(
            stat.S_IMODE(password_file.parent.stat().st_mode),
            0o700,
            "the state directory holding the secret must stay owner-only",
        )

        result = self._run_subprocess(
            [
                self.container_runtime,
                "run",
                "-d",
                "--name",
                self.GRAFANA_CONTAINER_NAME,
                "-e",
                f"{gc.GRAFANA_ADMIN_PASSWORD_FILE_ENV}="
                f"{gc.CONTAINER_GRAFANA_PASSWORD_PATH}",
                "-v",
                (f"{password_file}:{gc.CONTAINER_GRAFANA_PASSWORD_PATH}:ro,z"),
                "-v",
                f"{grafana_ini}:{CONTAINER_GRAFANA_INI_PATH}:ro",
                "-p",
                "127.0.0.1::3000",
                GRAFANA_IMAGE,
            ],
            timeout=GRAFANA_START_COMMAND_TIMEOUT,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"the Grafana container failed to start: {result.stderr}",
        )
        try:
            self.assertTrue(
                self.wait_for_container_running(
                    timeout=120,
                    container_name=self.GRAFANA_CONTAINER_NAME,
                ),
                "the Grafana container exited instead of becoming ready; the "
                "bind-mounted password file may be unreadable by the "
                "unprivileged Grafana uid",
            )
            host_port = self._published_host_port()
            self.assertTrue(
                self._wait_until_grafana_ready(host_port),
                "the Grafana container never became ready; inspect the "
                "container logs for a password-file read failure",
            )

            # Prove the mounted production ini denies unauthenticated admin
            # access, and that the generated password still authenticates.
            # With anonymous Viewer, a bad Basic credential falls through to
            # Viewer (403) instead of 401; neither status is org-admin.
            self.assertIn(
                self._grafana_api_status(
                    GRAFANA_ORG_ADMIN_API, host_port, password=None
                ),
                (HTTP_STATUS_UNAUTHORIZED, HTTP_STATUS_FORBIDDEN),
                "unauthenticated callers must not receive org-admin access",
            )
            self.assertIn(
                self._grafana_api_status(
                    GRAFANA_ORG_ADMIN_API, host_port, "definitely-not-the-password"
                ),
                (HTTP_STATUS_UNAUTHORIZED, HTTP_STATUS_FORBIDDEN),
                "an unknown admin password must not receive org-admin access",
            )
            self.assertEqual(
                self._grafana_api_status(GRAFANA_ORG_ADMIN_API, host_port, password),
                HTTP_STATUS_OK,
                "the generated admin password from the mounted file was "
                "rejected by a ready Grafana",
            )
        finally:
            self._run_subprocess(
                [self.container_runtime, "rm", "-f", self.GRAFANA_CONTAINER_NAME],
                timeout=30,
            )
        self.print_test_result(
            True,
            "production ini denied unauthenticated admin; password-file admin worked",
        )

    def _published_host_port(self) -> str:
        """Resolve the host port Docker assigned to the container's 3000."""
        result = self._run_subprocess(
            [
                self.container_runtime,
                "port",
                self.GRAFANA_CONTAINER_NAME,
                "3000",
            ],
            timeout=10,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"could not resolve the published Grafana port: {result.stderr}",
        )
        first_binding = result.stdout.strip().splitlines()[0]
        return first_binding.rsplit(":", 1)[-1]

    def _wait_until_grafana_ready(self, host_port: str) -> bool:
        """Poll the unauthenticated health endpoint until Grafana answers."""
        deadline = time.time() + GRAFANA_READY_TIMEOUT
        while time.time() < deadline:
            if (
                self._grafana_api_status("/api/health", host_port, password=None)
                == HTTP_STATUS_OK
            ):
                return True
            time.sleep(2)
        return False

    def _grafana_api_status(
        self, api_path: str, host_port: str, password: str | None
    ) -> int | None:
        """Return the HTTP status of *api_path*, or ``None`` when unreachable."""
        url = f"http://127.0.0.1:{host_port}{api_path}"
        headers = {}
        if password is not None:
            credentials = base64.b64encode(
                f"{gc.GRAFANA_ADMIN_USER}:{password}".encode()
            ).decode("ascii")
            headers["Authorization"] = f"Basic {credentials}"
        request = urllib_request.Request(url, headers=headers)
        try:
            with urllib_request.urlopen(request, timeout=5) as response:
                return response.status
        except urllib_error.HTTPError as error:
            return error.code
        except Exception:
            return None
