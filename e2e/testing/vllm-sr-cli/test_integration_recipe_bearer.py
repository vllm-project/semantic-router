#!/usr/bin/env python3
"""`vllm-sr serve` activates a Recipe with bearer management auth.

The Dashboard imports the fixture Recipe (fixtures/bearer-recipe) over HTTPS,
through Recipe import's real path: HTTPS only, the archive digest checked, and
the destination held to the Dashboard's public-address policy. A test-only CA
signs the server's certificate and only a copy of the Dashboard image built
here trusts it. The server runs in the Dashboard container's network namespace
on a public address bound to that namespace's loopback, so the policy holds
while the connection never leaves the container.

Bearer authentication needs the containers created anew, so the activation
waits for `vllm-sr serve`, run as the suite's user. CI runs the suite
unprivileged, and a non-root serve reads the Recipe store through the share
the Dashboard gives that user's group. Serve hands the stack's management
credential, which it owns, to the Dashboard and the Router by name. The test
checks the management API with and without the credential,
a chat through the Recipe's decision, that another serve keeps the package
active, and that no file but the CLI's owner-only state holds the credential.
"""

import json
import os
import secrets
import shutil
import stat
import subprocess
import tempfile
import time
import unittest
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from cli.commands.runtime_paths import _runtime_config_filename
from cli.consts import VLLM_SR_DASHBOARD_CONTAINER_IMAGE_DEFAULT
from cli.management_credential import management_credential_path
from cli.recipe_package import pack_recipe
from cli.recipe_topology_contract import MANAGEMENT_CREDENTIAL_ENV
from cli_test_base import CLITestBase
from mock_upstream import (
    DEFAULT_PROVIDER_MOCKER_IMAGE,
    PROVIDER_MOCKER_IMAGE_ENV,
    PROVIDER_MOCKER_PORT,
    MockUpstreamMixin,
)
from serve_session import ServeSessionMixin, startup_diagnostics

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bearer-recipe"
# The fixture's config.yaml routes to this host on the stack network.
UPSTREAM = "vllm-sr-recipe-fixture-upstream"
# A public address. The test binds it to the loopback of the Dashboard's
# network namespace only.
PACKAGE_HOST = "11.73.0.10"
PACKAGE_PORT = 8443
ADDRESS_IMAGE = "docker.io/library/alpine:3.20"
PACKAGE_SERVER = r"""
import functools, http.server, ssl, sys
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory="/package")
server = http.server.ThreadingHTTPServer((sys.argv[1], int(sys.argv[2])), handler)
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain("/tls/server.crt", "/tls/server.key")
server.socket = context.wrap_socket(server.socket, server_side=True)
print("serving", flush=True)
server.serve_forever()
"""


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
)
class TestBearerRecipeActivation(MockUpstreamMixin, ServeSessionMixin, CLITestBase):
    """The CLI, not the Dashboard, holds the management credential."""

    SERVE_GATEWAY = "standalone"

    def setUp(self):
        super().setUp()
        self.work = Path(tempfile.mkdtemp(prefix="vllm-sr-recipe-fixture-"))
        # The package server runs as an unprivileged container user.
        self.work.chmod(0o755)
        self.addCleanup(shutil.rmtree, self.work, ignore_errors=True)
        self.admin = {
            "email": "admin@example.com",
            "name": "Admin",
            "password": f"Bearer-{secrets.token_hex(12)}",
        }
        self._run_subprocess([self.container_runtime, "rm", "-f", UPSTREAM], timeout=30)

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
            with urllib_request.urlopen(request, timeout=120) as response:
                return response.status, json.loads(response.read())
        except urllib_error.HTTPError as error:
            self.fail(f"{path}: HTTP {error.code}: {error.read().decode()[:800]}")

    def _serve(self, env) -> str:
        """Run one `vllm-sr serve` to completion and return what it printed."""
        process = self._start_serve_background(
            env=env, arguments=("--config", "config.yaml")
        )
        try:
            stdout, stderr = process.communicate(timeout=self.HEALTH_CHECK_TIMEOUT)
        except subprocess.TimeoutExpired:
            stdout, stderr = self._stop_serve_process(process)
            self.fail(
                "Serve did not complete startup:\n"
                + startup_diagnostics(stdout, stderr, process.returncode)
            )
        self.assertEqual(
            process.returncode,
            0,
            startup_diagnostics(stdout, stderr, process.returncode),
        )
        self._wait_for_dashboard()
        return stdout + stderr

    def _wait_for_dashboard(self):
        """Serve returns once the Router is ready; the Dashboard may still start."""
        url = f"{self.runtime_stack.dashboard_url}/api/setup/state"
        deadline = time.time() + 120
        while True:
            try:
                with urllib_request.urlopen(url, timeout=10) as response:
                    if response.status == HTTPStatus.OK:
                        return
            except (urllib_error.URLError, ConnectionError, TimeoutError):
                pass
            self.assertLess(time.time(), deadline, "the Dashboard did not answer")
            time.sleep(1)

    def _sign_in(self, register=False):
        path = "/api/auth/bootstrap/register" if register else "/api/auth/login"
        body = (
            self.admin
            if register
            else {
                "email": self.admin["email"],
                "password": self.admin["password"],
            }
        )
        return self._dashboard(path, body)[1]["token"]

    def _issue_certificates(self) -> Path:
        tls = self.work / "tls"
        tls.mkdir(mode=0o755)

        def openssl(*arguments):
            result = self._run_subprocess(
                ["openssl", *arguments], timeout=60, cwd=str(tls)
            )
            self.assertEqual(result.returncode, 0, result.stderr)

        openssl(
            "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
            "-subj", "/CN=vllm-sr Recipe test CA",
            "-addext", "basicConstraints=critical,CA:TRUE",
            "-addext", "keyUsage=critical,keyCertSign,cRLSign",
            "-keyout", "ca.key", "-out", "ca.crt",
        )  # fmt: skip
        openssl(
            "req", "-newkey", "rsa:2048", "-nodes", "-subj", f"/CN={PACKAGE_HOST}",
            "-keyout", "server.key", "-out", "server.csr",
        )  # fmt: skip
        (tls / "server.ext").write_text(
            f"subjectAltName=IP:{PACKAGE_HOST}\n"
            "extendedKeyUsage=serverAuth\nbasicConstraints=CA:FALSE\n",
            encoding="utf-8",
        )
        openssl(
            "x509", "-req", "-in", "server.csr", "-CA", "ca.crt", "-CAkey", "ca.key",
            "-CAcreateserial", "-days", "1", "-extfile", "server.ext",
            "-out", "server.crt",
        )  # fmt: skip
        # A throwaway test key, readable by the server's container user.
        for path in tls.iterdir():
            path.chmod(0o644)
        return tls

    def _dashboard_image_trusting(self, ca: Path) -> str:
        """A copy of the Dashboard image that trusts the test CA, and nothing else changed."""
        context = self.work / "image"
        context.mkdir()
        shutil.copy(ca, context / "recipe-test-ca.crt")
        (context / "Dockerfile").write_text(
            "ARG BASE\nFROM ${BASE}\n"
            "COPY recipe-test-ca.crt /usr/local/share/ca-certificates/vllm-sr-recipe-test-ca.crt\n"
            "RUN update-ca-certificates\n",
            encoding="utf-8",
        )
        base = (
            os.environ.get("VLLM_SR_DASHBOARD_IMAGE")
            or VLLM_SR_DASHBOARD_CONTAINER_IMAGE_DEFAULT
        )
        tag = f"vllm-sr-dashboard-recipe-ca:{self.runtime_stack.stack_name}"
        result = self._run_subprocess(
            [
                self.container_runtime,
                "build",
                "--build-arg",
                f"BASE={base}",
                "-t",
                tag,
                str(context),
            ],
            timeout=600,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.addCleanup(
            self._run_subprocess, [self.container_runtime, "rmi", "-f", tag], timeout=60
        )
        return tag

    @contextmanager
    def _package_server(self, archive: Path, tls: Path):
        """Serve the archive over HTTPS from the Dashboard's network namespace."""
        package = self.work / "package"
        package.mkdir(mode=0o755)
        served = package / archive.name
        shutil.copy(archive, served)
        served.chmod(0o644)
        namespace = f"container:{self.DASHBOARD_CONTAINER_NAME}"
        bound = self._run_subprocess(
            [
                self.container_runtime,
                "run",
                "--rm",
                "--network",
                namespace,
                "--cap-add",
                "NET_ADMIN",
                ADDRESS_IMAGE,
                "ip",
                "address",
                "add",
                f"{PACKAGE_HOST}/32",
                "dev",
                "lo",
            ],
            timeout=300,
        )
        self.assertEqual(bound.returncode, 0, bound.stderr)
        server = f"{self.runtime_stack.stack_name}-recipe-package-server"
        image = os.getenv(PROVIDER_MOCKER_IMAGE_ENV) or os.getenv(
            "PROVIDER_MOCKER_IMAGE", DEFAULT_PROVIDER_MOCKER_IMAGE
        )
        self._run_subprocess([self.container_runtime, "rm", "-f", server], timeout=30)
        started = self._run_subprocess(
            [
                self.container_runtime,
                "run",
                "-d",
                "--name",
                server,
                "--network",
                namespace,
                "-v",
                f"{package}:/package:ro,z",
                "-v",
                f"{tls}:/tls:ro,z",
                "--entrypoint",
                "python3",
                image,
                "-c",
                PACKAGE_SERVER,
                PACKAGE_HOST,
                str(PACKAGE_PORT),
            ],
            timeout=60,
        )
        self.assertEqual(started.returncode, 0, started.stderr)
        try:
            deadline = time.time() + 60
            while "serving" not in self._container_output(server):
                self.assertLess(time.time(), deadline, self._container_output(server))
                time.sleep(1)
            yield f"https://{PACKAGE_HOST}:{PACKAGE_PORT}/{archive.name}"
        finally:
            self._run_subprocess(
                [self.container_runtime, "rm", "-f", server], timeout=30
            )

    def _container_output(self, container_name):
        logs = self._run_subprocess(
            [self.container_runtime, "logs", container_name], timeout=10
        )
        return logs.stdout + logs.stderr

    def _stack_credential(self) -> str:
        path = management_credential_path(
            self.test_dir, stack_layout=self.runtime_stack
        )
        return json.loads(path.read_text(encoding="utf-8"))["token"]

    def _assert_the_router_requires(self, credential):
        ready = f"http://localhost:{self.runtime_stack.api_port}/ready"
        with self.assertRaises(urllib_error.HTTPError) as anonymous:
            urllib_request.urlopen(ready, timeout=10)
        self.assertEqual(anonymous.exception.code, 401)
        request = urllib_request.Request(
            ready, headers={"Authorization": f"Bearer {credential}"}
        )
        with urllib_request.urlopen(request, timeout=10) as response:
            self.assertEqual(response.status, 200)

    def _assert_the_recipe_routes(self):
        # "vllm-sr/auto" lets the Router decide; a named model would skip the decisions.
        headers = self._send_mock_chat_completion(
            UPSTREAM, model="vllm-sr/auto", request_headers={"x-vsr-debug": "true"}
        )
        self.assertEqual(
            headers.get("x-vsr-selected-decision"), "bearer_route", dict(headers)
        )

    def _assert_the_dashboard_serves(self, recipe_digest, token):
        """The Dashboard reaches the Router with the credential and shows the package."""
        _, status = self._dashboard("/api/status")
        services = {service["name"]: service for service in status["services"]}
        self.assertTrue(services["Router"]["healthy"], services)
        self.assertTrue(services["Routing access"]["healthy"], services)
        _, packages = self._dashboard("/api/recipe/packages", token=token)
        self.assertEqual(packages["active_digest"], recipe_digest, packages)
        _, logs = self._dashboard("/api/logs?component=router&lines=50", token=token)
        self.assertTrue(logs["supported"], logs)
        self.assertGreater(logs["count"], 0, logs)
        return json.dumps(logs)

    def _assert_no_file_holds(self, credential):
        state = management_credential_path(
            self.test_dir, stack_layout=self.runtime_stack
        )
        self.assertEqual(stat.S_IMODE(state.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(state.parent.stat().st_mode) & 0o077, 0)
        runtime_config = (
            Path(self.test_dir)
            / ".vllm-sr"
            / _runtime_config_filename(self.runtime_stack.stack_name)
        )
        # The Router's config binds the credential by name.
        self.assertIn(
            f"env: {MANAGEMENT_CREDENTIAL_ENV}", runtime_config.read_text("utf-8")
        )
        # As root in the Dashboard container, so the files the Dashboard keeps
        # private count too. The credential travels on stdin.
        found = subprocess.run(
            [
                self.container_runtime,
                "exec",
                "-i",
                "-u",
                "0",
                self.DASHBOARD_CONTAINER_NAME,
                "sh",
                "-c",
                'read -r credential; grep -rlF --exclude-dir=management-credential -- "$credential" /app/.vllm-sr /app/data || true',
            ],
            input=credential + "\n",
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(found.returncode, 0, found.stderr)
        self.assertEqual(found.stdout.strip(), "", "files hold the credential")

    def test_serve_activates_a_bearer_recipe_the_dashboard_imported(self):
        self.print_test_header(
            "Bearer Recipe",
            "Imports a bearer-auth Recipe over HTTPS; the suite user's serve activates it",
        )
        tls = self._issue_certificates()
        archive = pack_recipe(FIXTURE, self.work / "bearer-fixture.zip")
        env = os.environ.copy()
        env.pop(MANAGEMENT_CREDENTIAL_ENV, None)
        env["VLLM_SR_DASHBOARD_IMAGE"] = self._dashboard_image_trusting(tls / "ca.crt")
        self.write_minimal_canonical_config(
            provider="openai",
            base_url=f"http://{UPSTREAM}:{PROVIDER_MOCKER_PORT}/v1",
        )

        printed = self._serve(env)
        try:
            with self._running_mock_upstream(UPSTREAM):
                self._send_mock_chat_completion(UPSTREAM)
                self.assert_dashboard_holds_no_container_runtime()
                token = self._sign_in(register=True)
                with self._package_server(archive.path, tls) as url:
                    status, imported = self._dashboard(
                        "/api/recipe/import",
                        {"url": url, "expected_archive_sha256": archive.archive_sha256},
                        token,
                    )
                self.assertEqual(status, 201, imported)
                self.assertTrue(imported["archive_verified"], imported)
                self.assertEqual(imported["recipe_digest"], archive.recipe_digest)

                request = {"recipe_digest": archive.recipe_digest}
                _, plan = self._dashboard(
                    "/api/recipe/activate/preview", request, token
                )
                self.assertEqual(plan["mode"], "stack_recreation", plan)
                self.assertEqual(plan["management_auth"]["mode"], "bearer", plan)
                status, activated = self._dashboard(
                    "/api/recipe/activate",
                    {
                        **request,
                        "expected_plan_digest": plan["plan_digest"],
                        "confirm_stack_recreation": True,
                    },
                    token,
                )
                self.assertEqual(status, 202, activated)
                self.assertEqual(activated["status"], "restart_required", activated)

                # The suite's user, not the Dashboard, applies the activation.
                printed += self._serve(env)
                credential = self._stack_credential()
                self._assert_the_router_requires(credential)
                self._assert_the_recipe_routes()
                printed += self._assert_the_dashboard_serves(
                    archive.recipe_digest, self._sign_in()
                )
                self.assert_dashboard_holds_no_container_runtime()

                # Another serve keeps the package and the credential.
                printed += self._serve(env)
                self.assertEqual(self._stack_credential(), credential)
                self._assert_the_router_requires(credential)
                self._assert_the_recipe_routes()
                printed += self._assert_the_dashboard_serves(
                    archive.recipe_digest, self._sign_in()
                )
                self.assertNotIn(credential, printed)
                self._assert_no_file_holds(credential)
        finally:
            self.run_cli(["stop"], timeout=120)
        self.print_test_result(
            True, "serve activated the bearer Recipe the Dashboard imported"
        )


if __name__ == "__main__":
    unittest.main()
