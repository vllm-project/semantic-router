#!/usr/bin/env python3
"""A change the running Router must restart for waits for the next serve (#7222).

`vllm-sr config apply` saves such a change beside the runtime config the
Router watches. Until the next `vllm-sr serve`, `status` reports it, while
`config get` and requests keep following the active config; that serve then
activates the saved document. A change the Router hot-reloads applies at once
and saves nothing.
"""

import json
import os
import unittest
from pathlib import Path

import yaml
from cli.commands.runtime_paths import _runtime_config_filename, pending_restart_path
from cli_test_base import CLITestBase
from mock_upstream import PROVIDER_MOCKER_PORT, MockUpstreamMixin
from serve_session import ServeSessionMixin

RESTART_LINE = "a change saved with `vllm-sr config apply` needs `vllm-sr serve`"
SAMPLING = {"type": "probabilistic", "rate": 0.25}


def _tracing(config: dict) -> dict:
    return (
        config.get("global", {})
        .get("services", {})
        .get("observability", {})
        .get("tracing", {})
    )


class PendingRestartChecks:
    """Serve, apply, and check that a refused change waits for the next serve."""

    def _runtime_config(self) -> Path:
        return (
            Path(self.test_dir)
            / ".vllm-sr"
            / _runtime_config_filename(self.runtime_stack.stack_name)
        )

    def _edit_config(self, edit) -> None:
        path = Path(self.test_dir) / "config.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        edit(document)
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    def _serve(self) -> None:
        process = self._start_serve_background(
            env=os.environ.copy(), arguments=("--config", "config.yaml")
        )
        try:
            self._wait_for_serve_success(process)
        finally:
            self._stop_serve_process(process)

    def _cli_json(self, *args: str) -> dict:
        code, stdout, stderr = self.run_cli(list(args), timeout=240)
        self.assertEqual(code, 0, f"{args}: {stdout}\n{stderr}")
        return json.loads(stdout)

    def _active_config(self) -> dict:
        return self._cli_json("config", "get", "--format", "json")["config"]

    def _status(self) -> str:
        code, stdout, stderr = self.run_cli(["status"], timeout=60)
        self.assertEqual(code, 0, f"{stdout}\n{stderr}")
        return stdout

    def _requests_served_by(self, mock_container: str) -> int:
        logs = self._run_subprocess(
            [self.container_runtime, "logs", mock_container], timeout=10
        )
        self.assertEqual(logs.returncode, 0, logs.stderr)
        return sum(
            1
            for line in logs.stdout.splitlines()
            if line.startswith("/") and line.strip().endswith("/chat/completions")
        )

    def _apply_refused_for_restart(self) -> None:
        """Apply a change the Router refuses; check it waits outside the watch."""

        active_before = self._runtime_config().read_bytes()
        saved = self._cli_json("config", "apply", "--config", "config.yaml")
        self.assertEqual(saved["status"], "restart_required", saved)
        self.assertFalse(saved["applied"], saved)
        self.assertEqual(
            Path(saved["saved"]), pending_restart_path(self._runtime_config()), saved
        )
        self.assertTrue(Path(saved["saved"]).is_file(), saved)
        # The Router reloads only its runtime config file, which stays as it was.
        self.assertEqual(self._runtime_config().read_bytes(), active_before)
        self.assertIn(RESTART_LINE, self._status())

    def _assert_served_and_spent(self) -> None:
        self.assertNotIn("Restart required", self._status())
        self.assertFalse(pending_restart_path(self._runtime_config()).exists())


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
)
class TestPendingRestartStandalone(
    PendingRestartChecks, MockUpstreamMixin, ServeSessionMixin, CLITestBase
):
    """Standalone: a tracing change waits; a hot-reloadable change does not."""

    SERVE_GATEWAY = "standalone"

    def test_a_tracing_change_waits_for_serve(self):
        self.print_test_header(
            "Pending restart, standalone tracing",
            "A hot change applies at once; a tracing change waits for serve",
        )
        mock_container = f"{self.runtime_stack.stack_name}-pending-tracing-upstream"
        self.write_minimal_canonical_config(
            provider="openai",
            endpoint=f"http://{mock_container}:{PROVIDER_MOCKER_PORT}/v1",
        )
        self._serve()
        try:
            with self._running_mock_upstream(mock_container):
                self._send_mock_chat_completion(mock_container)

                # Control: the Router takes this change without a restart.
                def describe(document):
                    document["recipes"][0]["decisions"][0]["description"] = "Hot"

                self._edit_config(describe)
                applied = self._cli_json("config", "apply", "--config", "config.yaml")
                self.assertTrue(applied["applied"], applied)
                active = self._active_config()
                self.assertEqual(
                    active["recipes"][0]["decisions"][0]["description"], "Hot"
                )
                self._assert_served_and_spent()

                tracing_before = _tracing(active)
                self.assertNotEqual(tracing_before.get("sampling"), SAMPLING)

                def sample(document):
                    services = document.setdefault("global", {}).setdefault(
                        "services", {}
                    )
                    services.setdefault("observability", {}).setdefault("tracing", {})[
                        "sampling"
                    ] = SAMPLING

                self._edit_config(sample)
                self._apply_refused_for_restart()
                # The tracer the Router started with is the one it serves.
                self.assertEqual(_tracing(self._active_config()), tracing_before)
                self._send_mock_chat_completion(mock_container)

                self._serve()
                self.assertEqual(_tracing(self._active_config())["sampling"], SAMPLING)
                self._assert_served_and_spent()
                self._send_mock_chat_completion(mock_container)
        finally:
            self.run_cli(["stop"], timeout=120)
        self.print_test_result(True, "the tracing change waited for serve")


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
)
class TestPendingRestartExtproc(
    PendingRestartChecks, MockUpstreamMixin, ServeSessionMixin, CLITestBase
):
    """ExtProc: a backend change waits for the serve that renders Envoy again."""

    SERVE_GATEWAY = "extproc"

    def test_a_backend_change_waits_for_serve(self):
        self.print_test_header(
            "Pending restart, extproc backend",
            "Requests reach the old backend until serve renders Envoy again",
        )
        old = f"{self.runtime_stack.stack_name}-pending-old-upstream"
        new = f"{self.runtime_stack.stack_name}-pending-new-upstream"
        self.write_minimal_canonical_config(
            provider="openai",
            endpoint=f"http://{old}:{PROVIDER_MOCKER_PORT}/v1",
        )

        def endpoint(active):
            return active["providers"]["models"][0]["backend_refs"][0]["endpoint"]

        self._serve()
        try:
            with self._running_mock_upstream(old), self._running_mock_upstream(new):
                self._send_mock_chat_completion(old)
                served_before = endpoint(self._active_config())
                self.assertIn(old, served_before)

                def move(document):
                    backend = document["providers"]["models"][0]["backend_refs"][0]
                    backend["endpoint"] = f"http://{new}:{PROVIDER_MOCKER_PORT}/v1"

                self._edit_config(move)
                self._apply_refused_for_restart()
                self.assertEqual(endpoint(self._active_config()), served_before)
                at_old = self._requests_served_by(old)
                self._send_mock_chat_completion(old)
                self.assertEqual(self._requests_served_by(old), at_old + 1)

                self._serve()
                self.assertIn(new, endpoint(self._active_config()))
                self._assert_served_and_spent()
                at_new = self._requests_served_by(new)
                self._send_mock_chat_completion(new)
                self.assertEqual(self._requests_served_by(new), at_new + 1)
        finally:
            self.run_cli(["stop"], timeout=120)
        self.print_test_result(True, "the backend change waited for serve")


if __name__ == "__main__":
    unittest.main()
