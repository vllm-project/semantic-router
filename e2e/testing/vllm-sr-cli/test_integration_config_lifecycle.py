#!/usr/bin/env python3
"""Configuration changes made from the CLI on a standalone local stack.

`vllm-sr config apply` sends what `vllm-sr serve` would run, so after an
apply the next serve keeps the CLI's access to the management API (#4698). A
change the running Router can't take is saved for the next serve, as the
Dashboard saves one (#4699). `vllm-sr config validate` runs the Router's own
validation from the stack's image, and prints the warnings the Router logs
(#4696, #4697).
"""

import json
import os
import unittest
from pathlib import Path

import yaml
from cli.pending_activation import ORIGIN_CLI, read_pending_activation
from cli_test_base import CLITestBase
from mock_upstream import PROVIDER_MOCKER_PORT, MockUpstreamMixin
from serve_session import ServeSessionMixin


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
)
class TestConfigLifecycle(MockUpstreamMixin, ServeSessionMixin, CLITestBase):
    """Apply, serve, restart-required and validate, all from the CLI."""

    SERVE_GATEWAY = "standalone"

    def _config_path(self) -> str:
        return os.path.join(self.test_dir, "config.yaml")

    def _edit_config(self, edit) -> None:
        with open(self._config_path(), encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        edit(document)
        with open(self._config_path(), "w", encoding="utf-8") as handle:
            yaml.safe_dump(document, handle, sort_keys=False)

    def _serve(self) -> str:
        process = self._start_serve_background(
            env=os.environ.copy(), arguments=("--config", "config.yaml")
        )
        try:
            stdout, stderr = self._wait_for_serve_success(process)
        finally:
            self._stop_serve_process(process)
        return stdout + stderr

    def _cli_json(self, *args: str) -> dict:
        code, stdout, stderr = self.run_cli(list(args), timeout=240)
        self.assertEqual(code, 0, f"{args}: {stdout}\n{stderr}")
        return json.loads(stdout)

    def _write_config(self, mock_container: str) -> None:
        self.write_minimal_canonical_config(
            provider="openai",
            base_url=f"http://{mock_container}:{PROVIDER_MOCKER_PORT}/v1",
        )

    def test_status_uses_the_active_management_port(self):
        self.print_test_header(
            "Status with a custom management port",
            "Router health keeps its startup port until a pending change restarts",
        )
        mock_container = f"{self.runtime_stack.stack_name}-status-port-upstream"
        self._write_config(mock_container)

        def set_management_port(document, port):
            management = (
                document.setdefault("global", {})
                .setdefault("services", {})
                .setdefault("management_api", {})
            )
            management["bind_address"] = "0.0.0.0"
            management["port"] = port

        self._edit_config(lambda document: set_management_port(document, 8080))
        try:
            self._serve()
            self._edit_config(lambda document: set_management_port(document, 8082))

            code, stdout, stderr = self.run_cli(["status", "router"], timeout=60)
            self.assertEqual(code, 0, f"{stdout}\n{stderr}")
            self.assertRegex(stdout, r"Router\s+Running")
            self.assertNotIn("Status unknown", stdout)

            def move_listener(document):
                document["listeners"][0]["port"] = 8890

            self._edit_config(move_listener)
            # Keep this status regression on the default local apply path;
            # the custom-endpoint workflow has its own lifecycle coverage.
            saved = self._cli_json(
                "config",
                "apply",
                "--config",
                "config.yaml",
            )
            self.assertEqual(saved["status"], "restart_required", saved)

            code, stdout, stderr = self.run_cli(["status", "router"], timeout=60)
            self.assertEqual(code, 0, f"{stdout}\n{stderr}")
            self.assertRegex(stdout, r"Router\s+Running")
            self.assertNotIn("Status unknown", stdout)

            self._serve()
            self._edit_config(lambda document: set_management_port(document, 8083))
            code, stdout, stderr = self.run_cli(["status", "router"], timeout=60)
            self.assertEqual(code, 0, f"{stdout}\n{stderr}")
            self.assertRegex(stdout, r"Router\s+Running")
            self.assertNotIn("Status unknown", stdout)
        finally:
            self.run_cli(["stop"], timeout=120)
        self.print_test_result(True, "status followed the running management port")

    def test_apply_to_custom_management_endpoint_saves_pending_restart(self):
        self.print_test_header(
            "config apply through a custom management endpoint",
            "Local pending changes survive explicit endpoints and later config edits",
        )
        mock_container = f"{self.runtime_stack.stack_name}-custom-apply-upstream"
        self._write_config(mock_container)

        def set_management_port(document, port):
            management = (
                document.setdefault("global", {})
                .setdefault("services", {})
                .setdefault("management_api", {})
            )
            management["bind_address"] = "0.0.0.0"
            management["port"] = port

        def move_listeners(document):
            set_management_port(document, 8082)
            document["listeners"][0]["port"] = 8890

        def endpoint(port):
            host_port = self.runtime_stack.host_port(
                port, name="management API host port"
            )
            return f"http://127.0.0.1:{host_port}"

        self._edit_config(lambda document: set_management_port(document, 8081))
        try:
            self._serve()
            running_endpoint = endpoint(8081)
            self._edit_config(move_listeners)

            # The second apply still targets the bound port after the first
            # one has saved a future port into the mounted runtime config.
            for attempt in range(2):
                if attempt:
                    self._edit_config(
                        lambda document: document["listeners"][0].update(port=8891)
                    )
                apply_endpoint = running_endpoint + ("/api/v1" if attempt else "")
                saved = self._cli_json(
                    "config",
                    "apply",
                    "--config",
                    "config.yaml",
                    "--endpoint",
                    apply_endpoint,
                )
                self.assertEqual(saved["status"], "restart_required", saved)
                self.assertFalse(saved["applied"], saved)
                self.assertTrue(os.path.isfile(saved["saved"]), saved)
                active = self._cli_json(
                    "config", "get", "--format", "json", "--endpoint", running_endpoint
                )["config"]
                # Config reads can expose the saved candidate. The successful
                # request proves the old endpoint still serves while it is pending.
                self.assertIsInstance(active, dict)
                code, status, stderr = self.run_cli(["status"], timeout=60)
                self.assertEqual(code, 0, f"{status}\n{stderr}")
                self.assertRegex(status, r"Router\s+Running")
                pending = read_pending_activation(Path(saved["saved"]))
                self.assertIsNotNone(pending, saved)
                self.assertEqual(pending.reason, "restart")
                self.assertEqual(pending.origin, ORIGIN_CLI)

            self._serve()
            active = self._cli_json(
                "config", "get", "--format", "json", "--endpoint", endpoint(8082)
            )["config"]
            self.assertEqual(
                active["global"]["services"]["management_api"]["port"], 8082
            )
            self.assertEqual(active["listeners"][0]["port"], 8891)
        finally:
            self.run_cli(["stop"], timeout=120)
        self.print_test_result(
            True, "custom endpoint saved and activated local changes"
        )

    def test_apply_then_serve_keeps_the_management_api(self):
        self.print_test_header(
            "config apply, then serve",
            "The management API stays reachable after apply and a later serve",
        )
        mock_container = f"{self.runtime_stack.stack_name}-lifecycle-upstream"
        self._write_config(mock_container)
        self._serve()
        try:
            with self._running_mock_upstream(mock_container):
                self._send_mock_chat_completion(mock_container)

                def describe(document):
                    document["recipes"][0]["decisions"][0]["description"] = "Applied"

                self._edit_config(describe)
                applied = self._cli_json("config", "apply", "--config", "config.yaml")
                self.assertTrue(applied["applied"], applied)
                self._cli_json("config", "versions")

                output = self._serve()
                self.assertNotIn("Preserving Dashboard or package changes", output)
                # Before #4698 this failed: the Router's management API bound
                # 127.0.0.1 inside its container after this serve.
                self._cli_json("config", "versions")
                active = self._cli_json("config", "get", "--format", "json")["config"]
                self.assertEqual(
                    active["recipes"][0]["decisions"][0]["description"], "Applied"
                )
                self._send_mock_chat_completion(mock_container)
        finally:
            self.run_cli(["stop"], timeout=120)
        self.print_test_result(True, "apply and serve kept the management API")

    def test_a_restart_required_apply_waits_for_serve(self):
        self.print_test_header(
            "Restart required from the CLI",
            "config apply saves a listener move; the next serve applies it",
        )
        mock_container = f"{self.runtime_stack.stack_name}-restart-cli-upstream"
        self._write_config(mock_container)
        self._serve()
        try:
            with self._running_mock_upstream(mock_container):
                self._send_mock_chat_completion(mock_container)

                def move(document):
                    document["listeners"][0]["port"] = 8890

                self._edit_config(move)
                code, stdout, stderr = self.run_cli(
                    ["config", "apply", "--config", "config.yaml"], timeout=240
                )
                self.assertEqual(code, 0, f"{stdout}\n{stderr}")
                saved = json.loads(stdout)
                self.assertEqual(saved["status"], "restart_required", saved)
                self.assertNotIn("Envoy", saved["reason"])
                self.assertIn("Restart required: run `vllm-sr serve` to apply.", stderr)

                code, status, _stderr = self.run_cli(["status"], timeout=60)
                self.assertEqual(code, 0, status)
                self.assertIn(
                    "a change saved with `vllm-sr config apply` needs `vllm-sr serve`",
                    status,
                )
                # The running Router keeps the listener it bound at startup.
                self._send_mock_chat_completion(mock_container)

                self._serve()
                headers = self._send_mock_chat_completion(
                    mock_container, listener_port=8890
                )
                self.assertEqual(headers.get("x-vsr-selected-model"), "test-model")
                code, status, _stderr = self.run_cli(["status"], timeout=60)
                self.assertNotIn("Restart required", status)
                self._cli_json("config", "versions")
        finally:
            self.run_cli(["stop"], timeout=120)
        self.print_test_result(True, "the next serve applied the change apply saved")

    def test_validate_runs_the_routers_own_validation(self):
        self.print_test_header(
            "config validate",
            "Refuses what the Router refuses and prints the Router's warnings",
        )
        self.write_minimal_canonical_config()

        # Only the Router knows the headers it owns and which models have a
        # calibrated threshold, so both verdicts below are the Router's.
        def guarded(header):
            def edit(document):
                document["global"] = {
                    "model_catalog": {
                        "deployments": {
                            "custom-judge": {
                                "provider": "model_runtime",
                                "artifact": "acme/custom-judge",
                                "revision": "0123456789abcdef0123456789abcdef01234567",
                            }
                        }
                    }
                }
                routing = document["recipes"][0]
                routing["signals"] = {
                    "jailbreak": [{"name": "attack", "deployment": "custom-judge"}]
                }
                routing["decisions"].insert(
                    0,
                    {
                        "name": "guarded",
                        "description": "Requests that attack the assistant",
                        "priority": 200,
                        "rules": {
                            "operator": "AND",
                            "conditions": [{"type": "jailbreak", "name": "attack"}],
                        },
                        "modelRefs": [{"model": "test-model", "use_reasoning": False}],
                        "plugins": [
                            {
                                "type": "header_mutation",
                                "configuration": {
                                    "add": [{"name": header, "value": "guarded"}]
                                },
                            }
                        ],
                    },
                )

            return edit

        self._edit_config(guarded("Authorization"))
        code, stdout, stderr = self.run_cli(
            ["config", "validate", "--config", "config.yaml"], timeout=240
        )
        self.assertEqual(code, 1, f"{stdout}\n{stderr}")
        self.assertIn(
            "Authorization is owned by the Router or Envoy", " ".join(stderr.split())
        )
        self.assertNotIn("Configuration is valid", stdout)

        self.write_minimal_canonical_config()
        self._edit_config(guarded("X-Guarded"))
        code, stdout, stderr = self.run_cli(
            ["config", "validate", "--config", "config.yaml"], timeout=240
        )
        self.assertEqual(code, 0, f"{stdout}\n{stderr}")
        self.assertIn("Checked by the Router in", stdout)
        self.assertIn(
            "has no calibrated threshold for the jailbreak preset",
            " ".join(stderr.split()),
        )
        self.print_test_result(True, "validate gave the Router's verdict and warnings")


if __name__ == "__main__":
    unittest.main()
