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

import yaml
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
                    document["routing"]["decisions"][0]["description"] = "Applied"

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
                    active["routing"]["decisions"][0]["description"], "Applied"
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

        def image_generation(condition):
            def edit(document):
                routing = document["routing"]
                routing["signals"] = {
                    "modality": [
                        {"name": "AR", "description": "Text requests."},
                        {"name": "BOTH", "description": "Text and image requests."},
                    ]
                }
                routing["decisions"].insert(
                    0,
                    {
                        "name": "image-gen",
                        "description": "Image generation",
                        "priority": 200,
                        "rules": {
                            "operator": "AND",
                            "conditions": [{"type": "modality", "name": condition}],
                        },
                        "modelRefs": [{"model": "test-model", "use_reasoning": False}],
                    },
                )

            return edit

        self._edit_config(image_generation("BOTH"))
        code, stdout, stderr = self.run_cli(
            ["config", "validate", "--config", "config.yaml"], timeout=240
        )
        self.assertEqual(code, 1, f"{stdout}\n{stderr}")
        self.assertIn('uses modality condition "BOTH"', stderr)
        self.assertNotIn("Configuration is valid", stdout)

        self.write_minimal_canonical_config()
        self._edit_config(image_generation("AR"))
        code, stdout, stderr = self.run_cli(
            ["config", "validate", "--config", "config.yaml"], timeout=240
        )
        self.assertEqual(code, 0, f"{stdout}\n{stderr}")
        self.assertIn("Checked by the Router in", stdout)
        self.assertIn("the modality signal never matches", stderr)
        self.print_test_result(True, "validate gave the Router's verdict and warnings")


if __name__ == "__main__":
    unittest.main()
