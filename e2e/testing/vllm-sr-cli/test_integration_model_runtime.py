#!/usr/bin/env python3
"""`vllm-sr serve --config` with a model the Router runs in its own runtime.

The Quickstart's step 4: a decision signal on a `provider: model_runtime`
deployment, which the Router starts and asks inside its container. The test
takes the signal, its route and the deployment from the page and serves a tiny
random-weight decision package that the router image's `vllm-srun fixture`
writes into the models directory, so nothing is downloaded.
"""

import copy
import os
import re
import time
import unittest
from pathlib import Path

import yaml
from cli_test_base import CLITestBase
from mock_upstream import PROVIDER_MOCKER_PORT, MockUpstreamMixin
from runtime_http import write_fixture
from serve_session import ServeSessionMixin

QUICKSTART = (
    Path(__file__).resolve().parents[3] / "website/docs/model-runtime/quickstart.md"
)
YAML_BLOCK = re.compile(r"^```yaml\n(.*?)^```", re.M | re.S)
PACKAGE = "decision-fixture"
# The Router mounts the models directory next to config.yaml here.
CONTAINER_MODELS_DIR = "/app/models"
# Until the managed runtime has loaded the package and passed its self-check,
# the signal is unknown and requests take the default route.
ANSWER_TIMEOUT_SECONDS = 300


def quickstart_router_config() -> dict:
    """The router configuration the Quickstart tells readers to save."""
    page = QUICKSTART.read_text(encoding="utf-8")
    for block in YAML_BLOCK.findall(page):
        document = yaml.safe_load(block)
        if isinstance(document, dict) and "routing" in document:
            return document
    raise AssertionError(f"{QUICKSTART} has no router configuration")


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
)
class TestServeManagedModelRuntime(MockUpstreamMixin, ServeSessionMixin, CLITestBase):
    """The Router starts a runtime for its deployment and routes on its answers."""

    def test_quickstart_decision_signal_is_answered_by_the_managed_runtime(self):
        self.print_test_header(
            "Managed Model Runtime Integration Test",
            "Serves the Quickstart's decision signal from a runtime the Router starts",
        )
        self._write_decision_package()
        mock_container = f"{self.runtime_stack.stack_name}-runtime-upstream"
        route = self._write_quickstart_config(mock_container)

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
                selected, matched = self._route_once_answered(mock_container, route)
        finally:
            self._stop_serve_process(serve_process)

        self.assertEqual(selected, route["name"])
        signal = route["rules"]["conditions"][0]["name"]
        self.assertIn(signal, matched.split(","))
        self.print_test_result(True, f"{signal} answered; {selected} selected")

    def _write_decision_package(self) -> None:
        write_fixture(Path(self.test_dir) / "models" / PACKAGE, "decision2", "qwen3")

    def _write_quickstart_config(self, mock_container: str) -> dict:
        """The test stack's config plus the Quickstart's signal, route and
        deployment; returns the route that needs the signal's answer."""
        page = quickstart_router_config()
        signal = copy.deepcopy(page["routing"]["signals"]["decision"][0])
        route = copy.deepcopy(
            next(
                decision
                for decision in page["routing"]["decisions"]
                if decision["rules"]["conditions"]
            )
        )
        deployments = copy.deepcopy(page["global"]["model_catalog"]["deployments"])
        deployment = deployments[signal["deployment"]]
        self.assertEqual(deployment["provider"], "model_runtime")
        self.assertNotIn("endpoint", deployment, "the page's model must be managed")
        deployment["artifact"] = f"{CONTAINER_MODELS_DIR}/{PACKAGE}"
        # The fixture's weights are random: any answer matches, so the route
        # depends only on the managed runtime answering.
        signal["predicate"] = {"gte": 0}

        config_path = Path(
            self.write_minimal_canonical_config(
                base_url=f"http://{mock_container}:{PROVIDER_MOCKER_PORT}",
                provider="openai",
                api_only=True,
            )
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        default_route = config["routing"]["decisions"][0]
        route["priority"] = default_route["priority"] + 100
        route["modelRefs"] = copy.deepcopy(default_route["modelRefs"])
        config["routing"]["signals"] = {"decision": [signal]}
        config["routing"]["decisions"].insert(0, route)
        config["global"]["model_catalog"]["deployments"] = deployments
        # The page names no auto model names, so its request's "auto" routes.
        config["global"].get("router", {}).pop("auto_model_names", None)
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        return route

    def _route_once_answered(self, mock_container: str, route: dict) -> tuple[str, str]:
        """Send requests until the route that needs the answer is selected."""
        deadline = time.time() + ANSWER_TIMEOUT_SECONDS
        selected = matched = ""
        while time.time() < deadline:
            # As on the page: "auto" lets the Router choose, a named model
            # would skip the decisions.
            headers = self._send_mock_chat_completion(
                mock_container,
                request_headers={"x-vsr-debug": "true"},
                model="auto",
                content="Plan a three-step proof that there are infinitely many primes.",
            )
            selected = headers.get("x-vsr-selected-decision", "")
            matched = headers.get("x-vsr-matched-decision-model", "")
            if selected == route["name"]:
                return selected, matched
            time.sleep(2)
        diagnostics = self._container_log_diagnostics((self.ROUTER_CONTAINER_NAME,))
        self.fail(
            f"{route['name']} was not selected within {ANSWER_TIMEOUT_SECONDS} s "
            f"(last: {selected or 'none'})\n{diagnostics}"
        )


if __name__ == "__main__":
    unittest.main()
