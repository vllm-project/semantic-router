#!/usr/bin/env python3
"""One decisions call per request stage, through `vllm-sr serve`.

The Router runs a Vela 2.0 deployment itself: a tiny random-weight encoder
package that the router image's `vllm-srun fixture` writes into the models
directory, so nothing is downloaded. Three signals ask it: the prompt guard
and a PII rule, which also read the conversation's earlier turns, and a
decision question. A long request with earlier turns must reach the
deployment as exactly one decisions call, which the Router counts in
`vsr_model_runtime_stage_decision_calls_total`.
"""

import json
import os
import re
import time
import unittest
from pathlib import Path
from urllib import request as urllib_request

import yaml
from cli_test_base import CLITestBase
from mock_upstream import PROVIDER_MOCKER_PORT, MockUpstreamMixin
from runtime_http import write_fixture
from serve_session import ServeSessionMixin

PACKAGE = "vela2-fixture"
DEPLOYMENT = "vela2"
# The Router mounts the models directory next to config.yaml here.
CONTAINER_MODELS_DIR = "/app/models"
ANSWER_TIMEOUT_SECONDS = 300
CALLS_METRIC = "vsr_model_runtime_stage_decision_calls_total"
# Longer than the fixture's one input, so the guard and PII questions read it in
# windows while the decision question reads its first tokens.
LONG_REQUEST = "Please summarise the attached release notes. " * 200
EARLIER_TURNS = [
    {"role": "user", "content": "Earlier I sent the release notes to Tom."},
    {"role": "assistant", "content": "I will keep them for the summary."},
]


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
)
class TestServeAsksADecisionModelOneCallPerStage(
    MockUpstreamMixin, ServeSessionMixin, CLITestBase
):
    """Every question a request asks the Vela 2.0 deployment travels in one call."""

    def test_a_long_request_with_history_is_one_decisions_call(self):
        self.print_test_header(
            "One Decisions Call Integration Test",
            "A long request with earlier turns asks the Vela 2.0 deployment once",
        )
        write_fixture(Path(self.test_dir) / "models" / PACKAGE, "vela2", "encoder")
        mock_container = f"{self.runtime_stack.stack_name}-onecall-upstream"
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
                self._wait_until_answered(mock_container)
                before = self._decision_calls()
                headers = self._chat(
                    mock_container,
                    [*EARLIER_TURNS, {"role": "user", "content": LONG_REQUEST}],
                )
                after = self._decision_calls()
        finally:
            self._stop_serve_process(serve_process)

        self.assertEqual(headers.get("x-vsr-selected-decision"), "answered")
        first = after.get("first", 0) - before.get("first", 0)
        later = after.get("later", 0) - before.get("later", 0)
        self.assertEqual(
            (first, later),
            (1, 0),
            f"the long request asked {DEPLOYMENT} in {first} first and {later} later calls",
        )
        self.print_test_result(True, "one decisions call for the request stage")

    def _write_config(self, mock_container: str) -> None:
        config_path = Path(
            self.write_minimal_canonical_config(
                base_url=f"http://{mock_container}:{PROVIDER_MOCKER_PORT}",
                provider="openai",
                api_only=True,
            )
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        catalog = config["global"].setdefault("model_catalog", {})
        catalog["deployments"] = {
            DEPLOYMENT: {
                "provider": "model_runtime",
                "artifact": f"{CONTAINER_MODELS_DIR}/{PACKAGE}",
                "device": "cpu",
            }
        }
        catalog["bindings"] = {
            "prompt_guard": {
                "deployment": DEPLOYMENT,
                "contract": "label_distribution.v1",
            },
            "pii_classifier": {"deployment": DEPLOYMENT, "contract": "token_spans.v1"},
        }
        # The bindings serve the guard and PII models, which the API-only global
        # block leaves unnamed and turns off.
        modules = catalog.setdefault("modules", {})
        modules["prompt_guard"] = {"enabled": True}
        modules.get("classifier", {}).pop("pii", None)
        config["routing"]["signals"] = {
            "jailbreak": [
                {"name": "attack", "threshold": 0.99, "include_history": True}
            ],
            "pii": [
                {"name": "personal_data", "threshold": 0.99, "include_history": True}
            ],
            "decision": [
                {
                    "name": "answered",
                    "deployment": DEPLOYMENT,
                    "question": {
                        "type": "noul",
                        "instructions": "Is this a request to the assistant?",
                    },
                    # The fixture's weights are random: any answer matches, so the
                    # route depends only on the deployment answering.
                    "predicate": {"gte": 0},
                }
            ],
        }
        default_route = config["routing"]["decisions"][0]
        refs = default_route["modelRefs"]
        config["routing"]["decisions"] = [
            {
                "name": "answered",
                "priority": default_route["priority"] + 200,
                "rules": {
                    "operator": "AND",
                    "conditions": [{"type": "decision", "name": "answered"}],
                },
                "modelRefs": refs,
            },
            {
                "name": "flagged",
                "priority": default_route["priority"] + 100,
                "rules": {
                    "operator": "OR",
                    "conditions": [
                        {"type": "jailbreak", "name": "attack"},
                        {"type": "pii", "name": "personal_data"},
                    ],
                },
                "modelRefs": refs,
            },
            default_route,
        ]
        config["global"].get("router", {}).pop("auto_model_names", None)
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

    def _chat(self, mock_container: str, messages: list[dict]) -> dict:
        """Send one chat request through the listener and return its headers."""
        port = 8888 + self.runtime_stack.port_offset
        request = urllib_request.Request(
            f"http://localhost:{port}/v1/chat/completions",
            data=json.dumps({"model": "auto", "messages": messages}).encode(),
            headers={"Content-Type": "application/json", "x-vsr-debug": "true"},
            method="POST",
        )
        with urllib_request.urlopen(request, timeout=120) as response:
            self.assertEqual(response.status, 200)
            response.read()
            return dict(response.headers)

    def _wait_until_answered(self, mock_container: str) -> None:
        """Send short requests until the deployment answers the decision question."""
        deadline = time.time() + ANSWER_TIMEOUT_SECONDS
        selected = ""
        while time.time() < deadline:
            try:
                headers = self._chat(
                    mock_container,
                    [{"role": "user", "content": "Is the deployment ready?"}],
                )
            except OSError:
                time.sleep(2)
                continue
            selected = headers.get("x-vsr-selected-decision", "")
            if selected == "answered":
                return
            time.sleep(2)
        diagnostics = self._container_log_diagnostics((self.ROUTER_CONTAINER_NAME,))
        self.fail(
            f"{DEPLOYMENT} did not answer within {ANSWER_TIMEOUT_SECONDS} s "
            f"(last route: {selected or 'none'})\n{diagnostics}"
        )

    def _decision_calls(self) -> dict[str, float]:
        """The Router's decisions calls to the deployment so far, by round."""
        url = f"http://localhost:{self.runtime_stack.metrics_port}/metrics"
        with urllib_request.urlopen(url, timeout=10) as response:
            text = response.read().decode()
        calls: dict[str, float] = {}
        pattern = re.compile(
            rf'^{CALLS_METRIC}\{{deployment="{DEPLOYMENT}",round="(\w+)"\}} ([0-9.e+]+)$'
        )
        for line in text.splitlines():
            match = pattern.match(line)
            if match:
                calls[match.group(1)] = float(match.group(2))
        return calls


if __name__ == "__main__":
    unittest.main()
