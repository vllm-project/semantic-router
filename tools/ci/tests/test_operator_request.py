"""The operator request oracle must reject readiness-only or bypassed responses."""

import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.error import HTTPError

import yaml

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "operator_request", ROOT / "tools/ci/check_operator_request.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class OperatorRequestTests(unittest.TestCase):
    def fixture(self):
        return {
            "model": "backend",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "mock": "provider-mocker",
                                "protocol": "chat_completions",
                                "model": "backend",
                                "user": ["nonce"],
                            }
                        )
                    }
                }
            ],
        }, {
            "x-vsr-selected-model": "backend",
            "x-vsr-selected-decision": "operator_ci_route",
            "x-vsr-response-path": "upstream",
        }

    def test_backend_payload_and_router_evidence_must_agree(self):
        payload, headers = self.fixture()
        probe.validate_response(payload, headers, model="backend", nonce="nonce")

    def test_missing_router_selection_rejects_direct_backend_bypass(self):
        payload, _ = self.fixture()
        with self.assertRaises(ValueError):
            probe.validate_response(payload, {}, model="backend", nonce="nonce")

    def test_backend_must_observe_model_rewrite_and_this_payload(self):
        for change in (
            {"model": "auto"},
            {"user": ["stale nonce"]},
            {"mock": "health"},
        ):
            with self.subTest(change=change):
                payload, headers = self.fixture()
                echo = json.loads(payload["choices"][0]["message"]["content"])
                echo.update(change)
                payload["choices"][0]["message"]["content"] = json.dumps(echo)
                with self.assertRaises(ValueError):
                    probe.validate_response(
                        payload, headers, model="backend", nonce="nonce"
                    )

    def test_selected_route_and_upstream_dispatch_are_required(self):
        for field, value in (
            ("x-vsr-selected-decision", ""),
            ("x-vsr-selected-decision", "unrelated-route"),
            ("x-vsr-response-path", "cache"),
            ("x-vsr-response-path", "fast_response"),
        ):
            with self.subTest(field=field, value=value):
                payload, headers = self.fixture()
                headers[field] = value
                with self.assertRaises(ValueError):
                    probe.validate_response(
                        payload, headers, model="backend", nonce="nonce"
                    )

    def test_invalid_request_fails_immediately_with_response_body(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            error = HTTPError(
                "http://router",
                400,
                "Bad Request",
                {},
                io.BytesIO(b"no model selected"),
            )
            with (
                mock.patch.object(
                    probe.argparse.ArgumentParser,
                    "parse_args",
                    return_value=SimpleNamespace(
                        url="http://router", model="backend", output=output
                    ),
                ),
                mock.patch.object(probe, "urlopen", side_effect=error) as request,
                mock.patch.object(probe.time, "sleep") as sleep,
                self.assertRaisesRegex(RuntimeError, "HTTP 400: no model selected"),
            ):
                probe.main()
            self.assertEqual(request.call_count, 1)
            sleep.assert_not_called()
            self.assertEqual(
                json.loads(output.read_text())["cases"][0]["status"], "failed"
            )

    def test_transient_unavailability_retries_then_verifies_actual_response(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            payload, headers = self.fixture()
            echo = json.loads(payload["choices"][0]["message"]["content"])
            echo["user"] = ["operator-reconciliation-fixture"]
            payload["choices"][0]["message"]["content"] = json.dumps(echo)
            response = io.StringIO(json.dumps(payload))
            response.headers = headers
            error = HTTPError(
                "http://router", 503, "Unavailable", {}, io.BytesIO(b"not ready")
            )
            with (
                mock.patch.object(
                    probe.argparse.ArgumentParser,
                    "parse_args",
                    return_value=SimpleNamespace(
                        url="http://router", model="backend", output=output
                    ),
                ),
                mock.patch.object(
                    probe.uuid, "uuid4", return_value=SimpleNamespace(hex="fixture")
                ),
                mock.patch.object(
                    probe, "urlopen", side_effect=[error, response]
                ) as request,
                mock.patch.object(probe.time, "sleep") as sleep,
            ):
                probe.main()
            self.assertEqual(request.call_count, 2)
            sleep.assert_called_once_with(1)
            self.assertEqual(
                json.loads(output.read_text())["cases"][0]["status"], "passed"
            )

    def test_every_workflow_variant_declares_the_route_exercised_by_the_probe(self):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/operator-ci.yml").read_text()
        )
        script = next(
            step["run"]
            for job in workflow["jobs"].values()
            for step in job.get("steps", [])
            if step.get("name") == "Create test SemanticRouter CR"
        )
        for backend in [
            "memory",
            "redis",
            "milvus",
            "qdrant",
            "hybrid",
            "mmbert",
            "complexity",
        ]:
            with (
                self.subTest(backend=backend),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory) / "resource.yaml"
                command = script.replace(
                    "${{ matrix.cache-backend }}", backend
                ).replace("${{ matrix.cache-backend }}", backend)
                subprocess.run(
                    ["bash", "-eu", "-c", 'kubectl() { cat > "$OUTPUT"; }\n' + command],
                    env={**os.environ, "OUTPUT": str(output)},
                    check=True,
                )
                spec = yaml.safe_load(output.read_text())["spec"]
                routing = spec["config"]["routing"]
                self.assertEqual(len(routing["decisions"]), 1)
                decision = routing["decisions"][0]
                self.assertEqual(decision["name"], "operator_ci_route")
                self.assertEqual(decision["rules"], {"operator": "AND"})
                self.assertEqual(
                    decision["modelRefs"][0]["model"], spec["vllmEndpoints"][0]["model"]
                )
