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


def memory_resource():
    """The SemanticRouter the operator workflow applies."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/operator-ci.yml").read_text())
    script = next(
        step["run"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Create test SemanticRouter CR"
    )
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "resource.yaml"
        subprocess.run(
            ["bash", "-eu", "-c", 'kubectl() { cat > "$OUTPUT"; }\n' + script],
            env={**os.environ, "OUTPUT": str(output)},
            check=True,
        )
        return yaml.safe_load(output.read_text())


def undeclared_fields(value, schema, path="spec"):
    """The paths of value's fields that the structural schema does not declare."""
    if schema.get("x-kubernetes-preserve-unknown-fields"):
        return []
    if isinstance(value, list):
        items = schema.get("items", {})
        return [
            field
            for index, item in enumerate(value)
            for field in undeclared_fields(item, items, f"{path}[{index}]")
        ]
    if not isinstance(value, dict):
        return []
    if "properties" not in schema:
        extra = schema.get("additionalProperties")
        if not isinstance(extra, dict):
            return []
        return [
            field
            for key, item in sorted(value.items())
            for field in undeclared_fields(item, extra, f"{path}.{key}")
        ]
    fields = []
    for key, item in sorted(value.items()):
        if key in schema["properties"]:
            field_schema = schema["properties"][key]
            fields.extend(undeclared_fields(item, field_schema, f"{path}.{key}"))
        else:
            fields.append(f"{path}.{key}")
    return fields


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
            {"model": "vllm-sr/auto"},
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

    def test_memory_deployment_declares_the_route_exercised_by_the_probe(self):
        resource = memory_resource()
        self.assertEqual(resource["metadata"]["name"], "test-router-memory")
        spec = resource["spec"]
        self.assertEqual(spec["config"]["response_cache"]["backend_type"], "memory")
        (recipe,) = spec["config"]["recipes"]
        self.assertEqual(len(recipe["decisions"]), 1)
        decision = recipe["decisions"][0]
        self.assertEqual(decision["name"], "operator_ci_route")
        self.assertEqual(decision["rules"], {"operator": "AND"})
        self.assertEqual(
            decision["modelRefs"][0]["model"], spec["vllmEndpoints"][0]["model"]
        )

    def test_memory_resource_sets_only_fields_the_crd_declares(self):
        # The API server's strict field validation refuses any other field.
        crd = yaml.safe_load(
            (
                ROOT / "deploy/operator/config/crd/bases/vllm.ai_semanticrouters.yaml"
            ).read_text()
        )
        (version,) = crd["spec"]["versions"]
        schema = version["schema"]["openAPIV3Schema"]["properties"]["spec"]
        self.assertEqual(undeclared_fields(memory_resource()["spec"], schema), [])

    def test_undeclared_fields_names_each_path_the_schema_lacks(self):
        schema = {
            "properties": {
                "config": {
                    "properties": {
                        "recipes": {"x-kubernetes-preserve-unknown-fields": True},
                        "api": {"properties": {"enabled": {}}},
                    }
                },
                "endpoints": {"items": {"properties": {"name": {}}}},
                "limits": {"additionalProperties": {}},
            }
        }
        spec = {
            "config": {
                "recipes": [{"any": 1}],
                "api": {"enabled": True, "old": 1},
                "tools": {},
            },
            "endpoints": [{"name": "a", "port": 1}],
            "limits": {"cpu": "1"},
        }
        self.assertEqual(
            undeclared_fields(spec, schema),
            ["spec.config.api.old", "spec.config.tools", "spec.endpoints[0].port"],
        )
