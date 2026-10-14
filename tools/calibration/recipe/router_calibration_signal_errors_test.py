"""Expected runtime failures are exact evidence, never a blanket exemption."""

import copy
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

manifest = importlib.import_module("router_calibration_manifest")
probe_module = importlib.import_module("router_calibration_probe")
evaluation = importlib.import_module("router_calibration_evaluation")
support = importlib.import_module("router_calibration_support")
report = importlib.import_module("router_calibration_report")

EXPECTED = {
    "reask:repeat": "reask_evaluation_failed",
    "projection:recovery": "projection_input_failed",
    "projection:retry": "projection_input_failed",
}


class ExpectedSignalErrorsTest(unittest.TestCase):
    def document(self):
        return {
            "schema_version": "v1",
            "name": "explicit-input-limit",
            "routing_assets": {"yaml": "config.yaml", "dsl": "recipe.dsl"},
            "coverage": {
                "min_signal_assertion_percent": 0,
                "min_projection_assertion_percent": 0,
                "min_algorithm_assertion_percent": 0,
                "min_plugin_assertion_percent": 0,
                "required_request_shapes": ["text"],
                "min_tag_counts": {},
                "min_tag_pass_rate": {},
            },
            "decisions": [
                {
                    "id": "route",
                    "expected_decision": "route",
                    "model": "vllm-sr/auto",
                    "expected_algorithm": "static",
                    "expected_plugins": ["header_mutation"],
                    "expected_signals": {"context": ["long"]},
                    "variants": [{"id": "sample", "query": "Complete input."}],
                }
            ],
        }

    def load(self, document):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probes.yaml"
            path.write_text(json.dumps(document))
            return manifest.load_probe_manifest(path)[1]

    def response(self):
        return {
            "recipe": "default",
            "routing_decision": "route",
            "requested_model": "vllm-sr/auto",
            "selected_model": "worker",
            "recommended_models": ["worker"],
            "selection_status": "selected",
            "selection_method": "static",
            "decision_result": {
                "algorithm": "static",
                "plugins": ["header_mutation"],
                "matched_signals": {"context": ["long"]},
            },
            "eval_trace": [{"decision_name": "route", "matched": True}],
        }

    def expected_probe(self):
        doc = self.document()
        doc["decisions"][0]["expected_signal_errors"] = dict(EXPECTED)
        return self.load(doc)[0]

    def evaluate(self, response, probe, scope="deployment"):
        return evaluation.evaluate_probe(
            "http://router.example",
            probe,
            http_client=lambda *a, **kw: (200, response),
            scope=scope,
        )

    def test_manifest_inheritance_replacement_reset_and_independent_storage(self):
        doc = self.document()
        group = doc["decisions"][0]
        group["expected_signal_errors"] = dict(EXPECTED)
        override = {"embedding:similarity": "embedding_evaluation_failed"}
        group["variants"] += [
            {
                "id": "replace",
                "query": "Other input.",
                "expected_signal_errors": override,
            },
            {"id": "reset", "query": "Short input.", "expected_signal_errors": {}},
            {"id": "inherit", "query": "Another complete input."},
        ]
        probes = self.load(doc)
        self.assertEqual(
            [p.expected_signal_errors for p in probes],
            [
                EXPECTED,
                override,
                {},
                EXPECTED,
            ],
        )
        for index, probe in enumerate(probes):
            self.assertEqual(
                evaluation._build_request_payload(probe),
                {
                    "model": "vllm-sr/auto",
                    "text": group["variants"][index]["query"],
                },
            )
        probes[0].expected_signal_errors.clear()
        self.assertEqual(probes[3].expected_signal_errors, EXPECTED)
        self.assertEqual(group["expected_signal_errors"], EXPECTED)

    def test_schema_and_parser_reject_invalid_maps_at_both_levels(self):
        invalid = [
            None,
            [],
            "ignore",
            {"": "failed"},
            {"reask:repeat": ""},
            {" reask:repeat": "failed"},
            {"reask:repeat": "failed "},
            {"reask:repeat": True},
            {"reask:repeat": 1},
            {"reask:repeat": None},
            {"reask:repeat": ["failed"]},
        ]
        for raw in invalid:
            for target in ("decision", "variant"):
                with self.subTest(raw=raw, target=target):
                    doc = self.document()
                    group = doc["decisions"][0]
                    owner = group if target == "decision" else group["variants"][0]
                    owner["expected_signal_errors"] = raw
                    with self.assertRaises((ValueError, TypeError)):
                        self.load(doc)
                    with self.assertRaises((ValueError, TypeError)):
                        probe_module.load_grouped_probes(
                            doc["decisions"], Path("probes.yaml")
                        )

    def test_default_still_requires_no_errors_and_rejects_malformed_maps(self):
        probe = self.load(self.document())[0]
        self.assertEqual(probe.expected_signal_errors, {})
        self.assertTrue(self.evaluate(self.response(), probe)["matched"])
        for scope in ("policy", "deployment"):
            for actual in ({}, None, EXPECTED, [], False, ""):
                with self.subTest(scope=scope, actual=actual):
                    response = {**self.response(), "signal_errors": actual}
                    self.assertEqual(
                        self.evaluate(response, probe, scope)["matched"],
                        actual is None or actual == {},
                    )

    def test_exact_errors_pass_but_missing_extra_wrong_and_silent_success_fail(self):
        probe = self.expected_probe()
        cases = [
            (dict(EXPECTED), True),
            ({}, False),
            (None, False),
            (
                {key: code for key, code in EXPECTED.items() if key != "reask:repeat"},
                False,
            ),
            ({**EXPECTED, "reask:repeat": "different_code"}, False),
            ({**EXPECTED, "embedding:unrelated": "embedding_evaluation_failed"}, False),
        ]
        for scope in ("policy", "deployment"):
            for actual, matched in cases:
                with self.subTest(scope=scope, actual=actual):
                    result = self.evaluate(
                        {**self.response(), "signal_errors": actual}, probe, scope
                    )
                    self.assertEqual(result["matched"], matched)
                    self.assertEqual(result["signal_errors_matched"], matched)
                    self.assertEqual(result["expected_signal_errors"], EXPECTED)
                    self.assertEqual(result["signal_errors"], actual or {})
                    self.assertEqual(
                        json.loads(json.dumps(result))["expected_signal_errors"],
                        EXPECTED,
                    )
        result["expected_signal_errors"].clear()
        self.assertEqual(probe.expected_signal_errors, EXPECTED)

    def test_expected_errors_never_waive_other_routing_or_value_checks(self):
        probe = self.expected_probe()
        valid = {**self.response(), "signal_errors": dict(EXPECTED)}
        for field, value in (
            ("routing_decision", "wrong"),
            ("requested_model", "other"),
            ("recipe", "other"),
            ("recommended_models", []),
        ):
            self.assertFalse(self.evaluate({**valid, field: value}, probe)["matched"])
        for field, value in (
            ("algorithm", "other"),
            ("plugins", []),
            ("matched_signals", {}),
        ):
            response = copy.deepcopy(valid)
            response["decision_result"][field] = value
            self.assertFalse(self.evaluate(response, probe)["matched"])
        probe.expected_signal_values = {"reask:repeat": {"gte": 0}}
        valid["signal_values"] = {"reask:repeat": 0.5}
        result = self.evaluate(valid, probe)
        self.assertTrue(result["signal_errors_matched"])
        self.assertFalse(result["signal_values_matched"])
        self.assertFalse(result["matched"])

    def test_http_failure_preserves_expected_and_observed_without_acceptance(self):
        probe = self.expected_probe()
        payload = {**self.response(), "signal_errors": dict(EXPECTED)}
        result = support.failed_probe_result(
            probe,
            evaluation.ProbeRequestError("HTTP 503", 503, payload),
        )
        self.assertEqual(result["expected_signal_errors"], EXPECTED)
        self.assertEqual(result["signal_errors"], EXPECTED)
        self.assertFalse(result["signal_errors_matched"])
        self.assertFalse(result["matched"])

    def test_failure_report_shows_both_expected_and_observed_maps(self):
        result = self.evaluate(self.response(), self.expected_probe())
        summary = {"decision_id": "route", "matched": 0, "total": 1, "pass_rate": 0}
        rendered = "\n".join(report._render_decision_failures(summary, [result]))
        self.assertIn("signal errors", rendered)
        self.assertIn("reask_evaluation_failed", rendered)
        self.assertIn("observed `{}`", rendered)


if __name__ == "__main__":
    unittest.main()
