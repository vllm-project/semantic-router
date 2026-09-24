"""Raw evidence must be observed, bounded and error-free, not necessarily matched."""

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
evaluation = importlib.import_module("router_calibration_evaluation")
support = importlib.import_module("router_calibration_support")
coverage = importlib.import_module("recipe_conformance_coverage")
conformance = importlib.import_module("recipe_conformance")
values = importlib.import_module("router_calibration_signal_values")
report = importlib.import_module("router_calibration_report")


class RawSignalValueTest(unittest.TestCase):
    def document(self):
        return {
            "schema_version": "v1",
            "name": "raw-evidence",
            "routing_assets": {"yaml": "config.yaml", "dsl": "recipe.dsl"},
            "acceptance": {},
            "coverage": {
                "min_signal_assertion_percent": 100,
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
                    "expected_algorithm": "static",
                    "expected_signal_values": {
                        "embedding:information": {"gte": -1, "lte": 1}
                    },
                    "variants": [{"id": "sample", "query": "Describe a neutral item."}],
                }
            ],
        }

    def load(self, document):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probes.yaml"
            path.write_text(json.dumps(document))
            return manifest.load_probe_manifest(path)

    def response(self):
        return {
            "recipe": "default",
            "routing_decision": "route",
            "selected_model": "worker",
            "recommended_models": ["worker"],
            "selection_status": "selected",
            "selection_method": "static",
            "signal_values": {"embedding:information": 0.3},
            "decision_result": {
                "algorithm": "static",
                "matched_signals": {},
            },
            "eval_trace": [{"decision_name": "route", "matched": True}],
        }

    def evaluate(self, response, probe=None, scope="deployment"):
        probe = probe or self.load(self.document())[1][0]
        return evaluation.evaluate_probe(
            "http://router.example",
            probe,
            http_client=lambda *a, **kw: (200, response),
            scope=scope,
        )

    def test_schema_and_group_variant_override_preserve_requests(self):
        doc = self.document()
        doc["decisions"][0]["variants"] += [
            {
                "id": "override",
                "query": "Another item.",
                "expected_signal_values": {"structure:size": {"gte": 2}},
            },
            {"id": "empty", "query": "A third item.", "expected_signal_values": {}},
        ]
        _, probes = self.load(doc)
        self.assertEqual(
            probes[0].expected_signal_values,
            {"embedding:information": {"gte": -1, "lte": 1}},
        )
        self.assertEqual(
            probes[1].expected_signal_values, {"structure:size": {"gte": 2}}
        )
        self.assertEqual(probes[2].expected_signal_values, {})
        for index, probe in enumerate(probes):
            self.assertEqual(
                evaluation._build_request_payload(probe),
                {"text": doc["decisions"][0]["variants"][index]["query"]},
            )
        probes[0].expected_signal_values["embedding:information"]["gte"] = 0
        self.assertEqual(
            doc["decisions"][0]["expected_signal_values"]["embedding:information"][
                "gte"
            ],
            -1,
        )

    def test_invalid_bounds_rejected_at_both_scopes(self):
        invalid = [
            None,
            [],
            {"embedding:information": {}},
            {"embedding:information": {"gt": 0}},
            {"embedding:information": {"gte": True}},
            {"embedding:information": {"gte": "0"}},
            {"embedding:information": {"gte": None, "lte": 1}},
            {"embedding:information": {"gte": float("nan")}},
            {"embedding:information": {"lte": float("inf")}},
            {"embedding:information": {"gte": 2, "lte": 1}},
            {"embedding:": {"gte": 0}},
        ]
        for raw in invalid:
            for target in ("group", "variant"):
                with self.subTest(raw=raw, target=target):
                    doc = self.document()
                    group = doc["decisions"][0]
                    owner = group if target == "group" else group["variants"][0]
                    owner["expected_signal_values"] = raw
                    with self.assertRaises((ValueError, TypeError)):
                        self.load(doc)

    def test_unmatched_finite_value_passes_but_does_not_replace_match_assertions(self):
        response = self.response()
        self.assertTrue(self.evaluate(response)["matched"])
        _, probes = self.load(self.document())
        probe = probes[0]
        probe.expected_signals = (("embeddings", "information"),)
        self.assertFalse(self.evaluate(response, probe)["matched"])
        probe.expected_signals = ()
        probe.forbidden_signals = (("embeddings", "information"),)
        self.assertTrue(self.evaluate(response, probe)["matched"])
        response["decision_result"]["matched_signals"] = {"embeddings": ["information"]}
        self.assertFalse(self.evaluate(response, probe)["matched"])

    def test_missing_invalid_range_and_errors_fail_both_scopes(self):
        for value in (None, "0.3", True, float("nan"), float("inf"), -2, 2):
            for scope in ("policy", "deployment"):
                with self.subTest(value=value, scope=scope):
                    response = self.response()
                    response["signal_values"]["embedding:information"] = value
                    result = self.evaluate(response, scope=scope)
                    self.assertFalse(result["matched"])
                    self.assertFalse(result["signal_values_matched"])
                    self.assertTrue(result["signal_value_errors"])
        response = self.response()
        response["decision_result"]["signal_values"] = response.pop("signal_values")
        self.assertFalse(
            self.evaluate(response)["signal_values_matched"],
            "nested mock values cannot replace the canonical top-level field",
        )
        response = self.response()
        response["signal_errors"] = {"embedding:information": "evaluation_failed"}
        result = self.evaluate(response)
        self.assertFalse(result["signal_values_matched"])
        self.assertEqual(
            result["observed_signal_values"], {"embedding:information": 0.3}
        )
        summary = {"decision_id": "route", "matched": 0, "total": 1, "pass_rate": 0}
        rendered = "\n".join(report._render_decision_failures(summary, [result]))
        self.assertIn("signal values", rendered)
        self.assertIn("embedding:information", rendered)

    def test_error_parent_and_complexity_verdict_invalidate_raw_value(self):
        for key, error in (
            ("classifier:risk:label", "classifier:risk"),
            ("complexity:depth:margin", "complexity:depth:hard"),
        ):
            self.assertFalse(
                values.compare_signal_values(
                    {key: {"gte": -1}}, {key: 0}, {error: "failed"}
                )["matched"]
            )

    def test_http_failure_keeps_raw_diagnostic_without_acceptance(self):
        _, probes = self.load(self.document())
        payload = self.response()
        payload["signal_errors"] = {"embedding:information": "failed"}
        exc = evaluation.ProbeRequestError("HTTP 503", 503, payload)
        result = support.failed_probe_result(probes[0], exc)
        self.assertFalse(result["matched"])
        self.assertFalse(result["signal_values_matched"])
        self.assertEqual(
            result["observed_signal_values"], {"embedding:information": 0.3}
        )
        self.assertIs(result["raw_response"], payload)

    def test_top_level_values_are_authoritative_when_nested_values_conflict(self):
        response = self.response()
        response["decision_result"]["signal_values"] = {"embedding:information": 2}
        self.assertTrue(self.evaluate(response)["signal_values_matched"])
        response["signal_values"]["embedding:information"] = 2
        response["decision_result"]["signal_values"]["embedding:information"] = 0.3
        self.assertFalse(self.evaluate(response)["signal_values_matched"])

    def test_complexity_errors_stay_with_the_complete_rule_name(self):
        for key, error, related in (
            ("complexity:depth", "complexity:other:hard", False),
            ("complexity:depth", "complexity:depth:hard", True),
            ("complexity:depth:margin", "complexity:other:hard", False),
            ("complexity:literal:part:margin", "complexity:literal:other:hard", False),
            ("complexity:literal:part:margin", "complexity:literal:part:hard", True),
            ("complexity:literal:hard:margin", "complexity:literal:hard", False),
            ("complexity:literal:hard:margin", "complexity:literal:hard:easy", True),
        ):
            with self.subTest(key=key, error=error):
                result = values.compare_signal_values(
                    {key: {"gte": -1}}, {key: 0}, {error: "failed"}
                )
                self.assertEqual(result["matched"], not related)

    def test_static_coverage_resolves_current_recipe_and_literal_rule_names(self):
        doc, probes = self.load(self.document())
        profile = {
            "signals": {"embeddings": [{"name": "information"}]},
            "decisions": [{"name": "route", "algorithm": {"type": "static"}}],
        }
        kwargs = {
            "profiles": {"default": profile},
            "decisions": {("default", "route"): profile["decisions"][0]},
            "entrypoints": {},
            "probes": probes,
            "policy": doc["coverage"],
        }
        result = coverage.collect_coverage(**kwargs)
        self.assertEqual(result["signals"]["asserted"], ["embeddings:information"])
        self.assertTrue(result["passed"])
        conformance._validate_probe_signals(
            Path("probes.yaml"), probes[0], {"embeddings": {"information"}}
        )
        other = copy.deepcopy(probes[0])
        other.expected_signal_values = {"embedding:missing": {"gte": 0}}
        with self.assertRaisesRegex(ValueError, "unknown signal rule"):
            coverage.collect_coverage(**{**kwargs, "probes": [other]})
        with self.assertRaisesRegex(ValueError, "unknown signal rule"):
            conformance._validate_probe_signals(
                Path("probes.yaml"), other, {"embeddings": {"information"}}
            )
        with self.assertRaisesRegex(ValueError, "unknown signal rule"):
            coverage.collect_coverage(**{**kwargs, "profiles": {"foreign": profile}})
        self.assertEqual(
            values.signal_value_reference(
                "embedding:literal:colon", {"embeddings": {"literal:colon"}}
            ),
            ("embeddings", "literal:colon"),
        )
        self.assertEqual(
            values.signal_value_reference(
                "complexity:depth:margin", {"complexity": {"depth"}}
            ),
            ("complexity", "depth"),
        )
        with self.assertRaises(ValueError):
            values.signal_value_reference(
                "embedding:information:invented", {"embeddings": {"information"}}
            )


if __name__ == "__main__":
    unittest.main()
