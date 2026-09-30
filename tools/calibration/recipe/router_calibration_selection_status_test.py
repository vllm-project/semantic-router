"""Selection rejection is distinct from both routing failure and successful dispatch."""

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

coverage_module = importlib.import_module("recipe_conformance_coverage")
evaluation_module = importlib.import_module("router_calibration_evaluation")
manifest_module = importlib.import_module("router_calibration_manifest")
probe_module = importlib.import_module("router_calibration_probe")
report_module = importlib.import_module("router_calibration_report")
collect_coverage = coverage_module.collect_coverage
compare_eval_selection = evaluation_module.compare_eval_selection
compare_expected_signals = evaluation_module.compare_expected_signals
evaluate_probe = evaluation_module.evaluate_probe
load_probe_manifest = manifest_module.load_probe_manifest
summarize_decision_results = manifest_module.summarize_decision_results
Probe = probe_module.Probe
_render_variant_section = report_module._render_variant_section


class UnavailableSelectionTest(unittest.TestCase):
    def probe(self):
        return Probe(
            decision_id="bounded",
            variant_id="oversized",
            probe_id="bounded:oversized",
            expected_decision="simple",
            expected_recipe="bounded",
            expected_algorithm="multi_factor",
            expected_selection_status="unavailable",
            expected_signals=(("keywords", "direct"),),
            query="Summarize the supplied record.",
        )

    def response(self):
        return {
            "recipe": "bounded",
            "routing_decision": "simple",
            "selection_status": "unavailable",
            "selection_reason": "No assigned candidate satisfies the request budget",
            "selected_model": "",
            "recommended_models": ["assigned-but-too-small"],
            "eval_trace": [{"decision_name": "simple", "matched": True}],
            "decision_result": {
                "algorithm": "multi_factor",
                "matched_signals": {"keywords": ["direct"]},
            },
        }

    def evaluate(self, response, status=200):
        return evaluate_probe(
            "http://router.example:8080",
            self.probe(),
            http_client=lambda *args, **kwargs: (status, response),
        )

    def test_unavailable_keeps_declared_candidates_without_a_final_model(self):
        result = self.evaluate(self.response())
        self.assertTrue(result["matched"])
        self.assertEqual(result["expected_selection_status"], "unavailable")
        self.assertEqual(
            result["selection_reason"], self.response()["selection_reason"]
        )
        self.assertEqual(result["selection_method"], "")
        markdown = "\n".join(_render_variant_section([result]))
        self.assertIn("simple / unavailable", markdown)
        summary = summarize_decision_results([result], {})
        self.assertEqual(summary[0]["expected_selection_status"], "unavailable")

    def test_unavailable_requires_exact_status_no_model_and_a_reason(self):
        mutations = (
            {
                "selection_status": "selected",
                "selected_model": "assigned-but-too-small",
            },
            {"selection_status": "execution_required"},
            {"selected_model": "unrelated"},
            {"selection_reason": "  "},
            {"selection_status": ""},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                response = self.response()
                response.update(mutation)
                self.assertFalse(self.evaluate(response)["selection_matched"])

    def test_unavailable_does_not_relax_decision_signals_trace_or_errors(self):
        mutations = (
            {"routing_decision": "reasoning"},
            {"recipe": "foreign"},
            {"decision_result": {"algorithm": "static", "matched_signals": {}}},
            {"eval_trace": []},
            {"signal_errors": {"complexity:difficulty": "inference unavailable"}},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                response = self.response()
                response.update(mutation)
                self.assertFalse(self.evaluate(response)["matched"])

    def test_http_error_is_not_an_unavailable_selection(self):
        with self.assertRaises(RuntimeError):
            self.evaluate(self.response(), status=503)

    def test_fast_response_requires_its_actual_model_free_selection_contract(self):
        probe = self.probe()
        probe.expected_selection_status = "not_required"
        probe.expected_algorithm = None
        probe.expected_plugins = ("fast_response",)
        response = self.response()
        response.update(
            selection_status="not_required", selection_method="fast_response"
        )
        response["decision_result"].update(algorithm="", plugins=["fast_response"])
        result = evaluate_probe(
            "http://router.example:8080",
            probe,
            http_client=lambda *args, **kwargs: (200, response),
        )
        self.assertTrue(result["matched"])
        for mutation in (
            {"selection_status": "unavailable"},
            {"selection_method": ""},
            {"selection_method": "multi_factor"},
            {"selected_model": "fabricated"},
            {"selection_reason": ""},
        ):
            with self.subTest(mutation=mutation):
                changed = {**response, **mutation}
                result = evaluate_probe(
                    "http://router.example:8080",
                    probe,
                    http_client=lambda *args, value=changed, **kwargs: (200, value),
                )
                self.assertFalse(result["matched"])

    def test_legacy_algorithm_expectation_still_rejects_unavailable(self):
        result = compare_eval_selection(
            algorithm="multi_factor",
            selected_model="",
            status="unavailable",
            method="",
            recommended_models=("assigned",),
        )
        self.assertFalse(result["matched"])
        legacy = compare_eval_selection(
            algorithm="multi_factor",
            selected_model="assigned",
            status="selected",
            method="multi_factor",
            recommended_models=("assigned",),
        )
        self.assertTrue(legacy["matched"])

    def test_unknown_direct_status_expectation_fails_closed(self):
        result = compare_eval_selection(
            algorithm=None,
            selected_model="",
            status="invented",
            method="",
            recommended_models=(),
            expected_status="invented",
            reason="synthetic reason",
        )
        self.assertFalse(result["matched"])

    def test_manifest_group_inheritance_and_legacy_default(self):
        document = {
            "schema_version": "v1",
            "name": "bounded",
            "routing_assets": {"yaml": "config.yaml", "dsl": "recipe.dsl"},
            "coverage": {
                "min_signal_assertion_percent": 100,
                "min_projection_assertion_percent": 100,
                "min_algorithm_assertion_percent": 100,
                "min_plugin_assertion_percent": 100,
                "required_request_shapes": ["text"],
                "min_tag_counts": {},
                "min_tag_pass_rate": {},
            },
            "decisions": [
                {
                    "id": "bounded",
                    "expected_decision": "simple",
                    "expected_selection_status": "unavailable",
                    "variants": [{"id": "oversized", "query": "Summarize a record."}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "probes.yaml"
            path.write_text(json.dumps(document))
            _, probes = load_probe_manifest(path)
            self.assertEqual(probes[0].expected_selection_status, "unavailable")
            document["decisions"][0]["expected_selection_status"] = "not_required"
            path.write_text(json.dumps(document))
            _, probes = load_probe_manifest(path)
            self.assertEqual(probes[0].expected_selection_status, "not_required")
            del document["decisions"][0]["expected_selection_status"]
            path.write_text(json.dumps(document))
            _, probes = load_probe_manifest(path)
            self.assertIsNone(probes[0].expected_selection_status)
            for value in ("invented", "", None):
                with self.subTest(value=value):
                    invalid = copy.deepcopy(document)
                    invalid["decisions"][0]["expected_selection_status"] = value
                    path.write_text(json.dumps(invalid))
                    with self.assertRaises(ValueError):
                        load_probe_manifest(path)


class FastResponseCoverageTest(unittest.TestCase):
    def test_containment_counts_decision_and_plugin_without_fake_static_algorithm(self):
        probe = Probe(
            decision_id="guard",
            variant_id="blocked",
            probe_id="guard:blocked",
            expected_decision="guard",
            expected_plugins=("fast_response",),
            expected_selection_status="not_required",
            query="Synthetic containment input.",
        )
        coverage = collect_coverage(
            profiles={"default": {}},
            decisions={
                ("default", "guard"): {"plugins": [{"type": "fast_response"}]},
            },
            entrypoints={},
            probes=[probe],
            policy={
                "min_algorithm_assertion_percent": 100,
                "min_plugin_assertion_percent": 100,
            },
        )
        self.assertTrue(coverage["passed"])
        self.assertEqual(coverage["algorithms"]["total"], 0)
        self.assertEqual(coverage["decisions"]["covered"], 1)
        self.assertEqual(coverage["plugins"]["covered"], 1)
        legacy = collect_coverage(
            profiles={"default": {}},
            decisions={("default", "plain"): {}},
            entrypoints={},
            probes=[],
            policy={},
        )
        self.assertEqual(legacy["algorithms"]["configured"], ["default:plain:static"])


class ComplexityCoverageTest(unittest.TestCase):
    def test_real_outcome_covers_rule_without_rewriting_other_colon_names(self):
        probe = Probe(
            decision_id="simple",
            variant_id="plain",
            probe_id="simple:plain",
            expected_decision="simple",
            query="Explain a term.",
            expected_signals=(
                ("complexity", "difficulty:easy"),
                ("keywords", "semantic:v1"),
            ),
        )
        coverage = collect_coverage(
            profiles={
                "default": {
                    "signals": {
                        "complexity": [{"name": "difficulty"}],
                        "keywords": [{"name": "semantic:v1"}],
                    }
                }
            },
            decisions={("default", "simple"): {}},
            entrypoints={},
            probes=[probe],
            policy={"min_signal_assertion_percent": 100},
        )
        self.assertTrue(coverage["passed"])
        self.assertEqual(coverage["signals"]["percent"], 100)
        self.assertEqual(
            coverage["signals"]["asserted"],
            [
                "complexity:difficulty",
                "keywords:semantic:v1",
            ],
        )
        exact = compare_expected_signals(
            expected=(("complexity", "difficulty:hard"),),
            forbidden=(),
            actual={"complexity": ["difficulty:easy"]},
            match_mode="contains",
        )
        self.assertFalse(exact["matched"])


if __name__ == "__main__":
    unittest.main()
