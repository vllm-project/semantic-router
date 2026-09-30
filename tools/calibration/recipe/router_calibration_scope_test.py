"""Policy evidence must not be mistaken for executable model selection."""

import importlib
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

evaluation = importlib.import_module("router_calibration_evaluation")
support = importlib.import_module("router_calibration_support")
probe_module = importlib.import_module("router_calibration_probe")
loop = importlib.import_module("router_calibration_loop")
conformance = importlib.import_module("recipe_conformance")
report = importlib.import_module("router_calibration_report")
consolidated = importlib.import_module("recipe_conformance_report")


class EvaluationScopeTest(unittest.TestCase):
    def probe(self):
        return probe_module.Probe(
            decision_id="direct",
            variant_id="context",
            probe_id="direct:context",
            expected_decision="simple",
            expected_recipe="default",
            expected_algorithm="multi_factor",
            expected_signals=(("keywords", "direct"),),
            query="Summarize the supplied record.",
        )

    def response(self):
        return {
            "recipe": "default",
            "routing_decision": "simple",
            "selection_status": "unavailable",
            "selection_reason": "All assigned models are too small",
            "recommended_models": ["assigned"],
            "eval_trace": [{"decision_name": "simple", "matched": True}],
            "decision_result": {
                "algorithm": "multi_factor",
                "matched_signals": {"keywords": ["direct"]},
            },
        }

    def run_scope(self, scope, response=None, probe=None, status=200):
        response = self.response() if response is None else response
        with mock.patch.object(
            support, "http_json", return_value=(status, response)
        ) as transport:
            result = support.evaluate_probes(
                "http://router.example", [probe or self.probe()], scope=scope
            )
        transport.assert_called_once()
        self.assertEqual(
            transport.call_args.args[:2],
            ("POST", "http://router.example/api/v1/routing/preview?trace=true"),
        )
        return result

    def test_capacity_rejection_passes_only_explicit_policy_scope(self):
        policy = self.run_scope("policy")
        self.assertTrue(policy["passed"])
        self.assertTrue(policy["scopes"]["policy"]["passed"])
        self.assertFalse(policy["scopes"]["deployment"]["passed"])
        self.assertEqual(policy["selection_status_counts"], {"unavailable": 1})
        self.assertEqual(
            policy["selection_reasons"][0]["reason"],
            self.response()["selection_reason"],
        )
        result = policy["results"][0]
        self.assertFalse(result["selection_matched"])
        self.assertFalse(result["deployment_matched"])
        self.assertEqual(result["raw_response"], self.response())
        self.assertEqual(result["http_status"], 200)
        strict = self.run_scope("deployment")
        self.assertFalse(strict["passed"])
        with mock.patch.object(
            support, "http_json", return_value=(200, self.response())
        ):
            default = support.evaluate_probes("http://router.example", [self.probe()])
        self.assertFalse(default["passed"])
        self.assertEqual(default["evaluation_scope"], "deployment")
        markdown = report.render_markdown_summary({}, None, policy, None, None)
        self.assertIn("Policy: `1/1`", markdown)
        self.assertIn("Deployment: `0/1`", markdown)
        self.assertIn("All assigned models are too small", markdown)

    def test_policy_requires_valid_selection_evidence_and_policy_results(self):
        mutations = (
            {"selection_status": ""},
            {"selection_status": "invented"},
            {"selected_model": "fabricated"},
            {"selection_reason": ""},
            {"selection_reason": {"message": "not a string"}},
            {"recommended_models": "assigned"},
            {"signal_errors": {"complexity:difficulty": "inference unavailable"}},
            {"routing_decision": "wrong"},
            {"recipe": "foreign"},
            {"eval_trace": []},
            {
                "decision_result": {
                    "algorithm": "static",
                    "matched_signals": {"keywords": ["direct"]},
                }
            },
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assertFalse(
                    self.run_scope("policy", {**self.response(), **mutation})["passed"]
                )

    def test_explicit_selection_contract_is_not_removed_by_policy_scope(self):
        probe = self.probe()
        probe.expected_selection_status = "unavailable"
        self.assertTrue(self.run_scope("policy", probe=probe)["passed"])
        selected = {
            **self.response(),
            "selection_status": "selected",
            "selected_model": "assigned",
            "selection_method": "multi_factor",
        }
        self.assertFalse(self.run_scope("policy", selected, probe)["passed"])
        probe.expected_selection_status = "not_required"
        probe.expected_algorithm = None
        probe.expected_plugins = ("fast_response",)
        contained = self.response()
        contained.update(
            selection_status="not_required", selection_method="fast_response"
        )
        contained["decision_result"].update(algorithm="", plugins=["fast_response"])
        self.assertTrue(self.run_scope("policy", contained, probe)["passed"])
        contained["selection_method"] = "multi_factor"
        self.assertFalse(self.run_scope("policy", contained, probe)["passed"])

    def test_http_failure_stays_failure_and_retains_raw_response(self):
        raw = {"error": "classifier unavailable"}
        result = self.run_scope("policy", raw, status=503)
        self.assertFalse(result["passed"])
        self.assertFalse(result["scopes"]["policy"]["passed"])
        self.assertFalse(result["scopes"]["deployment"]["passed"])
        self.assertEqual(result["results"][0]["raw_response"], raw)
        self.assertEqual(result["results"][0]["http_status"], 503)

    def test_executable_result_passes_both_scopes_without_backend_call(self):
        response = self.response()
        response.update(
            selection_status="selected",
            selected_model="assigned",
            selection_method="multi_factor",
        )
        result = self.run_scope("policy", response)
        self.assertTrue(all(scope["passed"] for scope in result["scopes"].values()))
        self.assertEqual(result["selection_status_counts"], {"selected": 1})

    def test_unknown_scope_fails_before_transport(self):
        with (
            mock.patch.object(support, "http_json") as transport,
            self.assertRaisesRegex(ValueError, "unknown evaluation scope"),
        ):
            support.evaluate_probes(
                "http://router.example", [self.probe()], scope="skip"
            )
        transport.assert_not_called()

    def test_cli_scope_defaults_and_explicit_options(self):
        variants = (
            (
                loop.build_parser(),
                [
                    "eval",
                    "--router-url",
                    "http://router.example",
                    "--probes",
                    "probes.yaml",
                ],
            ),
            (
                loop.build_parser(),
                [
                    "run",
                    "--router-url",
                    "http://router.example",
                    "--probes",
                    "probes.yaml",
                ],
            ),
            (
                conformance.build_parser(),
                [
                    "eval",
                    "--recipe",
                    "example",
                    "--router-url",
                    "http://router.example",
                ],
            ),
        )
        for parser, args in variants:
            with self.subTest(args=args):
                self.assertEqual(parser.parse_args(args).scope, "deployment")
                self.assertEqual(
                    parser.parse_args([*args, "--scope", "policy"]).scope, "policy"
                )

    def test_both_cli_paths_forward_policy_and_preserve_deployment_tag_gates(self):
        probe = self.probe()
        probe.tags = ("class:negative",)
        second = replace(probe, probe_id="direct:short", variant_id="short", tags=())
        ready = self.response()
        ready.update(
            selection_status="selected",
            selected_model="assigned",
            selection_method="multi_factor",
        )
        manifest = {
            "acceptance": {"min_probe_pass_rate": 50, "min_decision_pass_rate": 50},
            "coverage": {"min_tag_pass_rate": {"negative": 100}},
        }
        with mock.patch.object(
            support, "http_json", side_effect=[(200, self.response()), (200, ready)]
        ):
            evidence = support.evaluate_probes(
                "http://router.example", [probe, second], manifest, scope="policy"
            )
        # Deployment passes the deliberately loose aggregate rate but must
        # still fail its independent negative-control requirement.
        self.assertTrue(evidence["scopes"]["deployment"]["passed"])

        @dataclass
        class Inventory:
            name: str = "synthetic"

        with tempfile.TemporaryDirectory() as directory:
            args = conformance.build_parser().parse_args(
                [
                    "--output-dir",
                    directory,
                    "eval",
                    "--recipe",
                    "synthetic",
                    "--router-url",
                    "http://router.example",
                    "--scope",
                    "policy",
                ]
            )
            with (
                mock.patch.object(
                    conformance, "build_recipe_inventory", return_value=Inventory()
                ),
                mock.patch.object(conformance, "load_yaml_mapping", return_value={}),
                mock.patch.object(
                    conformance,
                    "load_probe_manifest",
                    return_value=(manifest, [probe, second]),
                ),
                mock.patch.object(
                    conformance, "evaluate_probes", return_value=evidence
                ) as run,
                mock.patch("builtins.print"),
            ):
                self.assertEqual(conformance.command_eval(args), 0)
            self.assertEqual(run.call_args.kwargs["scope"], "policy")
            saved = json.loads(
                (Path(directory) / "synthetic/eval-report.json").read_text()
            )["evaluation"]
            self.assertTrue(saved["scopes"]["policy"]["passed"])
            self.assertFalse(saved["scopes"]["deployment"]["passed"])
            self.assertFalse(
                saved["scopes"]["deployment"]["coverage_acceptance"]["passed"]
            )

        args = loop.build_parser().parse_args(
            [
                "eval",
                "--probes",
                "synthetic.yaml",
                "--router-url",
                "http://router.example",
                "--scope",
                "policy",
            ]
        )
        with (
            mock.patch.object(
                loop, "load_probe_manifest", return_value=(manifest, [probe, second])
            ),
            mock.patch.object(loop, "evaluate_probes", return_value=evidence) as run,
            mock.patch("builtins.print"),
        ):
            self.assertEqual(loop.cmd_eval(args), 0)
        self.assertEqual(run.call_args.kwargs["scope"], "policy")

    def test_consolidation_labels_policy_success_without_deployment_success(self):
        evidence = self.run_scope("policy")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inventory.json").write_text(
                json.dumps({"recipes": [{"name": "example"}]})
            )
            (root / "example").mkdir()
            (root / "example/eval-report.json").write_text(
                json.dumps({"evaluation": evidence})
            )
            combined = consolidated.build_consolidated_report(root)
        self.assertTrue(combined["summary"]["passed"])
        self.assertEqual(combined["summary"]["evaluation_scopes"], ["policy"])
        self.assertFalse(combined["summary"]["deployment_passed"])
        self.assertEqual(
            combined["results"][0]["selection_status_counts"], {"unavailable": 1}
        )
        self.assertEqual(
            combined["results"][0]["selection_reasons"], evidence["selection_reasons"]
        )


if __name__ == "__main__":
    unittest.main()
