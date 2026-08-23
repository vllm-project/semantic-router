import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

router_calibration_support = importlib.import_module("router_calibration_support")
router_calibration_manifest = importlib.import_module("router_calibration_manifest")
router_calibration_loop = importlib.import_module("router_calibration_loop")
router_calibration_http = importlib.import_module("router_calibration_http")


class CalibrationHTTPTest(unittest.TestCase):
    def test_http_json_uses_management_bearer_without_exposing_it_in_payload(
        self,
    ) -> None:
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.getcode.return_value = 200
        response.read.return_value = b'{"status":"ok"}'

        with (
            mock.patch.dict(
                "os.environ",
                {router_calibration_http.MANAGEMENT_TOKEN_ENV: "private-token"},
            ),
            mock.patch.object(
                router_calibration_http.request,
                "urlopen",
                return_value=response,
            ) as urlopen,
        ):
            status, payload = router_calibration_http.http_json(
                "POST",
                "http://router.example:8080/api/v1/eval",
                {"text": "safe fixture"},
            )

        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok"})
        sent_request = urlopen.call_args.args[0]
        self.assertEqual(
            sent_request.get_header("Authorization"), "Bearer private-token"
        )
        self.assertNotIn(b"private-token", sent_request.data)


class RecipeScopedProbeTest(unittest.TestCase):
    def test_write_json_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "nested" / "report.json"

            router_calibration_support.write_json(output, {"passed": True})

            self.assertEqual(
                output.read_text(encoding="utf-8"), '{\n  "passed": true\n}\n'
            )

    def test_manifest_loads_routing_model_and_expected_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "probes.yaml"
            manifest_path.write_text(
                """
schema_version: v1
name: test
routing_assets:
  yaml: test.yaml
  dsl: test.dsl
coverage:
  min_signal_assertion_percent: 0
  min_projection_assertion_percent: 0
  min_algorithm_assertion_percent: 0
  min_plugin_assertion_percent: 0
  required_request_shapes: []
  min_tag_counts: {}
  min_tag_pass_rate: {}
decisions:
  - id: balanced
    expected_decision: unified_balance_route
    model: vllm-sr/mom-v1-blend
    expected_recipe: balanced
    expected_algorithm: multi_factor
    expected_plugins: [semantic-cache]
    variants:
      - id: baseline
        query: Summarize this plan.
        expected_signals:
          projection: [balanced_score]
        repeat: 3
        tools:
          - type: function
            function:
              name: search
""".lstrip(),
                encoding="utf-8",
            )

            _, probes = router_calibration_manifest.load_probe_manifest(manifest_path)

        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0].model, "vllm-sr/mom-v1-blend")
        self.assertEqual(probes[0].expected_recipe, "balanced")
        self.assertEqual(probes[0].expected_algorithm, "multi_factor")
        self.assertEqual(probes[0].expected_plugins, ("semantic-cache",))
        self.assertEqual(
            probes[0].expected_signals, (("projection", "balanced_score"),)
        )
        self.assertEqual(probes[0].repeat, 3)
        self.assertEqual(probes[0].tools[0]["function"]["name"], "search")

    def test_manifest_padding_places_one_trigger_in_long_input(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "probes.yaml"
            manifest_path.write_text(
                """
schema_version: v1
name: test
routing_assets:
  yaml: test.yaml
  dsl: test.dsl
coverage:
  min_signal_assertion_percent: 0
  min_projection_assertion_percent: 0
  min_algorithm_assertion_percent: 0
  min_plugin_assertion_percent: 0
  required_request_shapes: []
  min_tag_counts: {}
  min_tag_pass_rate: {}
decisions:
  - id: privacy
    expected_decision: sensitive
    variants:
      - id: pii_at_tail
        query: alice@example.com
        padding:
          text: benign context
          repeat: 3
          placement: before
""".lstrip(),
                encoding="utf-8",
            )
            _, probes = router_calibration_manifest.load_probe_manifest(manifest_path)

        probe = probes[0]
        self.assertEqual(probe.padding.repeat, 3)
        self.assertEqual(
            router_calibration_support.materialize_probe_text(probe),
            "benign context\nbenign context\nbenign context\nalice@example.com",
        )

    def test_manifest_loads_display_prompt_and_disabled_playground_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "probes.yaml"
            manifest_path.write_text(
                """
schema_version: v1
name: test
routing_assets:
  yaml: test.yaml
  dsl: test.dsl
coverage:
  min_signal_assertion_percent: 0
  min_projection_assertion_percent: 0
  min_algorithm_assertion_percent: 0
  min_plugin_assertion_percent: 0
  required_request_shapes: []
  min_tag_counts: {}
  min_tag_pass_rate: {}
decisions:
  - id: long_context
    expected_decision: long_context
    variants:
      - id: synthetic_boundary
        query: Analyze the attached architecture notes.
        display_prompt: Analyze this long architecture document and identify its three highest-risk design decisions.
        playground:
          enabled: false
          reason: This synthetic boundary fixture is intentionally expanded only by Eval.
""".lstrip(),
                encoding="utf-8",
            )
            _, probes = router_calibration_manifest.load_probe_manifest(manifest_path)

        probe = probes[0]
        self.assertEqual(
            probe.display_prompt,
            "Analyze this long architecture document and identify its three highest-risk design decisions.",
        )
        self.assertFalse(probe.playground.enabled)
        self.assertEqual(
            probe.playground.reason,
            "This synthetic boundary fixture is intentionally expanded only by Eval.",
        )

    def test_manifest_rejects_disabled_playground_without_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "probes.yaml"
            manifest_path.write_text(
                """
schema_version: v1
name: test
routing_assets:
  yaml: test.yaml
  dsl: test.dsl
coverage:
  min_signal_assertion_percent: 0
  min_projection_assertion_percent: 0
  min_algorithm_assertion_percent: 0
  min_plugin_assertion_percent: 0
  required_request_shapes: []
  min_tag_counts: {}
  min_tag_pass_rate: {}
decisions:
  - id: long_context
    expected_decision: long_context
    variants:
      - id: invalid
        query: Analyze this synthetic long-context fixture.
        playground:
          enabled: false
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "playground.*reason"):
                router_calibration_manifest.load_probe_manifest(manifest_path)

    def test_manifest_rejects_disabled_playground_without_display_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "probes.yaml"
            manifest_path.write_text(
                """
schema_version: v1
name: test
routing_assets:
  yaml: test.yaml
  dsl: test.dsl
coverage:
  min_signal_assertion_percent: 0
  min_projection_assertion_percent: 0
  min_algorithm_assertion_percent: 0
  min_plugin_assertion_percent: 0
  required_request_shapes: []
  min_tag_counts: {}
  min_tag_pass_rate: {}
decisions:
  - id: long_context
    expected_decision: long_context
    variants:
      - id: invalid
        query: Analyze this synthetic long-context fixture.
        playground:
          enabled: false
          reason: This synthetic boundary fixture is intentionally expanded only by Eval.
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "display_prompt"):
                router_calibration_manifest.load_probe_manifest(manifest_path)

    def test_manifest_rejects_unknown_padding_placement(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "probes.yaml"
            manifest_path.write_text(
                """
schema_version: v1
name: test
routing_assets:
  yaml: test.yaml
  dsl: test.dsl
coverage:
  min_signal_assertion_percent: 0
  min_projection_assertion_percent: 0
  min_algorithm_assertion_percent: 0
  min_plugin_assertion_percent: 0
  required_request_shapes: []
  min_tag_counts: {}
  min_tag_pass_rate: {}
decisions:
  - id: privacy
    variants:
      - id: invalid
        query: alice@example.com
        padding:
          text: benign context
          placement: sideways
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "padding.placement"):
                router_calibration_manifest.load_probe_manifest(manifest_path)

    def test_evaluate_probe_requires_model_recipe_and_decision(self) -> None:
        probe = router_calibration_manifest.Probe(
            decision_id="balanced",
            variant_id="baseline",
            probe_id="balanced:baseline",
            expected_decision="unified_balance_route",
            model="vllm-sr/mom-v1-blend",
            expected_recipe="balanced",
            expected_algorithm="multi_factor",
            expected_plugins=("semantic-cache",),
            expected_signals=(("projection", "balanced_score"),),
            query="Summarize this plan.",
            display_prompt=(
                "Summarize this deployment plan and call out its two largest risks."
            ),
            playground=router_calibration_manifest.ProbePlaygroundPolicy(
                enabled=False,
                reason="This fixture is Eval-only.",
            ),
        )
        response = {
            "requested_model": probe.model,
            "selected_model": "backend/final",
            "selection_status": "selected",
            "selection_method": "multi_factor",
            "recommended_models": ["backend/final"],
            "signal_errors": {},
            "recipe": probe.expected_recipe,
            "routing_decision": probe.expected_decision,
            "eval_trace": [
                {
                    "decision_name": probe.expected_decision,
                    "matched": True,
                }
            ],
            "decision_result": {
                "algorithm": probe.expected_algorithm,
                "plugins": ["semantic-cache"],
                "matched_signals": {"projection": ["balanced_score"]},
            },
        }

        with mock.patch.object(
            router_calibration_support,
            "http_json",
            return_value=(200, response),
        ) as http_json:
            result = router_calibration_support.evaluate_probe(
                "http://router.example:8080", probe
            )

        self.assertTrue(result["matched"])
        self.assertEqual(result["selected_model"], "backend/final")
        self.assertEqual(result["selection_status"], "selected")
        self.assertEqual(result["selection_method"], "multi_factor")
        self.assertEqual(result["signal_errors"], {})
        self.assertEqual(
            result["display_prompt"],
            "Summarize this deployment plan and call out its two largest risks.",
        )
        self.assertEqual(
            result["playground"],
            {"enabled": False, "reason": "This fixture is Eval-only."},
        )
        http_json.assert_called_once_with(
            "POST",
            "http://router.example:8080/api/v1/eval?trace=true",
            {"text": probe.query, "model": probe.model},
            timeout_seconds=60.0,
        )

    def test_evaluate_probe_rejects_missing_signal_or_plugin(self) -> None:
        probe = router_calibration_manifest.Probe(
            decision_id="privacy",
            variant_id="pii",
            probe_id="privacy:pii",
            expected_decision="local_sensitive",
            expected_plugins=("tools",),
            expected_signals=(("pii", "pii_strict"),),
            query="My SSN is 123-45-6789.",
        )
        with mock.patch.object(
            router_calibration_support,
            "http_json",
            return_value=(
                200,
                {
                    "routing_decision": probe.expected_decision,
                    "eval_trace": [
                        {
                            "decision_name": probe.expected_decision,
                            "matched": True,
                        }
                    ],
                    "decision_result": {
                        "plugins": [],
                        "matched_signals": {"kb": ["privacy_policy"]},
                    },
                },
            ),
        ):
            result = router_calibration_support.evaluate_probe(
                "http://router.example:8080", probe
            )

        self.assertFalse(result["matched"])
        self.assertFalse(result["plugins_matched"])
        self.assertFalse(result["signals_matched"])
        self.assertEqual(result["missing_expected_signals"], ["pii:pii_strict"])

    def test_evaluate_probe_rejects_degraded_signal_execution(self) -> None:
        probe = router_calibration_manifest.Probe(
            decision_id="direct",
            variant_id="baseline",
            probe_id="direct:baseline",
            expected_decision="direct",
            query="Give one direct answer.",
        )
        with mock.patch.object(
            router_calibration_support,
            "http_json",
            return_value=(
                200,
                {
                    "recipe": "default",
                    "routing_decision": "direct",
                    "signal_errors": {"embedding": "backend unavailable"},
                    "eval_trace": [
                        {"decision_name": "direct", "matched": True},
                    ],
                    "decision_result": {},
                },
            ),
        ):
            result = router_calibration_support.evaluate_probe(
                "http://router.example:8080", probe
            )

        self.assertFalse(result["matched"])
        self.assertFalse(result["signal_errors_matched"])
        self.assertEqual(result["signal_errors"], {"embedding": "backend unavailable"})

    def test_eval_selection_contract_rejects_fabricated_or_missing_final_model(
        self,
    ) -> None:
        cases = [
            (
                {"algorithm": "static", "status": "selected", "selected": ""},
                "selected_model is required",
            ),
            (
                {
                    "algorithm": "confidence",
                    "status": "execution_required",
                    "selected": "fabricated-model",
                },
                "must not fabricate",
            ),
            (
                {
                    "algorithm": "fusion",
                    "status": "selected",
                    "selected": "judge-model",
                },
                "planned_final",
            ),
        ]
        for case, expected_error in cases:
            with self.subTest(case=case):
                comparison = router_calibration_support.compare_eval_selection(
                    algorithm=case["algorithm"],
                    selected_model=case["selected"],
                    status=case["status"],
                    method=case["algorithm"],
                    recommended_models=("candidate-a",),
                )
                self.assertFalse(comparison["matched"])
                self.assertTrue(
                    any(expected_error in error for error in comparison["errors"]),
                    comparison,
                )

    def test_eval_selection_contract_accepts_static_single_and_planned_final(
        self,
    ) -> None:
        selected = router_calibration_support.compare_eval_selection(
            algorithm="static",
            selected_model="candidate-a",
            status="selected",
            method="single",
            recommended_models=("candidate-a",),
        )
        planned = router_calibration_support.compare_eval_selection(
            algorithm="workflows",
            selected_model="final-model",
            status="planned_final",
            method="workflows",
            recommended_models=("worker-a",),
        )

        self.assertTrue(selected["matched"], selected)
        self.assertTrue(planned["matched"], planned)

    def test_eval_selection_contract_accepts_honest_execution_deferral(self) -> None:
        for algorithm in (
            "static",
            "multi_factor",
            "latency_aware",
            "workflows",
            "fusion",
            "remom",
        ):
            with self.subTest(algorithm=algorithm):
                deferred = router_calibration_support.compare_eval_selection(
                    algorithm=algorithm,
                    selected_model="",
                    status="execution_required",
                    method=algorithm,
                    recommended_models=("candidate-a", "candidate-b"),
                )
                self.assertTrue(deferred["matched"], deferred)

    def test_evaluate_probes_records_failure_and_continues(self) -> None:
        probes = [
            router_calibration_manifest.Probe(
                decision_id="balanced",
                variant_id=variant,
                probe_id=f"balanced:{variant}",
                expected_decision="unified_balance_route",
                query=f"probe {variant}",
            )
            for variant in ("timeout", "healthy")
        ]
        healthy_response = {
            "recipe": "default",
            "routing_decision": "unified_balance_route",
            "eval_trace": [
                {
                    "decision_name": "unified_balance_route",
                    "matched": True,
                }
            ],
            "decision_result": {},
        }
        with mock.patch.object(
            router_calibration_support,
            "http_json",
            side_effect=[RuntimeError("request timed out"), (200, healthy_response)],
        ) as http_json:
            report = router_calibration_support.evaluate_probes(
                "http://router.example:8080",
                probes,
                {"evaluation": {"request_timeout_seconds": 90}},
            )

        self.assertEqual(http_json.call_count, 2)
        self.assertEqual(report["matched"], 1)
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["request_timeout_seconds"], 90)
        self.assertEqual(report["results"][0]["error"], "request timed out")
        self.assertTrue(report["results"][1]["matched"])

    def test_http_transport_normalizes_remote_disconnect(self) -> None:
        with (
            mock.patch(
                "urllib.request.urlopen",
                side_effect=ConnectionResetError("remote closed connection"),
            ),
            self.assertRaisesRegex(RuntimeError, "remote closed connection"),
        ):
            router_calibration_support.http_json(
                "POST",
                "http://router.example:8080/api/v1/eval",
                {"text": "probe"},
            )

    def test_evaluate_probes_scopes_trace_inventory_by_recipe(self) -> None:
        probes = [
            router_calibration_manifest.Probe(
                decision_id=recipe,
                variant_id="baseline",
                probe_id=f"{recipe}:baseline",
                expected_decision=f"{recipe}_route",
                expected_recipe=recipe,
                query=f"probe {recipe}",
            )
            for recipe in ("balanced", "privacy")
        ]
        responses = [
            (
                200,
                {
                    "recipe": recipe,
                    "routing_decision": f"{recipe}_route",
                    "eval_trace": [
                        {
                            "decision_name": f"{recipe}_route",
                            "matched": True,
                        }
                    ],
                    "decision_result": {},
                },
            )
            for recipe in ("balanced", "privacy")
        ]
        with mock.patch.object(
            router_calibration_support,
            "http_json",
            side_effect=responses,
        ):
            report = router_calibration_support.evaluate_probes(
                "http://router.example:8080",
                probes,
                {"evaluation": {"concurrency": 1}},
            )

        self.assertTrue(report["passed"])
        self.assertTrue(all(result["trace_matched"] for result in report["results"]))

    def test_evaluate_probes_selects_ordered_ids_against_full_recipe_trace(
        self,
    ) -> None:
        probes = [
            router_calibration_manifest.Probe(
                decision_id=decision,
                variant_id="baseline",
                probe_id=f"{decision}:baseline",
                expected_decision=decision,
                expected_recipe="accuracy",
                query=f"probe {decision}",
            )
            for decision in ("direct", "workflow")
        ]
        response = {
            "recipe": "accuracy",
            "routing_decision": "workflow",
            "eval_trace": [
                {"decision_name": "direct", "matched": False},
                {"decision_name": "workflow", "matched": True},
            ],
            "decision_result": {},
        }
        with mock.patch.object(
            router_calibration_support,
            "http_json",
            return_value=(200, response),
        ) as http_json:
            report = router_calibration_support.evaluate_probes(
                "http://router.example:8080",
                probes,
                {"evaluation": {"concurrency": 1}},
                selected_probe_ids=["workflow:baseline"],
            )

        self.assertEqual(http_json.call_count, 1)
        self.assertTrue(report["passed"])
        self.assertEqual(
            [result["id"] for result in report["results"]],
            ["workflow:baseline"],
        )
        self.assertEqual(
            report["results"][0]["trace_decisions"], ["direct", "workflow"]
        )


if __name__ == "__main__":
    unittest.main()
