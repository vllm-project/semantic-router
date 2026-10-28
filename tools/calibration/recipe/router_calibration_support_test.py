import importlib
import json
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


class LocalValidationTest(unittest.TestCase):
    def test_validation_runs_relocated_dsl_from_router_module(self) -> None:
        with tempfile.TemporaryDirectory(prefix="calibration inputs ") as tempdir:
            yaml_path = Path(tempdir) / "router config.yaml"
            yaml_path.write_text("version: v0.3\n", encoding="utf-8")
            with mock.patch.object(
                router_calibration_support.subprocess,
                "run",
                return_value=mock.Mock(returncode=0, stdout="", stderr=""),
            ) as run:
                result = router_calibration_support.run_validate(None, yaml_path)

        self.assertTrue(result["valid"])
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[:2], ["go", "run"])
            sources = [Path(arg) for arg in command[2:] if arg.endswith(".go")]
            self.assertEqual(
                {path.name for path in sources},
                {"generate.go", "main.go", "test_runner.go"},
            )
            for source in sources:
                self.assertTrue(source.is_file())
                self.assertEqual(
                    source.parent,
                    router_calibration_support.REPO_ROOT / "tools" / "dev" / "dsl",
                )
            self.assertEqual(
                call.kwargs["cwd"],
                str(router_calibration_support.SEMANTIC_ROUTER_MODULE_ROOT),
            )
        self.assertEqual(run.call_args_list[0].args[0][-1], str(yaml_path))
        temporary_dsl = Path(run.call_args_list[1].args[0][-1])
        self.assertFalse(temporary_dsl.exists())


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
                "http://router.example:8080/api/v1/routing/preview",
                {"text": "safe fixture"},
            )

        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok"})
        sent_request = urlopen.call_args.args[0]
        self.assertEqual(
            sent_request.get_header("Authorization"), "Bearer private-token"
        )
        self.assertNotIn(b"private-token", sent_request.data)


class DeployConfigTest(unittest.TestCase):
    def test_deploy_config_plans_then_uses_cas_replace_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            yaml_path = Path(tempdir) / "router.yaml"
            yaml_path.write_text("version: v0.3\n", encoding="utf-8")

            with mock.patch.object(
                router_calibration_support,
                "http_json",
                side_effect=[
                    (200, {"current_etag": '"source-etag"'}),
                    (200, {"status": "success"}),
                ],
            ) as http_json:
                result = router_calibration_support.deploy_config(
                    "http://router.example:8080",
                    yaml_path,
                )

            self.assertEqual(result, {"status": "success"})
            self.assertEqual(
                http_json.call_args_list,
                [
                    mock.call(
                        "POST",
                        "http://router.example:8080/api/v1/config/plan",
                        {"yaml": "version: v0.3\n", "mode": "replace"},
                    ),
                    mock.call(
                        "PUT",
                        "http://router.example:8080/api/v1/config",
                        {"yaml": "version: v0.3\n"},
                        if_match='"source-etag"',
                    ),
                ],
            )

    def test_wait_for_config_activation_observes_exact_runtime_hash(self) -> None:
        with mock.patch.object(
            router_calibration_support,
            "http_json",
            side_effect=[
                (
                    200,
                    {
                        "activation_status": "pending",
                        "generated_runtime_hash": "next-runtime",
                        "active_runtime_hash": "old-runtime",
                    },
                ),
                (
                    200,
                    {
                        "activation_status": "active",
                        "generated_runtime_hash": "next-runtime",
                        "active_runtime_hash": "next-runtime",
                    },
                ),
            ],
        ) as http_json:
            result = router_calibration_support.wait_for_config_activation(
                "http://router.example:8080",
                "next-runtime",
                timeout_seconds=1,
                interval_seconds=0.001,
            )

        self.assertEqual(result["payload"]["activation_status"], "active")
        self.assertEqual(http_json.call_count, 2)

    def test_wait_for_config_activation_rejects_superseded_deploy(self) -> None:
        with (
            mock.patch.object(
                router_calibration_support,
                "http_json",
                return_value=(
                    200,
                    {
                        "activation_status": "pending",
                        "generated_runtime_hash": "newer-runtime",
                        "active_runtime_hash": "old-runtime",
                    },
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "superseded"),
        ):
            router_calibration_support.wait_for_config_activation(
                "http://router.example:8080",
                "expected-runtime",
                timeout_seconds=1,
                interval_seconds=0.001,
            )


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
        tool_choice: required
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
        self.assertEqual(probes[0].tool_choice, "required")

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
            tool_choice={"type": "function", "function": {"name": "search"}},
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
            "http://router.example:8080/api/v1/routing/preview?trace=true",
            {
                "text": probe.query,
                "model": probe.model,
                "tool_choice": probe.tool_choice,
            },
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

    def test_eval_selection_contract_accepts_single_candidate_and_planned_final(
        self,
    ) -> None:
        for algorithm in ("static", "multi_factor", "latency_aware"):
            with self.subTest(algorithm=algorithm):
                selected = router_calibration_support.compare_eval_selection(
                    algorithm=algorithm,
                    selected_model="candidate-a",
                    status="selected",
                    method="single",
                    recommended_models=("candidate-a",),
                )
                self.assertTrue(selected["matched"], selected)

        planned = router_calibration_support.compare_eval_selection(
            algorithm="workflows",
            selected_model="final-model",
            status="planned_final",
            method="workflows",
            recommended_models=("worker-a",),
        )

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


class SelectionExpectationTest(unittest.TestCase):
    def load_probes(self, default=None, override=None):
        decision = {
            "id": "long-context",
            "expected_decision": "long-context",
            "expected_algorithm": "static",
            "expected_recipe": "balance",
            "model": "vllm-sr/balance",
            "expected_signals": {"context": ["long"]},
            "variants": [
                {"id": "normal", "query": "Summarize this context."},
                {"id": "over-capacity", "query": "Summarize an oversized context."},
            ],
        }
        if default is not None:
            decision["expected_selection_status"] = default
        if override is not None:
            decision["variants"][1]["expected_selection_status"] = override
        document = {
            "schema_version": "v1",
            "name": "selection-contract",
            "routing_assets": {"yaml": "config.yaml", "dsl": "recipe.dsl"},
            "coverage": {
                "min_signal_assertion_percent": 0,
                "min_projection_assertion_percent": 0,
                "min_algorithm_assertion_percent": 0,
                "min_plugin_assertion_percent": 0,
                "required_request_shapes": [],
                "min_tag_counts": {},
                "min_tag_pass_rate": {},
            },
            "decisions": [decision],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probes.yaml"
            path.write_text(json.dumps(document), encoding="utf-8")
            return router_calibration_manifest.load_probe_manifest(path)[1]

    def response(self):
        return {
            "requested_model": "vllm-sr/balance",
            "recipe": "balance",
            "routing_decision": "long-context",
            "decision_result": {
                "decision_name": "long-context",
                "algorithm": "static",
                "matched_signals": {"context": ["long"]},
            },
            "eval_trace": [{"decision_name": "long-context", "matched": True}],
            "recommended_models": ["backend/context-model"],
            "selection_status": "unavailable",
            "selection_reason": "input exceeds all backend context capacities",
        }

    def evaluate(self, probe, response):
        with mock.patch.object(
            router_calibration_support, "http_json", return_value=(200, response)
        ):
            return router_calibration_support.evaluate_probe(
                "http://router.example:8080",
                probe,
                allowed_decisions=frozenset({"long-context"}),
            )

    def test_manifest_inherits_status_and_allows_one_variant_override(self):
        probes = self.load_probes("selected", "unavailable")
        self.assertEqual(
            [probe.expected_selection_status for probe in probes],
            ["selected", "unavailable"],
        )
        self.assertIsNone(self.load_probes()[0].expected_selection_status)

    def test_selection_status_enum_is_strict_at_both_manifest_levels(self):
        for invalid in ["", "anything", "unavailable ", 1, ["unavailable"]]:
            for field in ["decision", "variant"]:
                with (
                    self.subTest(invalid=invalid, field=field),
                    self.assertRaisesRegex(ValueError, "expected_selection_status"),
                ):
                    if field == "decision":
                        self.load_probes(invalid)
                    else:
                        self.load_probes(None, invalid)

    def test_explicit_capacity_negative_preserves_reason_and_other_assertions(self):
        probe = self.load_probes("selected", "unavailable")[1]
        response = self.response()
        result = self.evaluate(probe, response)
        self.assertTrue(result["matched"], result)
        self.assertEqual(result["expected_selection_status"], "unavailable")
        self.assertEqual(result["selection_status"], "unavailable")
        self.assertEqual(result["selected_model"], "")
        self.assertEqual(result["selection_reason"], response["selection_reason"])

        response["decision_result"]["matched_signals"] = {}
        result = self.evaluate(probe, response)
        self.assertTrue(result["selection_matched"])
        self.assertFalse(result["signals_matched"])
        self.assertFalse(result["matched"])

    def test_negative_requires_exact_status_empty_selection_and_explanation(self):
        probe = self.load_probes(None, "unavailable")[1]
        for change in [
            {
                "selection_status": "selected",
                "selected_model": "backend/context-model",
                "selection_method": "static",
            },
            {"selected_model": "backend/context-model"},
            {"selection_reason": ""},
        ]:
            with self.subTest(change=change):
                result = self.evaluate(probe, {**self.response(), **change})
                self.assertFalse(result["matched"])
                self.assertFalse(result["selection_matched"])
                self.assertTrue(result["selection_errors"])

    def test_default_positive_assertion_still_rejects_unavailable(self):
        probe = self.load_probes()[0]
        self.assertFalse(self.evaluate(probe, self.response())["selection_matched"])
        response = {
            **self.response(),
            "selection_status": "selected",
            "selection_method": "static",
            "selected_model": "backend/context-model",
        }
        self.assertTrue(self.evaluate(probe, response)["matched"])

    def test_negative_expectation_cannot_turn_an_http_failure_into_a_pass(self):
        probe = self.load_probes(None, "unavailable")[1]
        with mock.patch.object(
            router_calibration_support, "http_json", return_value=(503, self.response())
        ):
            report = router_calibration_support.evaluate_probes(
                "http://router.example:8080", [probe]
            )
        self.assertFalse(report["passed"])
        self.assertFalse(report["results"][0]["matched"])
        self.assertEqual(
            report["results"][0]["expected_selection_status"], "unavailable"
        )


class FailedProbeDiagnosticsTest(unittest.TestCase):
    def make_probe(self):
        return router_calibration_manifest.Probe(
            decision_id="sample",
            variant_id="neutral",
            probe_id="sample:neutral",
            expected_decision="sample",
            model="route-example",
            query="Sort these item names.",
            expected_recipe="example",
            expected_algorithm="static",
        )

    def test_http_failure_preserves_returned_diagnostics_without_passing(self):
        payload = {
            "requested_model": "route-example",
            "recipe": "example",
            "routing_decision": "sample",
            "selected_model": "backend-example",
            "selection_status": "selected",
            "selection_method": "static",
            "selection_reason": "Selection completed before a later failure",
            "recommended_models": ["backend-example"],
            "signal_errors": {"classifier:example": "classifier_evaluation_failed"},
            "signal_error_matches": {"classifier:example": True},
            "signal_confidences": {"keyword:example": 1.0},
            "signal_values": {"structure:items": 2},
            "applied_unknown_policies": {"sample": "fail_closed"},
            "decision_error": "classification was unavailable",
            "metrics": {"classifier": {"execution_time_ms": 12.5}},
            "decision_result": {
                "decision_name": "sample",
                "algorithm": "static",
                "plugins": ["header_mutation"],
                "used_signals": {"classifier": ["example"]},
                "matched_signals": {"keyword": ["example"]},
                "unmatched_signals": {"classifier": ["example"]},
            },
            "eval_trace": [{"decision_name": "sample", "matched": False}],
            # A failed response cannot override report status or expectations.
            "matched": True,
            "policy_matched": True,
            "deployment_matched": True,
            "expected_decision": "untrusted-override",
        }
        with mock.patch.object(
            router_calibration_support, "http_json", return_value=(503, payload)
        ):
            report = router_calibration_support.evaluate_probes(
                "http://router.invalid", [self.make_probe()], scope="policy"
            )
        result = report["results"][0]
        self.assertEqual(result["http_status"], 503)
        self.assertEqual(result["raw_response"], payload)
        for field in (
            "selected_model",
            "selection_status",
            "selection_method",
            "selection_reason",
            "recommended_models",
            "signal_errors",
            "signal_error_matches",
            "signal_confidences",
            "signal_values",
            "applied_unknown_policies",
            "decision_error",
            "metrics",
            "eval_trace",
        ):
            self.assertEqual(result[field], payload[field], field)
        for field, source in (
            ("actual_model", "requested_model"),
            ("actual_recipe", "recipe"),
            ("actual_decision", "routing_decision"),
        ):
            self.assertEqual(result[field], payload[source], field)
        for field, source in (
            ("actual_algorithm", "algorithm"),
            ("actual_plugins", "plugins"),
            ("used_signals", "used_signals"),
            ("matched_signals", "matched_signals"),
            ("unmatched_signals", "unmatched_signals"),
        ):
            self.assertEqual(result[field], payload["decision_result"][source], field)
        self.assertEqual(result["expected_decision"], "sample")
        self.assertIn("status 503", result["error"])
        self.assertNotIn("before model selection", result["selection_errors"][0])
        for field in (
            "matched",
            "policy_matched",
            "deployment_matched",
            "selection_matched",
        ):
            self.assertFalse(result[field], field)
        self.assertEqual(report["matched"], 0)
        self.assertFalse(report["passed"])
        self.assertFalse(report["scopes"]["policy"]["passed"])
        self.assertFalse(report["scopes"]["deployment"]["passed"])

    def test_partial_or_malformed_failure_does_not_invent_diagnostics(self):
        payloads = (
            None,
            "upstream unavailable",
            ["not an object"],
            {"signal_errors": [], "selected_model": 17, "decision_result": []},
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                exc = RuntimeError("request failed")
                exc.raw_response = payload
                result = router_calibration_support.failed_probe_result(
                    self.make_probe(), exc
                )
                self.assertEqual(result["raw_response"], payload)
                self.assertEqual(result["signal_errors"], {})
                self.assertEqual(result["selected_model"], "")
                self.assertEqual(result["actual_recipe"], "")
                self.assertEqual(result["actual_decision"], "")
                self.assertFalse(result["matched"])
                self.assertNotIn("decision_error", result)

    def test_nested_decision_identity_is_preserved_when_top_level_is_absent(self):
        exc = RuntimeError("request failed")
        exc.raw_response = {"decision_result": {"decision_name": "partial"}}
        result = router_calibration_support.failed_probe_result(self.make_probe(), exc)
        self.assertEqual(result["actual_decision"], "partial")
        self.assertEqual(result["selected_model"], "")
        self.assertFalse(result["matched"])


if __name__ == "__main__":
    unittest.main()
