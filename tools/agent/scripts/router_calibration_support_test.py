import importlib
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

router_calibration_support = importlib.import_module("router_calibration_support")
router_calibration_manifest = importlib.import_module("router_calibration_manifest")


class DeployConfigTest(unittest.TestCase):
    def test_deploy_config_uses_put_replace_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            yaml_path = Path(tempdir) / "router.yaml"
            dsl_path = Path(tempdir) / "router.dsl"
            yaml_path.write_text("version: v0.3\n", encoding="utf-8")
            dsl_path.write_text('ROUTE fallback { MODEL "qwen" }\n', encoding="utf-8")

            with mock.patch.object(
                router_calibration_support,
                "http_json",
                return_value=(200, {"status": "success"}),
            ) as http_json:
                result = router_calibration_support.deploy_config(
                    "http://router.example:8080",
                    yaml_path,
                    dsl_path,
                )

            self.assertEqual(result, {"status": "success"})
            http_json.assert_called_once_with(
                "PUT",
                "http://router.example:8080/config/router",
                {
                    "yaml": "version: v0.3\n",
                    "dsl": 'ROUTE fallback { MODEL "qwen" }\n',
                },
            )

    def test_wait_for_config_activation_observes_exact_runtime_hash(self) -> None:
        with mock.patch.object(
            router_calibration_support,
            "http_json",
            side_effect=[
                (
                    200,
                    {
                        "status": "pending",
                        "runtime_hash": "next-runtime",
                        "active_hash": "old-runtime",
                    },
                ),
                (
                    200,
                    {
                        "status": "active",
                        "runtime_hash": "next-runtime",
                        "active_hash": "next-runtime",
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

        self.assertEqual(result["payload"]["status"], "active")
        self.assertEqual(http_json.call_count, 2)

    def test_wait_for_config_activation_rejects_superseded_deploy(self) -> None:
        with (
            mock.patch.object(
                router_calibration_support,
                "http_json",
                return_value=(
                    200,
                    {
                        "status": "pending",
                        "runtime_hash": "newer-runtime",
                        "active_hash": "old-runtime",
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
decisions:
  - id: balanced
    expected_decision: unified_balance_route
    model: vllm-sr/mom-balanced-v1
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
        self.assertEqual(probes[0].model, "vllm-sr/mom-balanced-v1")
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

    def test_manifest_rejects_unknown_padding_placement(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "probes.yaml"
            manifest_path.write_text(
                """
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
            model="vllm-sr/mom-balanced-v1",
            expected_recipe="balanced",
            expected_algorithm="multi_factor",
            expected_plugins=("semantic-cache",),
            expected_signals=(("projection", "balanced_score"),),
            query="Summarize this plan.",
        )
        response = {
            "requested_model": probe.model,
            "recipe": probe.expected_recipe,
            "routing_decision": probe.expected_decision,
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
        http_json.assert_called_once_with(
            "POST",
            "http://router.example:8080/api/v1/eval",
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
            "routing_decision": "unified_balance_route",
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

    def test_eval_request_timeout_is_bounded(self) -> None:
        self.assertEqual(
            router_calibration_support.resolve_eval_request_timeout({}), 60.0
        )
        self.assertEqual(
            router_calibration_support.resolve_eval_request_timeout(
                {"evaluation": {"request_timeout_seconds": "300"}}
            ),
            300.0,
        )
        with self.assertRaisesRegex(ValueError, "between 1 and 1200"):
            router_calibration_support.resolve_eval_request_timeout(
                {"evaluation": {"request_timeout_seconds": 0}}
            )

    def test_evaluate_probes_runs_with_bounded_concurrency_and_reports_latency(
        self,
    ) -> None:
        probes = [
            router_calibration_manifest.Probe(
                decision_id="direct",
                variant_id=str(index),
                probe_id=f"direct:{index}",
                expected_decision="direct",
                query=f"probe {index}",
            )
            for index in range(6)
        ]
        lock = threading.Lock()
        active = 0
        max_active = 0

        def fake_http_json(*args, **kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return 200, {"routing_decision": "direct", "decision_result": {}}

        with mock.patch.object(
            router_calibration_support, "http_json", side_effect=fake_http_json
        ):
            report = router_calibration_support.evaluate_probes(
                "http://router.example:8080",
                probes,
                {"evaluation": {"concurrency": 3}},
            )

        self.assertGreaterEqual(max_active, 2)
        self.assertEqual(report["performance"]["concurrency"], 3)
        self.assertEqual(report["performance"]["requests"], 6)
        self.assertEqual(report["performance"]["errors"], 0)
        self.assertGreater(report["performance"]["throughput_rps"], 0)
        self.assertGreater(report["performance"]["latency_ms"]["p95"], 0)
        self.assertTrue(all("latency_ms" in item for item in report["results"]))

    def test_eval_concurrency_is_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 64"):
            router_calibration_support.resolve_evaluation_settings(
                {"evaluation": {"concurrency": 65}}
            )

    def test_evaluate_probe_rejects_wrong_recipe(self) -> None:
        probe = router_calibration_manifest.Probe(
            decision_id="balanced",
            variant_id="baseline",
            probe_id="balanced:baseline",
            expected_decision="shared_decision",
            model="vllm-sr/mom-balanced-v1",
            expected_recipe="balanced",
            expected_algorithm="multi_factor",
            query="Summarize this plan.",
        )
        with mock.patch.object(
            router_calibration_support,
            "http_json",
            return_value=(
                200,
                {
                    "requested_model": probe.model,
                    "recipe": "another-recipe",
                    "routing_decision": probe.expected_decision,
                    "decision_result": {"algorithm": probe.expected_algorithm},
                },
            ),
        ):
            result = router_calibration_support.evaluate_probe(
                "http://router.example:8080", probe
            )

        self.assertFalse(result["matched"])
        self.assertFalse(result["recipe_matched"])

    def test_evaluate_probe_rejects_wrong_algorithm(self) -> None:
        probe = router_calibration_manifest.Probe(
            decision_id="frontier",
            variant_id="deliberate",
            probe_id="frontier:deliberate",
            expected_decision="frontier_remom",
            model="vllm-sr/mom-frontier-v1",
            expected_recipe="accuracy-first",
            expected_algorithm="remom",
            query="Explore several approaches and synthesize the strongest answer.",
        )
        with mock.patch.object(
            router_calibration_support,
            "http_json",
            return_value=(
                200,
                {
                    "requested_model": probe.model,
                    "recipe": probe.expected_recipe,
                    "routing_decision": probe.expected_decision,
                    "decision_result": {"algorithm": "static"},
                },
            ),
        ):
            result = router_calibration_support.evaluate_probe(
                "http://router.example:8080", probe
            )

        self.assertFalse(result["matched"])
        self.assertFalse(result["algorithm_matched"])

    def test_tag_summary_groups_cross_cutting_robustness_axes(self) -> None:
        summaries = router_calibration_manifest.summarize_tag_results(
            [
                {"id": "a", "matched": True, "tags": ["language:en", "negative"]},
                {"id": "b", "matched": False, "tags": ["language:en", "conflict"]},
                {"id": "c", "matched": True, "tags": ["language:zh", "conflict"]},
            ]
        )

        by_tag = {summary["tag"]: summary for summary in summaries}
        self.assertEqual(by_tag["language:en"]["pass_rate"], 50.0)
        self.assertFalse(by_tag["language:en"]["passed"])
        self.assertEqual(by_tag["language:zh"]["pass_rate"], 100.0)
        self.assertEqual(by_tag["conflict"]["failing_variants"], ["b"])


if __name__ == "__main__":
    unittest.main()
