"""Runtime and CLI evaluation tests for router calibration support."""

import importlib
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

router_calibration_support = importlib.import_module("router_calibration_support")
router_calibration_manifest = importlib.import_module("router_calibration_manifest")
router_calibration_loop = importlib.import_module("router_calibration_loop")


class RecipeScopedProbeRuntimeTest(unittest.TestCase):
    def test_evaluate_probes_rejects_unknown_selected_id_before_network(self) -> None:
        probe = router_calibration_manifest.Probe(
            decision_id="direct",
            variant_id="baseline",
            probe_id="direct:baseline",
            expected_decision="direct",
            query="probe direct",
        )
        with (
            mock.patch.object(router_calibration_support, "http_json") as http_json,
            self.assertRaisesRegex(ValueError, "unknown probe IDs: missing:probe"),
        ):
            router_calibration_support.evaluate_probes(
                "http://router.example:8080",
                [probe],
                selected_probe_ids=["missing:probe"],
            )
        http_json.assert_not_called()

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
            return 200, {
                "recipe": "default",
                "routing_decision": "direct",
                "eval_trace": [{"decision_name": "direct", "matched": True}],
                "decision_result": {},
            }

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

    def test_evaluate_probe_enforces_alias_and_recipe_trace(self) -> None:
        probe = router_calibration_manifest.Probe(
            decision_id="private",
            variant_id="baseline",
            probe_id="private:baseline",
            expected_decision="private_route",
            expected_recipe="privacy",
            expected_alias="local/private",
            query="Keep this local.",
        )
        response = {
            "recipe": probe.expected_recipe,
            "routing_decision": probe.expected_decision,
            "recommended_models": ["cloud/frontier"],
            "eval_trace": [
                {"decision_name": probe.expected_decision, "matched": True},
                {"decision_name": "foreign_route", "matched": False},
            ],
            "decision_result": {},
        }
        with mock.patch.object(
            router_calibration_support,
            "http_json",
            return_value=(200, response),
        ):
            result = router_calibration_support.evaluate_probe(
                "http://router.example:8080",
                probe,
                allowed_decisions=frozenset({probe.expected_decision}),
            )

        self.assertFalse(result["matched"])
        self.assertFalse(result["alias_matched"])
        self.assertFalse(result["trace_matched"])
        self.assertIn("foreign_route", result["trace_decisions"])

    def test_eval_command_returns_nonzero_when_acceptance_fails(self) -> None:
        args = SimpleNamespace(
            probes="probes.yaml",
            router_url="http://router.example:8080",
            output=None,
            probe_ids=["direct:baseline"],
        )
        with (
            mock.patch.object(
                router_calibration_loop,
                "load_probe_manifest",
                return_value=({"schema_version": "v1"}, []),
            ),
            mock.patch.object(
                router_calibration_loop,
                "evaluate_probes",
                return_value={"passed": False},
            ),
            mock.patch("builtins.print"),
        ):
            self.assertEqual(router_calibration_loop.cmd_eval(args), 1)
            router_calibration_loop.evaluate_probes.assert_called_once_with(
                "http://router.example:8080",
                [],
                {"schema_version": "v1"},
                selected_probe_ids=["direct:baseline"],
            )

    def test_eval_parser_preserves_repeated_probe_id_order(self) -> None:
        args = router_calibration_loop.build_parser().parse_args(
            [
                "eval",
                "--router-url",
                "http://router.example:8080",
                "--probes",
                "probes.yaml",
                "--id",
                "workflow:tool_workstreams",
                "--id",
                "direct:baseline",
            ]
        )

        self.assertEqual(
            args.probe_ids,
            ["workflow:tool_workstreams", "direct:baseline"],
        )

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
