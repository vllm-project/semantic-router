import importlib
import sys
import tempfile
import unittest
from http import HTTPStatus
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

waivers = importlib.import_module("recipe_conformance_waivers")
recipe_conformance = importlib.import_module("recipe_conformance")
router_calibration_manifest = importlib.import_module("router_calibration_manifest")
router_calibration_support = importlib.import_module("router_calibration_support")

NAMED = "casual_chat:long_unclassified_fallback"
TIMEOUT_BODY = {
    "error": {
        "code": "REQUEST_TIMEOUT",
        "message": "routing preview exceeded its request deadline",
    }
}


def failed(probe_id: str = NAMED, **fields) -> dict:
    return {
        "id": probe_id,
        "matched": False,
        "http_status": 504,
        "raw_response": TIMEOUT_BODY,
        "signal_errors": {},
        "latency_ms": 120_009.0,
        **fields,
    }


class RecipeCpuWaiverTest(unittest.TestCase):
    def test_only_the_preview_deadline_on_a_named_probe_is_waived(self) -> None:
        waiver = waivers.known_issue_waiver("standalone", "balance", failed())
        self.assertEqual(
            waiver,
            {
                "issue": 4706,
                "kind": "cpu-preview-deadline",
                "outcome": "request_timeout",
            },
        )
        cut = failed(
            http_status=200,
            raw_response={},
            signal_errors={"domain:business": "domain_evaluation_failed"},
            latency_ms=119_700.0,
        )
        self.assertEqual(
            waivers.known_issue_waiver("standalone", "balance", cut)["outcome"],
            "signals_cut_at_deadline",
        )
        still_failing = {
            "unnamed probe": ("standalone", "balance", failed("formal_math_proof:x")),
            "other recipe": ("standalone", "privacy", failed()),
            "unknown source": (None, "balance", failed()),
            "misroute": (
                "standalone",
                "balance",
                failed(http_status=200, raw_response={}, actual_decision="medium"),
            ),
            "other HTTP error": ("standalone", "balance", failed(http_status=503)),
            "other 504": (
                "standalone",
                "balance",
                failed(raw_response={"error": {"code": "UPSTREAM"}}),
            ),
            "signal error before the deadline": (
                "standalone",
                "balance",
                {**cut, "latency_ms": 60_000.0},
            ),
            "non-deadline signal error": (
                "standalone",
                "balance",
                {**cut, "signal_errors": {"pii:pii_strict": "pii_input_limit"}},
            ),
            "passed": ("standalone", "balance", {**failed(), "matched": True}),
        }
        for name, (source, recipe, result) in still_failing.items():
            with self.subTest(name=name):
                self.assertIsNone(waivers.known_issue_waiver(source, recipe, result))

    def test_named_probes_exist_and_are_long_inputs(self) -> None:
        root = recipe_conformance.DEFAULT_RECIPE_ROOT
        sources = {
            source.name: source
            for source in recipe_conformance.discover_recipe_sources(root)
        }
        for source_name, recipes in waivers.WAIVED_PROBES.items():
            for recipe, probe_ids in recipes.items():
                manifest = sources[source_name].recipes_root / recipe / "probes.yaml"
                _, probes = router_calibration_manifest.load_probe_manifest(manifest)
                by_id = {probe.probe_id: probe for probe in probes}
                for probe_id in probe_ids:
                    with self.subTest(recipe=recipe, probe=probe_id):
                        self.assertIn(probe_id, by_id)
                        self.assertIn("stress:long-input", by_id[probe_id].tags)
        self.assertEqual(
            len(waivers.waiver_policy()["cases"]),
            sum(
                len(ids)
                for recipes in waivers.WAIVED_PROBES.values()
                for ids in recipes.values()
            ),
        )

    def test_sources_resolve_only_for_maintained_roots(self) -> None:
        root = recipe_conformance.DEFAULT_RECIPE_ROOT
        self.assertEqual(waivers.recipe_source_name(root, root), "standalone")
        self.assertEqual(
            waivers.recipe_source_name(root / "built-in" / "latest", root),
            "built-in-latest",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            self.assertIsNone(waivers.recipe_source_name(Path(tempdir), root))

    def test_preview_deadline_follows_the_runtime_config(self) -> None:
        self.assertEqual(waivers.preview_deadline_seconds(None), 120.0)
        config = {
            "global": {
                "services": {
                    "api": {"routing_preview": {"request_timeout_seconds": 30}}
                }
            }
        }
        self.assertEqual(waivers.preview_deadline_seconds(config), 30.0)

    def test_waived_results_stay_reported_but_leave_acceptance(self) -> None:
        probes = [
            router_calibration_manifest.Probe(
                decision_id=decision,
                variant_id=variant,
                probe_id=f"{decision}:{variant}",
                expected_decision=decision,
                query=variant,
                tags=("stress:long-input",),
            )
            for decision, variant in (("casual_chat", "slow"), ("direct", "fast"))
        ]

        def fake_http_json(method, url, payload, timeout_seconds):
            if payload["text"] == "slow":
                return 504, TIMEOUT_BODY
            return 200, {
                "recipe": "default",
                "routing_decision": "direct",
                "eval_trace": [
                    {"decision_name": "casual_chat", "matched": False},
                    {"decision_name": "direct", "matched": True},
                ],
                "decision_result": {},
            }

        def waive(result):
            if (
                result["id"] == "casual_chat:slow"
                and result.get("http_status") == HTTPStatus.GATEWAY_TIMEOUT
            ):
                return {
                    "issue": 4706,
                    "kind": "cpu-preview-deadline",
                    "outcome": "request_timeout",
                }
            return None

        with mock.patch.object(
            router_calibration_support, "http_json", side_effect=fake_http_json
        ):
            unwaived = router_calibration_support.evaluate_probes(
                "http://router.example:8080", probes, {}
            )
            waived = router_calibration_support.evaluate_probes(
                "http://router.example:8080", probes, {}, waive=waive
            )

        self.assertFalse(unwaived["passed"])
        self.assertTrue(waived["passed"])
        self.assertEqual(
            (waived["matched"], waived["total"], waived["waived"]), (1, 1, 1)
        )
        self.assertEqual(len(waived["results"]), 2)
        self.assertEqual(
            [decision["decision_id"] for decision in waived["decisions"]], ["direct"]
        )
        slow = next(
            item for item in waived["results"] if item["id"] == "casual_chat:slow"
        )
        self.assertEqual(slow["known_issue_waiver"]["outcome"], "request_timeout")
        section = waivers.describe_waivers(waived["results"])
        self.assertEqual(
            [item["id"] for item in section["waived"]], ["casual_chat:slow"]
        )
        self.assertIn("#4706", section["removal"])


if __name__ == "__main__":
    unittest.main()
