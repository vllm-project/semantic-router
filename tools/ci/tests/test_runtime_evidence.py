import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime_evidence
from ci_results import collection_errors
from run_model_tests import required_inventory, runtime_families


class RuntimeEvidenceTests(unittest.TestCase):
    def report(self, provider="ort", **overrides):
        suites = {}
        for package, name in sorted(required_inventory(provider)):
            item = suites.setdefault(
                package,
                {
                    "package": package,
                    "expected": [],
                    "passed": [],
                    "failed": [],
                    "skipped": [],
                    "exit_code": 0,
                },
            )
            item["expected"].append(name)
            item["passed"].append(name)
        first = next(iter(suites.values()))
        first["skipped"].append(first["passed"].pop())
        return {
            "source_sha": "a" * 40,
            "suite": "runtime",
            "provider": provider,
            "device": "cpu",
            "models": [{"name": name} for name in runtime_families(provider)],
            "suites": list(suites.values()),
            **overrides,
        }

    def test_missing_and_skipped_case_is_preserved_for_gate(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                runtime_evidence.subprocess, "check_output", return_value="a" * 40
            ),
        ):
            path = Path(tmp)
            (path / "results.json").write_text(json.dumps(self.report()))
            result = runtime_evidence.native_evidence(path)
            self.assertEqual(
                len(result["expected_cases"]), len(required_inventory("ort"))
            )
            self.assertTrue(
                any(case["status"] == "skipped" for case in result["cases"])
            )
            self.assertTrue(collection_errors(result, "native"))

    def test_candle_requires_multimodal_report(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                runtime_evidence.subprocess, "check_output", return_value="a" * 40
            ),
        ):
            path = Path(tmp)
            (path / "results.json").write_text(
                json.dumps(self.report(provider="candle"))
            )
            with self.assertRaises(FileNotFoundError):
                runtime_evidence.native_evidence(path)

    def test_prior_commit_cannot_supply_inference_evidence(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                runtime_evidence.subprocess, "check_output", return_value="b" * 40
            ),
        ):
            path = Path(tmp)
            (path / "results.json").write_text(json.dumps(self.report()))
            with self.assertRaisesRegex(ValueError, "source"):
                runtime_evidence.native_evidence(path)

    def test_receipt_cannot_shorten_models_or_required_cases(self):
        for field in ("models", "suites"):
            with (
                tempfile.TemporaryDirectory() as tmp,
                patch.object(
                    runtime_evidence.subprocess, "check_output", return_value="a" * 40
                ),
            ):
                path = Path(tmp)
                report = self.report()
                report[field].pop()
                (path / "results.json").write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "inventory"):
                    runtime_evidence.native_evidence(path)

    def test_runtime_receipt_cannot_relabel_itself_as_smaller_multimodal_suite(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                runtime_evidence.subprocess, "check_output", return_value="a" * 40
            ),
        ):
            path = Path(tmp)
            (path / "results.json").write_text(
                json.dumps(self.report(provider="candle", suite="multimodal"))
            )
            with self.assertRaisesRegex(ValueError, "suite identity"):
                runtime_evidence.native_evidence(path)

    def test_recipe_uses_authored_acceptance_without_hiding_observations(self):
        evaluation = {
            "results": [{"id": "stress", "matched": False, "http_status": 200}],
            "passed": True,
            "execution": {"complete": True},
        }
        cases = runtime_evidence.recipe_cases(evaluation, "recipe:")
        self.assertFalse(cases[0]["matched"])
        evidence = {
            "cases": cases,
            "expected_cases": ["recipe:request:stress", "recipe:acceptance"],
        }
        self.assertEqual(collection_errors(evidence, "test"), [])
        evaluation["passed"] = False
        evidence["cases"] = runtime_evidence.recipe_cases(evaluation, "recipe:")
        self.assertTrue(collection_errors(evidence, "test"))

    def test_recipe_cannot_pass_with_failed_or_missing_requests(self):
        evaluation = {
            "results": [{"id": "one", "matched": False, "http_status": 503}],
            "passed": True,
            "execution": {"complete": True},
        }
        errors = collection_errors(
            {
                "cases": runtime_evidence.recipe_cases(evaluation, "recipe:"),
                "expected_cases": [
                    "recipe:request:one",
                    "recipe:request:two",
                    "recipe:acceptance",
                ],
            },
            "test",
        )
        self.assertTrue(any("failed" in error for error in errors))
        self.assertTrue(any("missing" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
