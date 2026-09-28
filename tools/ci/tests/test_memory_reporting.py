"""Completion checks use actual unittest events, including missing and skipped cases."""

import importlib.util
import io
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "memory_reporting", ROOT / "e2e/testing/memory_tests/reporting.py"
)
reporting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporting)


class MemoryReportingTests(unittest.TestCase):
    def run_case(self, *, skip=False, required=True):
        class Case(unittest.TestCase):
            def runTest(self):  # noqa: N802 - unittest callback
                if skip:
                    self.skipTest("dependency unavailable")

        suite = unittest.TestSuite([Case()])
        inventory = reporting.test_ids(suite)
        result = unittest.TextTestRunner(
            stream=io.StringIO(), resultclass=reporting.InventoryResult
        ).run(suite)
        return (
            result,
            inventory,
            reporting.summarize_result(result, inventory, required=required),
        )

    def test_pass_records_executed_inventory(self):
        _, inventory, report = self.run_case()
        self.assertTrue(report["successful"])
        self.assertEqual(report["executed"], inventory)
        self.assertEqual(report["passed"], 1)

    def test_required_skip_fails_and_never_counts_as_pass(self):
        _, _, report = self.run_case(skip=True)
        self.assertFalse(report["successful"])
        self.assertEqual(report["passed"], 0)
        self.assertEqual(report["skipped"], 1)

    def test_optional_skip_is_reported(self):
        _, _, report = self.run_case(skip=True, required=False)
        self.assertTrue(report["successful"])
        self.assertEqual(report["passed"], 0)

    def test_missing_duplicate_or_zero_inventory_fails(self):
        result, inventory, _ = self.run_case()
        for expected in ([], inventory * 2, [*inventory, "missing"]):
            with self.subTest(expected=expected):
                report = reporting.summarize_result(result, expected, required=True)
                self.assertFalse(report["successful"])

    def test_failed_subtest_preserves_parent_failure_evidence(self):
        class Case(unittest.TestCase):
            def runTest(self):  # noqa: N802 - unittest callback
                with self.subTest(stage="storage"):
                    self.fail("write was lost")

        suite = unittest.TestSuite([Case()])
        inventory = reporting.test_ids(suite)
        result = unittest.TextTestRunner(
            stream=io.StringIO(), resultclass=reporting.InventoryResult
        ).run(suite)
        report = reporting.summarize_result(result, inventory, required=True)
        self.assertFalse(report["successful"])
        self.assertEqual(report["cases"], [{"id": inventory[0], "status": "failed"}])
