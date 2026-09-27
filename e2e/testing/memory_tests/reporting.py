"""Inventory and completion evidence for the live memory integration suite."""

import unittest
from collections import Counter


def test_ids(suite):
    """Capture the discovered inventory before unittest consumes the suite."""
    result = []
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            result.extend(test_ids(test))
        else:
            result.append(test.id())
    return result


class InventoryResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed_ids = []
        self.case_statuses = {}

    def startTest(self, test):  # noqa: N802 - unittest callback
        self.executed_ids.append(test.id())
        self.case_statuses[test.id()] = "running"
        super().startTest(test)

    def addSuccess(self, test):  # noqa: N802 - unittest callback
        self.case_statuses[test.id()] = "passed"
        super().addSuccess(test)

    def addFailure(self, test, error):  # noqa: N802 - unittest callback
        self.case_statuses[test.id()] = "failed"
        super().addFailure(test, error)

    def addError(self, test, error):  # noqa: N802 - unittest callback
        self.case_statuses[test.id()] = "failed"
        super().addError(test, error)

    def addSkip(self, test, reason):  # noqa: N802 - unittest callback
        self.case_statuses[test.id()] = "skipped"
        super().addSkip(test, reason)

    def addExpectedFailure(self, test, error):  # noqa: N802 - unittest callback
        self.case_statuses[test.id()] = "skipped"
        super().addExpectedFailure(test, error)

    def addUnexpectedSuccess(self, test):  # noqa: N802 - unittest callback
        self.case_statuses[test.id()] = "failed"
        super().addUnexpectedSuccess(test)

    def addSubTest(self, test, subtest, error):  # noqa: N802 - unittest callback
        if error is not None:
            self.case_statuses[test.id()] = "failed"
        super().addSubTest(test, subtest, error)


def summarize_result(result, inventory, *, required):
    expected = Counter(inventory)
    executed = Counter(result.executed_ids)
    problems = []
    if not expected:
        problems.append("Memory case inventory is empty")
    if any(count != 1 for count in expected.values()):
        problems.append("Memory case inventory contains duplicates")
    if executed != expected:
        problems.append("Executed memory cases do not match the selected inventory")
    if required and (result.skipped or result.expectedFailures):
        problems.append("Required memory cases were skipped")
    skips = {test.id(): reason for test, reason in result.skipped}
    return {
        "inventory": inventory,
        "expected_cases": inventory,
        "cases": [
            {"id": name, "status": result.case_statuses[name]}
            for name in result.executed_ids
        ],
        "executed": result.executed_ids,
        "total": result.testsRun,
        "passed": sum(status == "passed" for status in result.case_statuses.values()),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "skip_reasons": skips,
        "inventory_errors": problems,
        "successful": result.wasSuccessful() and not problems,
    }
