"""Observe existing unittest/pytest runners without changing discovery or assertions.

Loaded through an isolated sitecustomize for unittest and PYTEST_PLUGINS for
pytest. This adapter records framework collection before execution, including
skips; it never turns a command exit code into a fabricated passed test.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from contextlib import suppress
from pathlib import Path
from typing import override


def emit(kind: str, identity: str, status: str | None = None) -> None:
    destination = os.environ.get("CI_PYTHON_TEST_EVENTS")
    if not destination:
        return
    row = {"kind": kind, "id": identity, "pid": os.getpid()}
    if status:
        row["status"] = status
    with Path(destination).open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def test_id(test) -> str:
    module = sys.modules.get(type(test).__module__)
    filename = getattr(module, "__file__", "")
    root = Path(os.environ["CI_REPO_ROOT"])
    with suppress(ValueError):
        filename = str(Path(filename).resolve().relative_to(root))
    return f"{filename}::{test.id()}"


_UNITTEST_STATE = {"depth": 0, "installed": False}


def observe_unittest() -> None:
    if _UNITTEST_STATE["installed"]:
        return
    _UNITTEST_STATE["installed"] = True
    original_run = unittest.TextTestRunner.run

    def run(runner, suite):
        if _UNITTEST_STATE["depth"] or "pytest" in sys.modules:
            return original_run(runner, suite)

        def collect(test):
            if isinstance(test, unittest.TestSuite):
                for child in test:
                    collect(child)
            else:
                emit("expected", test_id(test))

        collect(suite)
        original = runner.resultclass

        class ObservedResult(original):
            @override
            def addSuccess(self, test):
                super().addSuccess(test)
                emit("case", test_id(test), "passed")

            @override
            def addError(self, test, error):
                super().addError(test, error)
                emit("case", test_id(test), "failed")

            @override
            def addFailure(self, test, error):
                super().addFailure(test, error)
                emit("case", test_id(test), "failed")

            @override
            def addSkip(self, test, reason):
                super().addSkip(test, reason)
                emit("case", test_id(test), "skipped")

            @override
            def addExpectedFailure(self, test, error):
                super().addExpectedFailure(test, error)
                emit("case", test_id(test), "skipped")

            @override
            def addUnexpectedSuccess(self, test):
                super().addUnexpectedSuccess(test)
                emit("case", test_id(test), "failed")

            @override
            def addSubTest(self, test, subtest, error):
                super().addSubTest(test, subtest, error)
                if error:
                    emit("case", test_id(test), "failed")

        runner.resultclass = ObservedResult
        _UNITTEST_STATE["depth"] += 1
        try:
            return original_run(runner, suite)
        finally:
            _UNITTEST_STATE["depth"] -= 1
            runner.resultclass = original

    unittest.TextTestRunner.run = run


_PYTEST_IDENTITIES: dict[str, str] = {}


def pytest_collection_finish(session) -> None:
    for item in session.items:
        identity = str(item.path) + "::" + item.nodeid
        _PYTEST_IDENTITIES[item.nodeid] = identity
        emit("expected", identity)


def pytest_runtest_logreport(report) -> None:
    identity = _PYTEST_IDENTITIES[report.nodeid]
    if report.failed:
        emit("case", identity, "failed")
    elif report.skipped:
        emit("case", identity, "skipped")
    elif report.when == "call":
        emit("case", identity, "passed")


def summarize(path: Path) -> dict:
    expected: list[str] = []
    statuses: dict[str, str] = {}
    severity = {"passed": 0, "skipped": 1, "failed": 2}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["kind"] == "expected":
            expected.append(row["id"])
        elif row["kind"] == "case":
            prior = statuses.get(row["id"], "passed")
            statuses[row["id"]] = max((prior, row["status"]), key=severity.get)
    if len(expected) != len(set(expected)):
        raise ValueError("Python framework selected duplicate tests across runners")
    return {
        "runtime": "none",
        "device": "none",
        "platform": "linux/amd64",
        "expected_cases": expected,
        "discovered_cases": expected,
        "cases": [
            {"id": name, "status": status} for name, status in sorted(statuses.items())
        ],
    }


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(
        json.dumps(summarize(Path(sys.argv[1])), indent=2) + "\n"
    )
