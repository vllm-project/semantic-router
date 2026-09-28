import contextlib
import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
harness = importlib.import_module("harness")
support = importlib.import_module("check_support")


class CIStaticChecksTests(unittest.TestCase):
    def run_checks(self, static_only: bool) -> dict[str, mock.Mock]:
        names = (
            "run_test_commands",
            "run_precommit",
            "run_python_lint",
            "run_go_lint",
            "run_rust_lint",
            "run_reference_config_lint",
        )
        with contextlib.ExitStack() as stack:
            calls = {
                name: stack.enter_context(
                    mock.patch.object(harness, name, return_value=0)
                )
                for name in names
            }
            self.assertEqual(
                harness.run_check(["README.md"], None, ci_static_only=static_only), 0
            )
        return calls

    def test_ci_runs_lint_without_domain_or_reference_tests(self) -> None:
        calls = self.run_checks(True)
        calls["run_reference_config_lint"].assert_not_called()
        self.assertEqual(calls["run_test_commands"].call_count, 1)
        calls["run_precommit"].assert_called_once_with(
            ["README.md"], None, ci_static_only=True
        )
        for name in ("run_python_lint", "run_go_lint", "run_rust_lint"):
            calls[name].assert_called_once()

    def test_local_checks_keep_domain_and_reference_tests(self) -> None:
        calls = self.run_checks(False)
        calls["run_reference_config_lint"].assert_called_once()
        self.assertEqual(calls["run_test_commands"].call_count, 2)

    def test_ci_hook_ownership_preserves_requested_skips(self) -> None:
        with mock.patch.dict(os.environ, {"SKIP": "shellcheck"}), mock.patch.object(
            support.subprocess, "run", return_value=mock.Mock(returncode=0)
        ) as run:
            support.run_precommit(["README.md"], "origin/main", ci_static_only=True)
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["BASE_REF"], "origin/main")
        skips = set(env["SKIP"].split(","))
        self.assertIn("shellcheck", skips)
        self.assertIn("supply-chain-security-scan", skips)
        self.assertNotIn("architecture-check", skips)

    def test_local_security_hook_remains_enabled(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            support.subprocess, "run", return_value=mock.Mock(returncode=0)
        ) as run:
            support.run_precommit(["README.md"], None)
        self.assertNotIn("SKIP", run.call_args.kwargs["env"])


if __name__ == "__main__":
    unittest.main()
