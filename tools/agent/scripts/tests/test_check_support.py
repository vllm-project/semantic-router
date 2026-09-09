import importlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

check_support = importlib.import_module("check_support")


class RustLintFailureTests(unittest.TestCase):
    def test_infrastructure_failure_without_diagnostics_fails(self) -> None:
        result = subprocess.CompletedProcess(
            [], 101, "", "dependency download failed\n"
        )
        with mock.patch.object(check_support.subprocess, "run", return_value=result):
            self.assertEqual(
                check_support.run_rust_clippy_for_crate(Path("/crate"), set()), 101
            )

    def test_failed_build_with_other_file_diagnostic_still_fails(self) -> None:
        diagnostic = json.dumps(
            {
                "reason": "compiler-message",
                "message": {
                    "level": "error",
                    "spans": [{"file_name": "other.rs"}],
                    "rendered": "build failed",
                },
            }
        )
        result = subprocess.CompletedProcess([], 101, diagnostic, "")
        with mock.patch.object(check_support.subprocess, "run", return_value=result):
            self.assertEqual(
                check_support.run_rust_clippy_for_crate(
                    Path("/crate"), {Path("/crate/changed.rs")}
                ),
                101,
            )

    def test_successful_build_with_no_diagnostic_passes(self) -> None:
        result = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(check_support.subprocess, "run", return_value=result):
            self.assertEqual(
                check_support.run_rust_clippy_for_crate(Path("/crate"), set()), 0
            )


class ReferenceConfigLintTests(unittest.TestCase):
    def test_unrelated_change_skips_reference_config_test(self) -> None:
        with mock.patch.object(check_support.subprocess, "run") as run:
            result = check_support.run_reference_config_lint(["README.md"])

        self.assertEqual(result, 0)
        run.assert_not_called()

    def test_config_change_runs_reference_config_test(self) -> None:
        completed = mock.Mock(returncode=0)
        with mock.patch.object(
            check_support.subprocess,
            "run",
            return_value=completed,
        ) as run:
            result = check_support.run_reference_config_lint(
                ["src/semantic-router/pkg/config/config.go"]
            )

        self.assertEqual(result, 0)
        run.assert_called_once()

    def test_command_execution_does_not_use_a_shell(self) -> None:
        with mock.patch.object(check_support.subprocess, "run") as run:
            check_support.run_command("make test-semantic-router")

        run.assert_called_once_with(
            ["make", "test-semantic-router"],
            cwd=check_support.REPO_ROOT,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
