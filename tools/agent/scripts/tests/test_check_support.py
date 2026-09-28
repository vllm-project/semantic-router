import contextlib
import importlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

check_support = importlib.import_module("check_support")


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


class RustClippyTests(unittest.TestCase):
    def test_compiler_failure_outside_changed_files_is_not_ignored(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".clippy-exit-test-", dir=check_support.REPO_ROOT
        ) as directory:
            crate = Path(directory)
            (crate / "src").mkdir()
            (crate / "Cargo.toml").write_text(
                '[package]\nname = "clippy_exit_test"\n'
                'version = "0.1.0"\nedition = "2021"\n[workspace]\n'
            )
            (crate / "src/lib.rs").write_text("pub mod changed;\npub mod other;\n")
            changed = crate / "src/changed.rs"
            changed.write_text("pub fn value() -> i32 { 42 }\n")
            other = crate / "src/other.rs"
            other.write_text("pub fn value() -> i32 { missing_symbol() }\n")

            with mock.patch.dict(
                os.environ,
                {
                    "CARGO_NET_OFFLINE": "true",
                    "CARGO_BUILD_JOBS": "1",
                    "CARGO_TARGET_DIR": str(crate / "target"),
                },
            ):
                compiler: subprocess.CompletedProcess[str] = subprocess.run(
                    args=check_support.build_rust_clippy_command(crate_root=crate),
                    cwd=crate,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(compiler.returncode, 0)
                self.assertIn("E0425", compiler.stdout)
                self.assertIn("src/other.rs", compiler.stdout)

                errors = io.StringIO()
                with contextlib.redirect_stdout(
                    io.StringIO()
                ), contextlib.redirect_stderr(errors):
                    result: int = check_support.run_rust_clippy_for_crate(
                        crate_root=crate, changed_paths={changed.resolve()}
                    )
                print(f"invalid crate: cargo={compiler.returncode}, check={result}")
                self.assertEqual(result, compiler.returncode, msg=errors.getvalue())
                self.assertIn("E0425", errors.getvalue())

                other.write_text("pub fn value() -> i32 { 7 }\n")
                with contextlib.redirect_stdout(
                    io.StringIO()
                ), contextlib.redirect_stderr(io.StringIO()):
                    result = check_support.run_rust_clippy_for_crate(
                        crate_root=crate, changed_paths={changed.resolve()}
                    )
                print(f"valid crate: check={result}")
                self.assertEqual(result, 0)

    def test_tool_failure_without_compiler_messages_preserves_status(self) -> None:
        for stdout in (
            "",
            "cargo failed before emitting JSON\n",
            '{"reason":"build-finished","success":false}\n',
        ):
            with self.subTest(stdout=stdout):
                completed: subprocess.CompletedProcess[str] = (
                    subprocess.CompletedProcess(
                        args=[], returncode=17, stdout=stdout, stderr="cargo failed\n"
                    )
                )
                errors = io.StringIO()
                with mock.patch.object(
                    check_support.subprocess, "run", return_value=completed
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ), contextlib.redirect_stderr(
                    errors
                ):
                    result: int = check_support.run_rust_clippy_for_crate(
                        crate_root=check_support.REPO_ROOT / "candle-binding",
                        changed_paths=set(),
                    )
                self.assertEqual(result, 17)
                self.assertIn("cargo failed", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
