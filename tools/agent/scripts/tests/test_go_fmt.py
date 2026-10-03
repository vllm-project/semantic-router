import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

go_fmt = importlib.import_module("go_fmt")


class GoFmtTests(unittest.TestCase):
    def run_with_mocked_lint(self, changed_files: list[str]) -> mock.Mock:
        with (
            mock.patch.object(
                go_fmt, "resolve_golangci_lint", return_value="golangci-lint"
            ),
            mock.patch.object(
                go_fmt.subprocess, "run", return_value=mock.Mock(returncode=0)
            ) as run,
        ):
            self.assertEqual(go_fmt.run_go_fmt(changed_files), 0)
        return run

    def test_module_local_files_format_from_their_module_root(self) -> None:
        run = self.run_with_mocked_lint(["e2e/testcases/decision_fallback.go"])

        run.assert_called_once()
        command, kwargs = run.call_args.args[0], run.call_args.kwargs
        self.assertEqual(kwargs["cwd"], go_fmt.REPO_ROOT / "e2e")
        self.assertEqual(command[-1], "testcases/decision_fallback.go")

    def test_tools_outside_module_roots_format_from_router_module(self) -> None:
        run = self.run_with_mocked_lint(
            [
                "tools/dev/dsl/main.go",
                "tools/calibration/image-routing/main.go",
                "bench/grounded_fusion/fusioneval/main.go",
            ]
        )

        run.assert_called_once()
        command, kwargs = run.call_args.args[0], run.call_args.kwargs
        self.assertEqual(kwargs["cwd"], go_fmt.ROUTER_MODULE_ROOT)
        self.assertEqual(
            command[-3:],
            [
                "../../bench/grounded_fusion/fusioneval/main.go",
                "../../tools/calibration/image-routing/main.go",
                "../../tools/dev/dsl/main.go",
            ],
        )

    def test_mixed_changes_group_by_module_and_router_fallback(self) -> None:
        run = self.run_with_mocked_lint(
            ["tools/dev/dsl/main.go", "src/semantic-router/cmd/main.go"]
        )

        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[-2:], ["../../tools/dev/dsl/main.go", "cmd/main.go"])

    def test_non_go_or_missing_files_are_ignored(self) -> None:
        run = self.run_with_mocked_lint(["README.md", "tools/dev/dsl/missing.go"])

        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
